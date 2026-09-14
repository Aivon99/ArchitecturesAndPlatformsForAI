#include "backend/gpu_tiled_backend.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

static void cuda_check(cudaError_t err, const char* ctx) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(ctx) + ": "
                                 + cudaGetErrorString(err));
}

constexpr int TILE = 16;

static __global__ void k_gemm_tiled(const float* X, const float* W,
                                     const float* bias, float* Y,
                                     int B, int in_dim, int out_dim)
{
    __shared__ float Xs[TILE][TILE];
    __shared__ float Ws[TILE][TILE]; // holds W^T tile: Ws[k][col] = W[col][k]

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.f;
    const int num_tiles = (in_dim + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        const int k_x = t * TILE + threadIdx.x;
        Xs[threadIdx.y][threadIdx.x] =
            (row < B && k_x < in_dim) ? X[row * in_dim + k_x] : 0.f;

        const int k_w = t * TILE + threadIdx.y;
        Ws[threadIdx.y][threadIdx.x] =
            (col < out_dim && k_w < in_dim) ? W[col * in_dim + k_w] : 0.f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += Xs[threadIdx.y][k] * Ws[k][threadIdx.x];

        __syncthreads();
    }

    if (row < B && col < out_dim)
        Y[row * out_dim + col] = acc + bias[col];
}

static __global__ void k_relu_forward(float* X, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) X[i] = X[i] > 0.f ? X[i] : 0.f;
}

static __global__ void k_fill(float* X, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) X[i] = val;
}

namespace Backend::GPUTiled {

void gemm(const Tensor& X, const Tensor& W,
          const Tensor& b, Tensor& Y)
{
    if (X.device != Device::CUDA || W.device != Device::CUDA ||
        b.device != Device::CUDA || Y.device != Device::CUDA)
        throw std::runtime_error("GPUTiled::gemm — all tensors must be on CUDA");

    const int B       = static_cast<int>(X.rows());
    const int in_dim  = static_cast<int>(X.cols());
    const int out_dim = static_cast<int>(W.rows());

    dim3 threads(TILE, TILE);
    dim3 blocks((out_dim + TILE - 1) / TILE, (B + TILE - 1) / TILE);

    k_gemm_tiled<<<blocks, threads>>>(X.data(), W.data(), b.data(), Y.data(),
                                       B, in_dim, out_dim);
    cuda_check(cudaGetLastError(), "k_gemm_tiled");
}

void relu_forward(Tensor& X) {
    int n = static_cast<int>(X.numel());
    k_relu_forward<<<(n + 255) / 256, 256>>>(X.data(), n);
    cuda_check(cudaGetLastError(), "k_relu_forward");
}

void fill(Tensor& X, float val) {
    int n = static_cast<int>(X.numel());
    k_fill<<<(n + 255) / 256, 256>>>(X.data(), val, n);
    cuda_check(cudaGetLastError(), "k_fill");
}

KernelLaunchInfo query_occupancy(int B, int in_dim, int out_dim) {
    (void)B; (void)in_dim; (void)out_dim;
    KernelLaunchInfo info;
    info.threads_per_block = TILE * TILE;
    info.shared_mem_bytes_per_block = 2 * TILE * TILE * sizeof(float);

    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &info.max_active_blocks_per_sm, k_gemm_tiled,
                   info.threads_per_block, info.shared_mem_bytes_per_block),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor (tiled)");

    int device = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    info.occupancy = static_cast<double>(info.max_active_blocks_per_sm) *
                      info.threads_per_block /
                      static_cast<double>(prop.maxThreadsPerMultiProcessor);

    return info;
}

} // namespace Backend::GPUTiled
