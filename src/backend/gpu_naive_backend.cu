#include "backend/gpu_naive_backend.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

static void cuda_check(cudaError_t err, const char* ctx) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(ctx) + ": "
                                 + cudaGetErrorString(err));
}

static __global__ void k_gemm_naive(const float* X, const float* W,
                                     const float* bias, float* Y,
                                     int B, int in_dim, int out_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * out_dim) return;

    int row = idx / out_dim;
    int col = idx % out_dim;

    float acc = 0.f;
    for (int k = 0; k < in_dim; ++k)
        acc += X[row * in_dim + k] * W[col * in_dim + k];

    Y[idx] = acc + bias[col];
}

static __global__ void k_relu_forward(float* X, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) X[i] = X[i] > 0.f ? X[i] : 0.f;
}

static __global__ void k_fill(float* X, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) X[i] = val;
}

namespace Backend::GPUNaive {

void gemm(const Tensor& X, const Tensor& W,
          const Tensor& b, Tensor& Y)
{
    if (X.device != Device::CUDA || W.device != Device::CUDA ||
        b.device != Device::CUDA || Y.device != Device::CUDA)
        throw std::runtime_error("GPUNaive::gemm — all tensors must be on CUDA");

    const int B       = static_cast<int>(X.rows());
    const int in_dim   = static_cast<int>(X.cols());
    const int out_dim  = static_cast<int>(W.rows());

    const int n = B * out_dim;
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    k_gemm_naive<<<blocks, threads>>>(X.data(), W.data(), b.data(), Y.data(),
                                       B, in_dim, out_dim);
    cuda_check(cudaGetLastError(), "k_gemm_naive");
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
    info.threads_per_block = 256;
    info.shared_mem_bytes_per_block = 0;

    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &info.max_active_blocks_per_sm, k_gemm_naive,
                   info.threads_per_block, info.shared_mem_bytes_per_block),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor (naive)");

    int device = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    info.occupancy = static_cast<double>(info.max_active_blocks_per_sm) *
                      info.threads_per_block /
                      static_cast<double>(prop.maxThreadsPerMultiProcessor);
    return info;
}

} // namespace Backend::GPUNaive
