#include "core/linear.h"
#include "backend/cpu_backend.h"
#include "backend/gpu_naive_backend.h"
#include "backend/gpu_tiled_backend.h"
#include <cmath>

Linear::Linear(size_t in_dim_, size_t out_dim_, Kernel kernel_)
    : in_dim(in_dim_), out_dim(out_dim_), kernel(kernel_)
{
    const Device dev = device_for(kernel);
    const float he_std = std::sqrt(2.f / static_cast<float>(in_dim));

    W = Tensor::randn({out_dim, in_dim}, 0.f, he_std, /*seed=*/42, dev);
    b = Tensor::zeros({out_dim}, dev);
}

Tensor Linear::forward(const Tensor& x) {
    Tensor y = Tensor::zeros({x.rows(), out_dim}, x.device);

    switch (kernel) {
        case Kernel::CPU_NAIVE: Backend::CPU::gemm(x, W, b, y);       break;
        case Kernel::GPU_NAIVE: Backend::GPUNaive::gemm(x, W, b, y);  break;
        case Kernel::GPU_TILED: Backend::GPUTiled::gemm(x, W, b, y);  break;
    }

    return y;
}
