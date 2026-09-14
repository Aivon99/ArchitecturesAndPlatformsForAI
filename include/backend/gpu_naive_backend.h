#pragma once
#include "core/tensor.h"
#include "core/kernel.h"

namespace Backend::GPUNaive {

    // Y = X @ W^T + b, one thread per output element.
    void gemm(const Tensor& X, const Tensor& W,
              const Tensor& b, Tensor& Y);

    void relu_forward(Tensor& X);
    void fill(Tensor& X, float val);

    KernelLaunchInfo query_occupancy(int B, int in_dim, int out_dim);
}
