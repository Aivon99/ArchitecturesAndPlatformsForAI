#pragma once
#include "core/tensor.h"
#include "core/kernel.h"

namespace Backend::GPUTiled {

    // Y = X @ W^T + b, shared-memory tiled GEMM.
    void gemm(const Tensor& X, const Tensor& W,
              const Tensor& b, Tensor& Y);

    void relu_forward(Tensor& X);
    void fill(Tensor& X, float val);

    KernelLaunchInfo query_occupancy(int B, int in_dim, int out_dim);
}
