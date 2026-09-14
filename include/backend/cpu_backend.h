#pragma once
#include "core/tensor.h"

namespace Backend::CPU {

    // Naive sequential, row-major GEMM: Y = X @ W^T + b.
    void gemm(const Tensor& X, const Tensor& W,
              const Tensor& b, Tensor& Y);

    void relu_forward(Tensor& X);
    void fill(Tensor& X, float val);
}
