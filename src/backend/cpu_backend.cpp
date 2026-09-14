#include "backend/cpu_backend.h"
#include <stdexcept>
#include <algorithm>

namespace Backend::CPU {

void gemm(const Tensor& X, const Tensor& W,
          const Tensor& b, Tensor& Y)
{
    if (X.device != Device::CPU || W.device != Device::CPU ||
        b.device != Device::CPU || Y.device != Device::CPU)
        throw std::runtime_error("CPU::gemm — all tensors must be on CPU");

    const int B       = static_cast<int>(X.rows());
    const int in_dim  = static_cast<int>(X.cols());
    const int out_dim = static_cast<int>(W.rows());

    if (static_cast<int>(W.cols()) != in_dim)
        throw std::runtime_error("CPU::gemm — W shape mismatch");

    const float* xp = X.data();
    const float* wp = W.data();
    const float* bp = b.data();
    float*       yp = Y.data();

    for (int row = 0; row < B; ++row) {
        for (int col = 0; col < out_dim; ++col) {
            float acc = 0.f;
            for (int k = 0; k < in_dim; ++k)
                acc += xp[row * in_dim + k] * wp[col * in_dim + k];
            yp[row * out_dim + col] = acc + bp[col];
        }
    }
}

void relu_forward(Tensor& X) {
    float* p = X.data();
    size_t n = X.numel();
    for (size_t i = 0; i < n; ++i)
        p[i] = p[i] > 0.f ? p[i] : 0.f;
}

void fill(Tensor& X, float val) {
    float* p = X.data();
    size_t n = X.numel();
    std::fill(p, p + n, val);
}

} // namespace Backend::CPU
