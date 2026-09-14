#pragma once
#include "core/layer.h"
#include "core/kernel.h"

// Y = X @ W^T + b
struct Linear : Layer {
    Tensor W;       // (out x in)
    Tensor b;       // (out,)

    size_t in_dim;
    size_t out_dim;
    Kernel kernel;

    Linear(size_t in_dim, size_t out_dim, Kernel kernel);

    Tensor forward(const Tensor& x) override;

    std::string name() const override { return "Linear"; }
};
