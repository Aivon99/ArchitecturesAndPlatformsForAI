#pragma once
#include "core/layer.h"
#include "core/kernel.h"

// Element-wise ReLU, in place on a copy of the input.
struct ReLU : Layer {
    explicit ReLU(Kernel kernel) : kernel(kernel) {}

    Tensor forward(const Tensor& x) override;

    std::string name() const override { return "ReLU"; }

    Kernel kernel;
};
