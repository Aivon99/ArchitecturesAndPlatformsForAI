#pragma once
#include "core/layer.h"
#include "core/kernel.h"
#include <vector>
#include <memory>

// Linear -> ReLU -> ... -> Linear. `widths` has one entry per layer boundary,
// e.g. {784, 512, 512, 10} builds three Linear layers with ReLU between them.
struct MLP {
    MLP(const std::vector<size_t>& widths, Kernel kernel);

    Tensor forward(const Tensor& x) const;

    Kernel kernel;

private:
    std::vector<std::unique_ptr<Layer>> layers;
};
