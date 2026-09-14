#include "core/mlp.h"
#include "core/linear.h"
#include "core/relu.h"
#include <stdexcept>

MLP::MLP(const std::vector<size_t>& widths, Kernel kernel_) : kernel(kernel_) {
    if (widths.size() < 2)
        throw std::runtime_error("MLP requires at least an input and an output width");

    for (size_t i = 0; i + 1 < widths.size(); ++i) {
        layers.push_back(std::make_unique<Linear>(widths[i], widths[i + 1], kernel));
        if (i + 2 < widths.size())
            layers.push_back(std::make_unique<ReLU>(kernel));
    }
}

Tensor MLP::forward(const Tensor& x) const {
    Tensor out = x;
    for (const auto& layer : layers)
        out = layer->forward(out);
    return out;
}
