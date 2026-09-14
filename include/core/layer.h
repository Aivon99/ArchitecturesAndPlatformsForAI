#pragma once
#include "core/tensor.h"
#include <string>

struct Layer {
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& x) = 0;
    virtual std::string name() const = 0;
};
