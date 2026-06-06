#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <stdexcept>

class Tensor {
protected:
    size_t height;
    size_t width;
    size_t depth;

public:
    virtual ~Tensor() = default;

    Tensor(size_t h = 1,
           size_t w = 1,
           size_t d = 1);

    size_t get_height() const;
    size_t get_width() const;
    size_t get_depth() const;

    size_t size() const;

    virtual float* get_raw() = 0;
    virtual const float* get_raw() const = 0;
};

#endif
