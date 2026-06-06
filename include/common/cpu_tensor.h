#ifndef CPU_TENSOR_H
#define CPU_TENSOR_H

#include <vector>

#include "tensor.h"

class CpuTensor : public Tensor {
private:
    std::vector<float> data;

    size_t flatten_index(size_t h,
                         size_t w,
                         size_t d) const;

public:
    CpuTensor(size_t h = 1,
              size_t w = 1,
              size_t d = 1);

    float* get_raw() override;
    const float* get_raw() const override;

    float& at(size_t h,
              size_t w,
              size_t d);

    const float& at(size_t h,
                    size_t w,
                    size_t d) const;
};

#endif