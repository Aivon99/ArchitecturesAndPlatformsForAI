#ifndef GPU_TENSOR_H
#define GPU_TENSOR_H

#include "tensor.h"

class CpuTensor;

class GpuTensor : public Tensor {
private:
    float* device_data;

public:
    GpuTensor(size_t h = 1,
              size_t w = 1,
              size_t d = 1);

    ~GpuTensor() override;

    float* get_raw() override;
    const float* get_raw() const override;

    void copy_from_cpu(const CpuTensor& cpu_tensor);

    void copy_to_cpu(CpuTensor& cpu_tensor) const;
};

#endif