#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <string>

enum class Device { CPU, CUDA };

struct Tensor {
    float*              host_ptr   = nullptr;
    float*              device_ptr = nullptr;
    std::vector<size_t> shape;
    Device              device     = Device::CPU;

    Tensor() = default;
    Tensor(float* ptr, std::vector<size_t> shape, Device dev);

    Tensor(const Tensor&);
    Tensor& operator=(const Tensor&);
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;
    ~Tensor();

    static Tensor zeros(std::vector<size_t> shape, Device dev = Device::CPU);
    static Tensor ones (std::vector<size_t> shape, Device dev = Device::CPU);
    static Tensor randn(std::vector<size_t> shape,
                        float mean = 0.f, float std = 1.f,
                        unsigned seed = 42,
                        Device dev = Device::CPU);

    Tensor cuda() const;
    Tensor cpu()  const;

    float*       data()       noexcept;
    const float* data() const noexcept;

    float&       at(size_t i);
    const float& at(size_t i) const;

    size_t numel() const noexcept;
    size_t ndim()  const noexcept { return shape.size(); }

    size_t rows() const { return shape.at(0); }
    size_t cols() const { return shape.at(1); }

    std::string shape_str() const;

private:
    void free_memory();
    void alloc_cpu(size_t n);
    void alloc_gpu(size_t n);
    void copy_from(const Tensor& other);
};
