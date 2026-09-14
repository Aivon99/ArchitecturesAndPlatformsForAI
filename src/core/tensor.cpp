#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstring>
#include <random>
#include <sstream>
#include <numeric>
#include <cassert>

static void cuda_check(cudaError_t err, const char* ctx) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(ctx) + ": " + cudaGetErrorString(err));
}

void Tensor::alloc_cpu(size_t n) {
    host_ptr = new float[n]();
}

void Tensor::alloc_gpu(size_t n) {
    cuda_check(cudaMalloc(&device_ptr, n * sizeof(float)), "Tensor alloc_gpu");
    cuda_check(cudaMemset(device_ptr, 0, n * sizeof(float)), "Tensor alloc_gpu memset");
}

void Tensor::free_memory() {
    if (host_ptr) {
        delete[] host_ptr;
        host_ptr = nullptr;
    }
    if (device_ptr) {
        cudaFree(device_ptr);
        device_ptr = nullptr;
    }
}

size_t Tensor::numel() const noexcept {
    if (shape.empty()) return 0;
    size_t n = 1;
    for (size_t d : shape) n *= d;
    return n;
}

Tensor::Tensor(float* ptr, std::vector<size_t> sh, Device dev)
    : shape(std::move(sh)), device(dev)
{
    if (dev == Device::CPU) host_ptr   = ptr;
    else                    device_ptr = ptr;
}

void Tensor::copy_from(const Tensor& other) {
    shape         = other.shape;
    device        = other.device;
    size_t n      = other.numel();

    if (device == Device::CPU) {
        alloc_cpu(n);
        std::memcpy(host_ptr, other.host_ptr, n * sizeof(float));
    } else {
        alloc_gpu(n);
        cuda_check(cudaMemcpy(device_ptr, other.device_ptr,
                              n * sizeof(float), cudaMemcpyDeviceToDevice),
                   "Tensor copy_from D2D");
    }
}

Tensor::Tensor(const Tensor& other) {
    copy_from(other);
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        free_memory();
        copy_from(other);
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : host_ptr(other.host_ptr), device_ptr(other.device_ptr),
      shape(std::move(other.shape)), device(other.device)
{
    other.host_ptr = other.device_ptr = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        free_memory();
        host_ptr      = other.host_ptr;
        device_ptr    = other.device_ptr;
        shape         = std::move(other.shape);
        device        = other.device;
        other.host_ptr = other.device_ptr = nullptr;
    }
    return *this;
}

Tensor::~Tensor() {
    free_memory();
}

Tensor Tensor::zeros(std::vector<size_t> shape, Device dev) {
    Tensor t;
    t.shape  = shape;
    t.device = dev;
    size_t n = t.numel();

    if (dev == Device::CPU) t.alloc_cpu(n);
    else                    t.alloc_gpu(n);
    return t;
}

Tensor Tensor::ones(std::vector<size_t> shape, Device dev) {
    Tensor t = zeros(shape, dev);
    if (dev == Device::CPU) {
        std::fill(t.host_ptr, t.host_ptr + t.numel(), 1.f);
    } else {
        std::vector<float> tmp(t.numel(), 1.f);
        cuda_check(cudaMemcpy(t.device_ptr, tmp.data(),
                              t.numel() * sizeof(float),
                              cudaMemcpyHostToDevice), "Tensor::ones");
    }
    return t;
}

Tensor Tensor::randn(std::vector<size_t> shape,
                     float mean, float std,
                     unsigned seed, Device dev)
{
    // No cuRAND dependency: always generate on CPU, then copy to device if needed.
    Tensor t;
    t.shape  = shape;
    t.device = Device::CPU;
    size_t n = t.numel();
    t.alloc_cpu(n);

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(mean, std);
    for (size_t i = 0; i < n; ++i) t.host_ptr[i] = dist(rng);

    if (dev == Device::CUDA) return t.cuda();
    return t;
}

Tensor Tensor::cuda() const {
    if (device == Device::CUDA) return *this;

    Tensor t;
    t.shape         = shape;
    t.device        = Device::CUDA;
    size_t n        = numel();
    t.alloc_gpu(n);

    cuda_check(cudaMemcpy(t.device_ptr, host_ptr,
                          n * sizeof(float), cudaMemcpyHostToDevice),
               "Tensor::cuda H2D");
    return t;
}

Tensor Tensor::cpu() const {
    if (device == Device::CPU) return *this;

    Tensor t;
    t.shape         = shape;
    t.device        = Device::CPU;
    size_t n        = numel();
    t.alloc_cpu(n);

    cuda_check(cudaMemcpy(t.host_ptr, device_ptr,
                          n * sizeof(float), cudaMemcpyDeviceToHost),
               "Tensor::cpu D2H");
    return t;
}

float* Tensor::data() noexcept {
    return device == Device::CPU ? host_ptr : device_ptr;
}

const float* Tensor::data() const noexcept {
    return device == Device::CPU ? host_ptr : device_ptr;
}

float& Tensor::at(size_t i) {
    if (device != Device::CPU)
        throw std::runtime_error("Tensor::at — call cpu() first");
    if (i >= numel())
        throw std::out_of_range("Tensor::at — index out of range");
    return host_ptr[i];
}

const float& Tensor::at(size_t i) const {
    if (device != Device::CPU)
        throw std::runtime_error("Tensor::at — call cpu() first");
    if (i >= numel())
        throw std::out_of_range("Tensor::at — index out of range");
    return host_ptr[i];
}

std::string Tensor::shape_str() const {
    std::ostringstream ss;
    ss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        ss << shape[i];
        if (i + 1 < shape.size()) ss << ", ";
    }
    ss << ")";
    return ss.str();
}
