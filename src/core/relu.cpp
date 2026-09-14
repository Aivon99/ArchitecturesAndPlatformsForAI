#include "core/relu.h"
#include "backend/cpu_backend.h"
#include "backend/gpu_naive_backend.h"
#include "backend/gpu_tiled_backend.h"

Tensor ReLU::forward(const Tensor& x) {
    Tensor out = x;   // deep copy — modified in place below

    switch (kernel) {
        case Kernel::CPU_NAIVE: Backend::CPU::relu_forward(out);      break;
        case Kernel::GPU_NAIVE: Backend::GPUNaive::relu_forward(out); break;
        case Kernel::GPU_TILED: Backend::GPUTiled::relu_forward(out); break;
    }

    return out;
}
