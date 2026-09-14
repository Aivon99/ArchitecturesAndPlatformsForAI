#pragma once
#include "core/tensor.h"
#include <cstddef>

enum class Kernel { CPU_NAIVE, GPU_NAIVE, GPU_TILED };

// Left default-zeroed for CPU_NAIVE, which has no such notion.
struct KernelLaunchInfo {
    int    threads_per_block          = 0;
    int    max_active_blocks_per_sm   = 0;
    size_t shared_mem_bytes_per_block = 0;
    double occupancy                  = 0.0;
};

inline Device device_for(Kernel k) {
    return k == Kernel::CPU_NAIVE ? Device::CPU : Device::CUDA;
}

inline const char* kernel_name(Kernel k) {
    switch (k) {
        case Kernel::CPU_NAIVE: return "cpu_naive";
        case Kernel::GPU_NAIVE: return "gpu_naive";
        case Kernel::GPU_TILED: return "gpu_tiled";
    }
    return "unknown";
}
