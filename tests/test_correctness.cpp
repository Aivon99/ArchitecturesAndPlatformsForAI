// Linear() seeds its He-init RNG with a fixed constant, so identical MLP
// shapes under different Kernels get identical weights — any mismatch below
// is a kernel bug, not init drift.
#include "core/mlp.h"
#include "core/kernel.h"
#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool allclose(const Tensor& a_in, const Tensor& b_in, float atol, float rtol) {
    Tensor a = a_in.cpu();
    Tensor b = b_in.cpu();

    if (a.numel() != b.numel()) {
        std::cerr << "shape mismatch: " << a.shape_str() << " vs " << b.shape_str() << "\n";
        return false;
    }

    bool ok = true;
    for (size_t i = 0; i < a.numel(); ++i) {
        float av = a.at(i), bv = b.at(i);
        float diff = std::fabs(av - bv);
        if (diff > atol + rtol * std::fabs(bv)) {
            std::cerr << "mismatch at index " << i << ": " << av << " vs " << bv
                      << " (diff " << diff << ")\n";
            ok = false;
        }
    }
    return ok;
}

bool check_shape(const std::vector<size_t>& widths, size_t B) {
    Tensor x_cpu = Tensor::randn({B, widths.front()}, 0.f, 1.f, /*seed=*/7, Device::CPU);

    MLP cpu_naive(widths, Kernel::CPU_NAIVE);
    MLP gpu_naive(widths, Kernel::GPU_NAIVE);
    MLP gpu_tiled(widths, Kernel::GPU_TILED);

    Tensor y_cpu        = cpu_naive.forward(x_cpu);
    Tensor y_gpu_naive  = gpu_naive.forward(x_cpu.cuda());
    Tensor y_gpu_tiled  = gpu_tiled.forward(x_cpu.cuda());

    const float atol = 1e-3f, rtol = 1e-2f;

    bool ok = true;
    if (!allclose(y_gpu_naive, y_cpu, atol, rtol)) {
        std::cerr << "FAIL: gpu_naive != cpu_naive for widths={";
        for (size_t w : widths) std::cerr << w << " ";
        std::cerr << "} B=" << B << "\n";
        ok = false;
    }
    if (!allclose(y_gpu_tiled, y_cpu, atol, rtol)) {
        std::cerr << "FAIL: gpu_tiled != cpu_naive for widths={";
        for (size_t w : widths) std::cerr << w << " ";
        std::cerr << "} B=" << B << "\n";
        ok = false;
    }
    return ok;
}

} // namespace

int main() {
    bool ok = true;

    ok &= check_shape({16, 32, 16}, /*B=*/4);
    ok &= check_shape({64, 64, 64, 64}, /*B=*/8);
    ok &= check_shape({33, 17}, /*B=*/5);     // non-tile-aligned dims (TILE=16)

    if (ok) {
        std::cout << "all backends agree\n";
        return 0;
    }
    std::cerr << "correctness check FAILED\n";
    return 1;
}
