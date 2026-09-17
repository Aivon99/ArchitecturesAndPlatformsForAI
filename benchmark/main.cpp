// MLP::forward allocates a fresh output Tensor per layer per call, so
// cudaMalloc/cudaFree churn is included in the GPU-timed window below, not
// just the GEMM kernels themselves — noisy at B=1 if that ever matters.

#include "core/mlp.h"
#include "core/kernel.h"
#include "backend/gpu_naive_backend.h"
#include "backend/gpu_tiled_backend.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

struct Args {
    std::vector<Kernel> kernels   = {Kernel::CPU_NAIVE, Kernel::GPU_NAIVE, Kernel::GPU_TILED};
    std::vector<size_t> batch_sizes = {1, 8, 64, 256, 1024, 4096};
    std::vector<size_t> widths      = {128, 512, 2048};
    std::vector<size_t> depths      = {1, 2, 4, 8, 16};
    int    repeats = 10;
    int    warmup  = 3;
    std::string out_path = "results/benchmark.csv";
};

Kernel parse_kernel(const std::string& s) {
    if (s == "cpu" || s == "cpu_naive") return Kernel::CPU_NAIVE;
    if (s == "gpu_naive")               return Kernel::GPU_NAIVE;
    if (s == "gpu_tiled")               return Kernel::GPU_TILED;
    throw std::runtime_error("unknown --kernel value: " + s);
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
            return argv[++i];
        };
        if (flag == "--kernel") {
            args.kernels = {parse_kernel(next())};
        } else if (flag == "--repeats") {
            args.repeats = std::stoi(next());
        } else if (flag == "--warmup") {
            args.warmup = std::stoi(next());
        } else if (flag == "--depth") {
            args.depths = {static_cast<size_t>(std::stoul(next()))};
        } else if (flag == "--out") {
            args.out_path = next();
        } else if (flag == "--max-batch") {
            size_t cap = static_cast<size_t>(std::stoul(next()));
            args.batch_sizes.erase(
                std::remove_if(args.batch_sizes.begin(), args.batch_sizes.end(),
                                [cap](size_t b) { return b > cap; }),
                args.batch_sizes.end());
        } else {
            throw std::runtime_error("unknown argument: " + flag);
        }
    }
    return args;
}

double flops_for_layers(const std::vector<size_t>& widths, size_t B) {
    double flops = 0.0;
    for (size_t i = 0; i + 1 < widths.size(); ++i)
        flops += 2.0 * static_cast<double>(B) * widths[i] * widths[i + 1];
    return flops;
}

// Idealised (lower-bound) memory traffic: X, W, b, Y each read/written once
// per layer. Kept independent of which kernel runs it, so arithmetic
// intensity is a property of the shape, not of the kernel's actual traffic.
double bytes_for_layers(const std::vector<size_t>& widths, size_t B) {
    double bytes = 0.0;
    for (size_t i = 0; i + 1 < widths.size(); ++i) {
        size_t in_dim = widths[i], out_dim = widths[i + 1];
        double elems = static_cast<double>(B) * in_dim
                     + static_cast<double>(out_dim) * in_dim
                     + static_cast<double>(out_dim)
                     + static_cast<double>(B) * out_dim;
        bytes += elems * sizeof(float);
    }
    return bytes;
}

void write_device_info(const std::string& path) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return;

    double peak_bw_gbs = 2.0 * (prop.memoryClockRate * 1e3) *
                          (prop.memoryBusWidth / 8.0) / 1e9;

    std::ofstream out(path);
    out << "name," << prop.name << "\n";
    out << "sm_count," << prop.multiProcessorCount << "\n";
    out << "max_threads_per_sm," << prop.maxThreadsPerMultiProcessor << "\n";
    out << "max_shared_mem_per_block_bytes," << prop.sharedMemPerBlock << "\n";
    out << "memory_clock_khz," << prop.memoryClockRate << "\n";
    out << "memory_bus_width_bits," << prop.memoryBusWidth << "\n";
    out << "peak_memory_bandwidth_gbs," << peak_bw_gbs << "\n";
}

KernelLaunchInfo occupancy_for(Kernel kernel, int B, int in_dim, int out_dim) {
    switch (kernel) {
        case Kernel::GPU_NAIVE: return Backend::GPUNaive::query_occupancy(B, in_dim, out_dim);
        case Kernel::GPU_TILED: return Backend::GPUTiled::query_occupancy(B, in_dim, out_dim);
        case Kernel::CPU_NAIVE: return KernelLaunchInfo{};
    }
    return KernelLaunchInfo{};
}

double time_forward_cpu_ms(const MLP& mlp, const Tensor& x, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) mlp.forward(x);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeats; ++i) mlp.forward(x);
    auto t1 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;
}

double time_forward_gpu_ms(const MLP& mlp, const Tensor& x, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) mlp.forward(x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i) mlp.forward(x);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(ms) / repeats;
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    std::filesystem::create_directories(
        std::filesystem::path(args.out_path).parent_path());

    write_device_info(
        (std::filesystem::path(args.out_path).parent_path() / "device_info.csv").string());

    std::ofstream csv(args.out_path);
    csv << "kernel,batch_size,width,depth,time_ms,throughput_samples_per_sec,gflops,"
           "bytes_moved,arithmetic_intensity,occupancy_pct,shared_mem_bytes_per_block\n";

    for (Kernel kernel : args.kernels) {
        for (size_t width : args.widths) {
            KernelLaunchInfo launch_info = occupancy_for(
                kernel, /*B=*/1, static_cast<int>(width), static_cast<int>(width));

            for (size_t depth : args.depths) {
                std::vector<size_t> layer_widths(depth + 1, width);
                MLP mlp(layer_widths, kernel);

                for (size_t B : args.batch_sizes) {
                    Tensor x = Tensor::randn({B, width}, 0.f, 1.f, /*seed=*/123,
                                              device_for(kernel));

                    double ms = (kernel == Kernel::CPU_NAIVE)
                                    ? time_forward_cpu_ms(mlp, x, args.warmup, args.repeats)
                                    : time_forward_gpu_ms(mlp, x, args.warmup, args.repeats);

                    double seconds   = ms / 1000.0;
                    double throughput = static_cast<double>(B) / seconds;
                    double flops      = flops_for_layers(layer_widths, B);
                    double bytes      = bytes_for_layers(layer_widths, B);
                    double gflops     = flops / seconds / 1e9;
                    double arith_intensity = flops / bytes;
                    double occupancy_pct   = launch_info.occupancy * 100.0;

                    csv << kernel_name(kernel) << ',' << B << ',' << width << ','
                        << depth << ',' << ms << ',' << throughput << ','
                        << gflops << ',' << bytes << ',' << arith_intensity << ','
                        << occupancy_pct << ',' << launch_info.shared_mem_bytes_per_block << '\n';

                    std::cout << kernel_name(kernel) << " B=" << B << " d=" << width
                              << " depth=" << depth << " -> " << ms << " ms, "
                              << throughput << " samples/s, " << gflops << " GFLOP/s, "
                              << "AI=" << arith_intensity << " FLOP/B, "
                              << "occ=" << occupancy_pct << "%\n";
                }
            }
        }
    }

    std::cout << "wrote " << args.out_path << "\n";
    return 0;
}
