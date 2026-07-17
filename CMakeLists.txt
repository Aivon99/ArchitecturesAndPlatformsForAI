cmake_minimum_required(VERSION 3.24)
project(mlp_inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Auto-detect the local GPU's compute capability so kernels run natively
# instead of falling back to PTX JIT. Override with -DCMAKE_CUDA_ARCHITECTURES=<arch>
# if you're cross-compiling for a different card than the build machine.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

option(MLP_NATIVE_ARCH "Build the CPU baseline with -march=native" ON)

find_package(CUDAToolkit REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

#  core library (Tensor, Layer, Linear, ReLU, MLP)
add_library(core_lib
    src/core/tensor.cpp
    src/core/relu.cpp
    src/core/linear.cpp
    src/core/mlp.cpp
)
target_include_directories(core_lib PUBLIC include)
target_link_libraries(core_lib PUBLIC CUDA::cudart)

#  CPU baseline backend (naive sequential GEMM) 
add_library(cpu_backend src/backend/cpu_backend.cpp)
target_include_directories(cpu_backend PUBLIC include)
if(NOT MSVC)
    target_compile_options(cpu_backend PRIVATE -O3 $<$<BOOL:${MLP_NATIVE_ARCH}>:-march=native>)
endif()

# GPU backends: naive + shared-memory tiled kernels 
add_library(gpu_backend
    src/backend/gpu_naive_backend.cu
    src/backend/gpu_tiled_backend.cu
)
target_include_directories(gpu_backend PUBLIC include)
set_target_properties(gpu_backend PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(core_lib PUBLIC cpu_backend gpu_backend)

# benchmark sweep
add_executable(benchmark benchmark/main.cpp)
target_link_libraries(benchmark PRIVATE core_lib)

# correctness test
enable_testing()
add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE core_lib)
add_test(NAME correctness COMMAND test_correctness)
