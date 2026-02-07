cmake_minimum_required(VERSION 3.20)
project(mgpu_sobel LANGUAGES CXX)

# Schalter fuer optionale Backends.
option(USE_OPENMP "Enable OpenMP backend" ON)
option(USE_CUDA "Enable CUDA backends" OFF)

# Einheitlicher C++-Standard fuer alle Targets.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Zentrale Library: CPU-Logik, Pipeline und Benchmarking.
add_library(mgpu_sobel
    src/core/error.cpp
    src/core/timer.cpp
    src/benchmark/benchmark_runner.cpp
    src/io/depth_io.cpp
    src/io/image_io.cpp
    src/pipeline/edge_pipeline.cpp
    src/sobel/sobel_factory.cpp
    src/sobel/sobel_cpu.cpp
    src/sobel/sobel_omp.cpp
)

# Header aus include/ stehen allen Nutzern der Library zur Verfuegung.
target_include_directories(mgpu_sobel PUBLIC include)

# OpenMP nur aktivieren, wenn gewuenscht.
if (USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(mgpu_sobel PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(mgpu_sobel PUBLIC MGPU_USE_OPENMP=1)
endif()

# CUDA nur aktivieren, wenn gewuenscht.
if (USE_CUDA)
    enable_language(CUDA)
    target_sources(mgpu_sobel PRIVATE
        src/sobel/sobel_cuda_naive.cu
        src/sobel/sobel_cuda_tiled.cu
    )
    set_target_properties(mgpu_sobel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_definitions(mgpu_sobel PUBLIC MGPU_USE_CUDA=1)
endif()

# Ausfuehrbares Programm, das die Library nutzt.
add_executable(mgpu_sobel_app src/main.cpp)
target_link_libraries(mgpu_sobel_app PRIVATE mgpu_sobel)
