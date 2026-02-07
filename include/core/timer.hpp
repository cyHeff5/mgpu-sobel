#pragma once

#include <chrono>

namespace mgpu {

// CPU-Timer fuer Benchmarking im Host-Code.
class CpuTimer {
public:
    void start();
    double stop_ms();

private:
    std::chrono::high_resolution_clock::time_point start_{};
};

// CUDA-Timer fuer GPU-Kernels (Zeit in ms).
class CudaTimer {
public:
    void start();
    double stop_ms();

private:
    std::chrono::high_resolution_clock::time_point start_{};
};

} // namespace mgpu
