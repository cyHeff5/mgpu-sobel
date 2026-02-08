#pragma once

#include <chrono>

#if defined(MGPU_USE_CUDA)
#include <cuda_runtime.h>
#endif

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
    CudaTimer() = default;
    ~CudaTimer();

    void start();
    double stop_ms();

private:
#if defined(MGPU_USE_CUDA)
    // CUDA-Events messen GPU-Zeit auf der Device-Timeline.
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
    // Erst beim ersten start() initialisieren.
    bool initialized_ = false;
#else
    std::chrono::high_resolution_clock::time_point start_{};
#endif
};

} // namespace mgpu
