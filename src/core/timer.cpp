#include "core/timer.hpp"

#include <chrono>

#if defined(MGPU_USE_CUDA)
#include <cuda_runtime.h>
#endif

namespace mgpu {

namespace {

using Clock = std::chrono::high_resolution_clock;

} // namespace

void CpuTimer::start() {
    // Store start time as opaque pointer-sized integer.
    start_ = Clock::now();
}

double CpuTimer::stop_ms() {
    const auto end = Clock::now();
    const auto duration = std::chrono::duration<double, std::milli>(end - start_);
    return duration.count();
}

CudaTimer::~CudaTimer() {
#if defined(MGPU_USE_CUDA)
    // Events gehoeren dem Timer-Objekt und werden beim Lebensende freigegeben.
    if (initialized_) {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
#endif
}

void CudaTimer::start() {
#if defined(MGPU_USE_CUDA)
    // Lazy-Init, damit ein ungenutzter Timer keine CUDA-Ressourcen belegt.
    if (!initialized_) {
        if (cudaEventCreate(&start_) != cudaSuccess) {
            return;
        }
        if (cudaEventCreate(&stop_) != cudaSuccess) {
            cudaEventDestroy(start_);
            return;
        }
        initialized_ = true;
    }

    // Startmarke in die Default-Stream-Sequenz schreiben.
    cudaEventRecord(start_);
#else
    // Fallback ohne CUDA-Unterstuetzung.
    start_ = Clock::now();
#endif
}

double CudaTimer::stop_ms() {
#if defined(MGPU_USE_CUDA)
    if (!initialized_) {
        return 0.0;
    }
    // Stoppmarke setzen und auf Abschluss warten, dann Laufzeit in ms auslesen.
    if (cudaEventRecord(stop_) != cudaSuccess) {
        return 0.0;
    }
    if (cudaEventSynchronize(stop_) != cudaSuccess) {
        return 0.0;
    }

    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, start_, stop_) != cudaSuccess) {
        return 0.0;
    }
    return static_cast<double>(ms);
#else
    const auto end = Clock::now();
    const auto duration = std::chrono::duration<double, std::milli>(end - start_);
    return duration.count();
#endif
}

} // namespace mgpu
