#include "core/timer.hpp"

#include <chrono>

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

void CudaTimer::start() {
    // Placeholder CPU timer; replace with CUDA events in .cu later.
    start_ = Clock::now();
}

double CudaTimer::stop_ms() {
    const auto end = Clock::now();
    const auto duration = std::chrono::duration<double, std::milli>(end - start_);
    return duration.count();
}

} // namespace mgpu
