#pragma once

#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/types.hpp"

namespace mgpu {

// Ergebnis eines einzelnen Backend-Laufs.
struct BenchmarkResult {
    std::string backend_name;
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double stddev_ms = 0.0;
    int warmup_runs = 0;
    int measured_runs = 0;
};

// Fuehrt wiederholbare Benchmarks fuer mehrere Backends aus.
class BenchmarkRunner {
public:
    void add_backend(Backend backend);
    Status run(const AppConfig& config,
               const std::string& input_path,
               std::vector<BenchmarkResult>& results);

private:
    std::vector<Backend> backends_;
};

} // namespace mgpu
