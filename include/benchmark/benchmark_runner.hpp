#pragma once

#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/types.hpp"

namespace mgpu {

// Ergebnis eines einzelnen Backend-Laufs.
struct BenchmarkResult {
    std::string backend_name;
    double time_ms = 0.0;
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
