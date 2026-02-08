#pragma once

#include "core/types.hpp"

namespace mgpu {

// Parameter fuer den Sobel-Operator.
struct SobelParams {
    float scale = 1.0f;
    float threshold = 0.0f;
    bool clamp = true;
};

// Steuerung fuer Benchmark-Durchlaeufe.
struct BenchmarkParams {
    int warmup_runs = 2;
    int measured_runs = 10;
};

// Zentrale App-Konfiguration fuer Backend und Inputtyp.
struct AppConfig {
    Backend backend = Backend::Cpu;
    InputType input_type = InputType::Rgb;
    SobelParams sobel;
    BenchmarkParams benchmark;
};

} // namespace mgpu
