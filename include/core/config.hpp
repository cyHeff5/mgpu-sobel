#pragma once

#include "core/types.hpp"

namespace mgpu {

// Parameter fuer den Sobel-Operator.
struct SobelParams {
    float scale = 1.0f;
    float threshold = 0.0f;
    bool clamp = true;
};

// Zentrale App-Konfiguration fuer Backend und Inputtyp.
struct AppConfig {
    Backend backend = Backend::Cpu;
    InputType input_type = InputType::Rgb;
    SobelParams sobel;
};

} // namespace mgpu
