#pragma once

#include <string>

#include "core/config.hpp"
#include "core/types.hpp"

namespace mgpu {

// Orchestriert Input -> Sobel -> Output unabhaengig vom Backend.
class EdgePipeline {
public:
    Status run(const AppConfig& config,
               const std::string& input_path,
               const std::string& output_path);
};

} // namespace mgpu
