#pragma once

#include <string>

#include "core/depth.hpp"
#include "core/types.hpp"

namespace mgpu {

// Laden/Speichern von DepthMaps.
Status load_depth(const std::string& path, DepthMap& depth);
Status save_depth(const std::string& path, const DepthMap& depth);

} // namespace mgpu
