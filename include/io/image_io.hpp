#pragma once

#include <string>

#include "core/image.hpp"
#include "core/types.hpp"

namespace mgpu {

// Laden/Speichern von RGB- und Graubildern.
Status load_rgb(const std::string& path, RgbImage& image);
Status load_gray(const std::string& path, GrayImage& image);
Status save_gray(const std::string& path, const GrayImage& image);
Status save_rgb(const std::string& path, const RgbImage& image);

// RGB -> Gray Konvertierung (Luminanz).
Status rgb_to_gray(const RgbImage& input, GrayImage& output);

} // namespace mgpu
