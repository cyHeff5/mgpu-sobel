#pragma once

#include <cstdint>
#include <cmath>

namespace mgpu {

// Klemmt einen int-Wert in einen gueltigen Bereich.
inline int clamp_int(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

// Klemmt einen float-Wert in einen gueltigen Bereich.
inline float clamp_float(float v, float lo, float hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

// Wandelt in 8-Bit-Pixelwert, inkl. Begrenzung auf [0,255].
inline std::uint8_t saturate_u8(int v) {
    return static_cast<std::uint8_t>(clamp_int(v, 0, 255));
}

// Wandelt float in 8-Bit-Pixelwert, inkl. Begrenzung auf [0,255].
inline std::uint8_t saturate_u8(float v) {
    return static_cast<std::uint8_t>(clamp_int(static_cast<int>(v), 0, 255));
}

// Sobel-Groesse aus gx/gy (klassische L2-Norm).
inline float sobel_magnitude(float gx, float gy) {
    return std::sqrt(gx * gx + gy * gy);
}

// Normalisiert einen Wert von [vmin, vmax] nach [0,1].
inline float normalize(float v, float vmin, float vmax) {
    return (v - vmin) / (vmax - vmin);
}

// RGB -> Gray (Luminanz nach Standardgewichtung).
inline std::uint8_t to_gray(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    return static_cast<std::uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);
}

} // namespace mgpu
