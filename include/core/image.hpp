#pragma once

#include <cstdint>
#include <vector>

#include "core/types.hpp"

namespace mgpu {

// Graubilddaten im einfachen, zusammenhaengenden Speicherlayout.
struct GrayImage {
    Size2D size;
    int stride = 0;
    std::vector<std::uint8_t> data;

    void resize(Size2D new_size) {
        size = new_size;
        stride = new_size.width;
        data.assign(static_cast<size_t>(stride * size.height), 0);
    }
};

// RGB-Bilddaten im interleaved Layout (RGBRGB...).
struct RgbImage {
    Size2D size;
    int stride = 0;
    std::vector<std::uint8_t> data;

    void resize(Size2D new_size) {
        size = new_size;
        stride = new_size.width * 3;
        data.assign(static_cast<size_t>(stride * size.height), 0);
    }
};

} // namespace mgpu
