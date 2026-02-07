#pragma once

#include <vector>

#include "core/types.hpp"

namespace mgpu {

// DepthMap mit float-Werten fuer geometrische Kanten.
struct DepthMap {
    Size2D size;
    int stride = 0;
    std::vector<float> data;

    void resize(Size2D new_size) {
        size = new_size;
        stride = new_size.width;
        data.assign(static_cast<size_t>(stride * size.height), 0.0f);
    }
};

} // namespace mgpu
