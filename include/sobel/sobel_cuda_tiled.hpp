#pragma once

#include "sobel/sobel.hpp"

namespace mgpu {

// CUDA-Implementierung mit Shared-Memory-Tiling.
class SobelCudaTiled final : public ISobelOperator {
public:
    const char* name() const override;

    void apply(const GrayImage& input,
               GrayImage& output,
               const SobelParams& params) override;

    void apply(const DepthMap& input,
               GrayImage& output,
               const SobelParams& params) override;
};

} // namespace mgpu
