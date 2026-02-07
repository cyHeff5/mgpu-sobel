#pragma once

#include "sobel/sobel.hpp"

namespace mgpu {

// CPU-Referenzimplementierung (Single Thread).
class SobelCpu final : public ISobelOperator {
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
