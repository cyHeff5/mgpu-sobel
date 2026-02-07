#pragma once

#include <memory>

#include "core/config.hpp"
#include "core/depth.hpp"
#include "core/image.hpp"
#include "core/types.hpp"

namespace mgpu {

// Gemeinsames Interface fuer alle Sobel-Backends.
class ISobelOperator {
public:
    virtual ~ISobelOperator() = default;

    // Kurzer Backend-Name fuer Logs/Benchmarks.
    virtual const char* name() const = 0;

    // Sobel auf Graubildern (z.B. RGB->Gray vorgelagert).
    virtual void apply(const GrayImage& input,
                       GrayImage& output,
                       const SobelParams& params) = 0;

    // Sobel auf DepthMaps fuer geometrische Kanten.
    virtual void apply(const DepthMap& input,
                       GrayImage& output,
                       const SobelParams& params) = 0;
};

// Factory fuer die aktuelle Backend-Auswahl.
std::unique_ptr<ISobelOperator> CreateSobelOperator(Backend backend);

} // namespace mgpu
