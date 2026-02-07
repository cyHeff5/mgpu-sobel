#pragma once

#include <string>

namespace mgpu {

// Globaler Enumsatz fuer austauschbare Backends.
enum class Backend {
    Cpu,
    Omp,
    CudaNaive,
    CudaTiled
};

// Kennzeichnet die Art des Inputs fuer die Pipeline.
enum class InputType {
    Rgb,
    Gray,
    Depth
};

// Einfache 2D-Groesse fuer Bild- und Depth-Dimensionen.
struct Size2D {
    int width = 0;
    int height = 0;
};

// Einheitlicher Status fuer IO, Pipeline und Benchmarking.
struct Status {
    bool ok = true;
    std::string message;
};

} // namespace mgpu
