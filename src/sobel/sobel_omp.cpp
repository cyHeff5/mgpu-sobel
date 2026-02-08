#include "sobel/sobel_omp.hpp"

#include "core/math.hpp"

#if defined(MGPU_USE_OPENMP)
#include <omp.h>
#endif

namespace mgpu {

const char* SobelOmp::name() const {
    return "OpenMP";
}

void SobelOmp::apply(const GrayImage& input, GrayImage& output, const SobelParams& params) {
    // Ungueltige Eingabe frueh abfangen.
    if (input.size.width <= 0 || input.size.height <= 0) {
        return;
    }

    // Ausgabe immer auf gleiche Groesse bringen.
    output.resize(input.size);

    const int w = input.size.width;
    const int h = input.size.height;
    const int in_stride = input.stride;
    const int out_stride = output.stride;

    // Rand auf 0 setzen (seriell).
    // Diese Pixel haben kein vollstaendiges 3x3-Fenster.
    for (int x = 0; x < w; ++x) {
        output.data[x] = 0;
        output.data[(h - 1) * out_stride + x] = 0;
    }
    for (int y = 0; y < h; ++y) {
        output.data[y * out_stride] = 0;
        output.data[y * out_stride + (w - 1)] = 0;
    }

#if defined(MGPU_USE_OPENMP)
// Parallelisiert ueber die y-Zeilen. Jede Iteration schreibt nur in ihre eigene Zeile.
#pragma omp parallel for
#endif
    // Nur innere Zeilen bearbeiten (Rand bleibt 0).
    for (int y = 1; y < h - 1; ++y) {
        // Zeiger auf die drei relevanten Eingabezeilen.
        const std::uint8_t* r0 = input.data.data() + (y - 1) * in_stride;
        const std::uint8_t* r1 = input.data.data() + y * in_stride;
        const std::uint8_t* r2 = input.data.data() + (y + 1) * in_stride;
        // Zeiger auf die aktuelle Ausgabezeile.
        std::uint8_t* out = output.data.data() + y * out_stride;

        // Ueber alle inneren Spalten iterieren (Rand auslassen).
        // x startet bei 1 und endet bei w-2, damit x-1 und x+1 gueltig sind.
        for (int x = 1; x < w - 1; ++x) {
            // 3x3-Nachbarschaft um (x,y) laden.
            const int p00 = r0[x - 1];
            const int p01 = r0[x];
            const int p02 = r0[x + 1];
            const int p10 = r1[x - 1];
            const int p12 = r1[x + 1];
            const int p20 = r2[x - 1];
            const int p21 = r2[x];
            const int p22 = r2[x + 1];

            // Sobel Gx (horizontale Aenderung).
            // Kernel:
            // [-1  0  1]
            // [-2  0  2]
            // [-1  0  1]
            const int gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
            // Sobel Gy (vertikale Aenderung).
            // Kernel:
            // [-1 -2 -1]
            // [ 0  0  0]
            // [ 1  2  1]
            const int gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;

            // Gradienten zu einer Kantenstaerke zusammenfassen.
            float mag = sobel_magnitude(static_cast<float>(gx), static_cast<float>(gy));
            // Optional skalieren (z.B. fuer Sichtbarkeit/Normierung).
            mag *= params.scale;
            // Kleine Kanten unterdruecken.
            if (mag < params.threshold) {
                mag = 0.0f;
            }

            // In den Ausgabebereich [0,255] bringen.
            out[x] = params.clamp ? saturate_u8(mag)
                                  : static_cast<std::uint8_t>(mag);
        }
    }
}

void SobelOmp::apply(const DepthMap& input, GrayImage& output, const SobelParams& params) {
    // Ungueltige Eingabe frueh abfangen.
    if (input.size.width <= 0 || input.size.height <= 0) {
        return;
    }

    // Ausgabe immer auf gleiche Groesse bringen.
    output.resize(input.size);

    const int w = input.size.width;
    const int h = input.size.height;
    const int in_stride = input.stride;
    const int out_stride = output.stride;

    // Rand wie oben auf 0 setzen.
    for (int x = 0; x < w; ++x) {
        output.data[x] = 0;
        output.data[(h - 1) * out_stride + x] = 0;
    }
    for (int y = 0; y < h; ++y) {
        output.data[y * out_stride] = 0;
        output.data[y * out_stride + (w - 1)] = 0;
    }

#if defined(MGPU_USE_OPENMP)
// Auch hier parallel ueber Zeilen, jede Iteration schreibt nur in ihre eigene Zeile.
#pragma omp parallel for
#endif
    // Innere Zeilen bearbeiten.
    for (int y = 1; y < h - 1; ++y) {
        // Drei Zeilen der DepthMap.
        const float* r0 = input.data.data() + (y - 1) * in_stride;
        const float* r1 = input.data.data() + y * in_stride;
        const float* r2 = input.data.data() + (y + 1) * in_stride;
        // Ausgabezeile.
        std::uint8_t* out = output.data.data() + y * out_stride;

        // Ueber alle inneren Spalten iterieren (Rand auslassen).
        // x startet bei 1 und endet bei w-2, damit x-1 und x+1 gueltig sind.
        for (int x = 1; x < w - 1; ++x) {
            // 3x3-Nachbarschaft laden (hier float statt uint8).
            const float p00 = r0[x - 1];
            const float p01 = r0[x];
            const float p02 = r0[x + 1];
            const float p10 = r1[x - 1];
            const float p12 = r1[x + 1];
            const float p20 = r2[x - 1];
            const float p21 = r2[x];
            const float p22 = r2[x + 1];

            // Sobel auf Depth-Werten (gleiche Kerne, float-Arithmetik).
            const float gx = -p00 + p02 - 2.0f * p10 + 2.0f * p12 - p20 + p22;
            const float gy = -p00 - 2.0f * p01 - p02 + p20 + 2.0f * p21 + p22;

            // Gradientenbetrag, dann Parameter anwenden.
            float mag = sobel_magnitude(gx, gy);
            mag *= params.scale;
            if (mag < params.threshold) {
                mag = 0.0f;
            }

            // Ausgabe bleibt ein Graubild (uint8).
            out[x] = params.clamp ? saturate_u8(mag)
                                  : static_cast<std::uint8_t>(mag);
        }
    }
}

} // namespace mgpu
