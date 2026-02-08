#include "sobel/sobel_cuda_tiled.hpp"

#include <cuda_runtime.h>

namespace mgpu {

namespace {

// Kachelgroesse pro Block (Thread-Block = 16x16 Pixel).
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;

// Clamp auf gueltigen Grauwertbereich.
__device__ __forceinline__ unsigned char clamp_u8(float v) {
    if (v < 0.0f) return 0;
    if (v > 255.0f) return 255;
    return static_cast<unsigned char>(v);
}

// Liest Gray-Pixel mit Randcheck (ausserhalb -> 0).
__device__ __forceinline__ unsigned char load_gray_safe(const unsigned char* in,
                                                        int x,
                                                        int y,
                                                        int w,
                                                        int h,
                                                        int stride) {
    if (x < 0 || y < 0 || x >= w || y >= h) return 0;
    return in[y * stride + x];
}

// Liest Depth-Pixel mit Randcheck (ausserhalb -> 0).
__device__ __forceinline__ float load_depth_safe(const float* in,
                                                 int x,
                                                 int y,
                                                 int w,
                                                 int h,
                                                 int stride) {
    if (x < 0 || y < 0 || x >= w || y >= h) return 0.0f;
    return in[y * stride + x];
}

__global__ void sobel_gray_tiled_kernel(const unsigned char* in,
                                        unsigned char* out,
                                        int w,
                                        int h,
                                        int in_stride,
                                        int out_stride,
                                        float scale,
                                        float threshold,
                                        bool clamp) {
    // Shared Tile: 16x16 Kern + 1 Pixel Halo in alle Richtungen = 18x18.
    __shared__ unsigned char tile[BLOCK_Y + 2][BLOCK_X + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    // Globale Koordinate dieses Threads.
    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;

    // Lokale Koordinate im Shared Tile (+1 wegen Halo).
    const int sx = tx + 1;
    const int sy = ty + 1;

    // Mittelpunkt laden (der Pixel, den der Thread repraesentiert).
    tile[sy][sx] = load_gray_safe(in, x, y, w, h, in_stride);

    // Halo links/rechts laden (nur Rand-Threads im Block).
    if (tx == 0) {
        tile[sy][0] = load_gray_safe(in, x - 1, y, w, h, in_stride);
    }
    if (tx == BLOCK_X - 1) {
        tile[sy][BLOCK_X + 1] = load_gray_safe(in, x + 1, y, w, h, in_stride);
    }
    // Halo oben/unten laden.
    if (ty == 0) {
        tile[0][sx] = load_gray_safe(in, x, y - 1, w, h, in_stride);
    }
    if (ty == BLOCK_Y - 1) {
        tile[BLOCK_Y + 1][sx] = load_gray_safe(in, x, y + 1, w, h, in_stride);
    }

    // Ecken des Halos laden.
    if (tx == 0 && ty == 0) {
        tile[0][0] = load_gray_safe(in, x - 1, y - 1, w, h, in_stride);
    }
    if (tx == BLOCK_X - 1 && ty == 0) {
        tile[0][BLOCK_X + 1] = load_gray_safe(in, x + 1, y - 1, w, h, in_stride);
    }
    if (tx == 0 && ty == BLOCK_Y - 1) {
        tile[BLOCK_Y + 1][0] = load_gray_safe(in, x - 1, y + 1, w, h, in_stride);
    }
    if (tx == BLOCK_X - 1 && ty == BLOCK_Y - 1) {
        tile[BLOCK_Y + 1][BLOCK_X + 1] = load_gray_safe(in, x + 1, y + 1, w, h, in_stride);
    }

    // Sicherstellen, dass das komplette Tile geladen ist.
    __syncthreads();

    if (x >= w || y >= h) return;

    // Bildrand direkt auf 0 setzen.
    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
        out[y * out_stride + x] = 0;
        return;
    }

    // 3x3 Nachbarschaft aus Shared Memory lesen.
    const int p00 = tile[sy - 1][sx - 1];
    const int p01 = tile[sy - 1][sx];
    const int p02 = tile[sy - 1][sx + 1];
    const int p10 = tile[sy][sx - 1];
    const int p12 = tile[sy][sx + 1];
    const int p20 = tile[sy + 1][sx - 1];
    const int p21 = tile[sy + 1][sx];
    const int p22 = tile[sy + 1][sx + 1];

    // Sobel Gx/Gy (identisch zur CPU/naive Variante).
    const int gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
    const int gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;

    // Magnitude + Parameter anwenden.
    float mag = sqrtf(static_cast<float>(gx * gx + gy * gy));
    mag *= scale;
    if (mag < threshold) {
        mag = 0.0f;
    }

    out[y * out_stride + x] = clamp ? clamp_u8(mag)
                                    : static_cast<unsigned char>(mag);
}

__global__ void sobel_depth_tiled_kernel(const float* in,
                                         unsigned char* out,
                                         int w,
                                         int h,
                                         int in_stride,
                                         int out_stride,
                                         float scale,
                                         float threshold,
                                         bool clamp) {
    // Shared Tile fuer Depth (float) + Halo.
    __shared__ float tile[BLOCK_Y + 2][BLOCK_X + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    // Globale Koordinate dieses Threads.
    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;

    // Lokale Koordinate im Shared Tile (+1 wegen Halo).
    const int sx = tx + 1;
    const int sy = ty + 1;

    // Mittelpunkt laden.
    tile[sy][sx] = load_depth_safe(in, x, y, w, h, in_stride);

    // Halo links/rechts laden.
    if (tx == 0) {
        tile[sy][0] = load_depth_safe(in, x - 1, y, w, h, in_stride);
    }
    if (tx == BLOCK_X - 1) {
        tile[sy][BLOCK_X + 1] = load_depth_safe(in, x + 1, y, w, h, in_stride);
    }
    // Halo oben/unten laden.
    if (ty == 0) {
        tile[0][sx] = load_depth_safe(in, x, y - 1, w, h, in_stride);
    }
    if (ty == BLOCK_Y - 1) {
        tile[BLOCK_Y + 1][sx] = load_depth_safe(in, x, y + 1, w, h, in_stride);
    }

    // Ecken des Halos laden.
    if (tx == 0 && ty == 0) {
        tile[0][0] = load_depth_safe(in, x - 1, y - 1, w, h, in_stride);
    }
    if (tx == BLOCK_X - 1 && ty == 0) {
        tile[0][BLOCK_X + 1] = load_depth_safe(in, x + 1, y - 1, w, h, in_stride);
    }
    if (tx == 0 && ty == BLOCK_Y - 1) {
        tile[BLOCK_Y + 1][0] = load_depth_safe(in, x - 1, y + 1, w, h, in_stride);
    }
    if (tx == BLOCK_X - 1 && ty == BLOCK_Y - 1) {
        tile[BLOCK_Y + 1][BLOCK_X + 1] = load_depth_safe(in, x + 1, y + 1, w, h, in_stride);
    }

    // Warten bis alle Thread-Daten im Shared Memory liegen.
    __syncthreads();

    if (x >= w || y >= h) return;

    // Bildrand direkt auf 0 setzen.
    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
        out[y * out_stride + x] = 0;
        return;
    }

    // 3x3 Nachbarschaft aus Shared Memory lesen.
    const float p00 = tile[sy - 1][sx - 1];
    const float p01 = tile[sy - 1][sx];
    const float p02 = tile[sy - 1][sx + 1];
    const float p10 = tile[sy][sx - 1];
    const float p12 = tile[sy][sx + 1];
    const float p20 = tile[sy + 1][sx - 1];
    const float p21 = tile[sy + 1][sx];
    const float p22 = tile[sy + 1][sx + 1];

    // Sobel Gx/Gy fuer Depth.
    const float gx = -p00 + p02 - 2.0f * p10 + 2.0f * p12 - p20 + p22;
    const float gy = -p00 - 2.0f * p01 - p02 + p20 + 2.0f * p21 + p22;

    // Magnitude + Parameter anwenden.
    float mag = sqrtf(gx * gx + gy * gy);
    mag *= scale;
    if (mag < threshold) {
        mag = 0.0f;
    }

    out[y * out_stride + x] = clamp ? clamp_u8(mag)
                                    : static_cast<unsigned char>(mag);
}

} // namespace

const char* SobelCudaTiled::name() const {
    return "CUDA Tiled";
}

void SobelCudaTiled::apply(const GrayImage& input, GrayImage& output, const SobelParams& params) {
    // Ungueltige Eingabe frueh abfangen.
    if (input.size.width <= 0 || input.size.height <= 0) {
        return;
    }

    // Ausgabe auf Eingabegroesse bringen.
    output.resize(input.size);

    const int w = input.size.width;
    const int h = input.size.height;
    const int in_stride = input.stride;
    const int out_stride = output.stride;
    const size_t in_bytes = static_cast<size_t>(in_stride * h);
    const size_t out_bytes = static_cast<size_t>(out_stride * h);

    unsigned char* d_in = nullptr;
    unsigned char* d_out = nullptr;

    // Device-Speicher allokieren.
    if (cudaMalloc(&d_in, in_bytes) != cudaSuccess) {
        return;
    }
    if (cudaMalloc(&d_out, out_bytes) != cudaSuccess) {
        cudaFree(d_in);
        return;
    }

    // Input auf Device kopieren.
    cudaMemcpy(d_in, input.data.data(), in_bytes, cudaMemcpyHostToDevice);

    // Klassische 2D-Launch-Konfiguration.
    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((w + block.x - 1) / block.x,
                    (h + block.y - 1) / block.y);

    // Tiled Sobel auf GPU ausfuehren.
    sobel_gray_tiled_kernel<<<grid, block>>>(d_in, d_out, w, h, in_stride, out_stride,
                                             params.scale, params.threshold, params.clamp);
    // Ergebnis zurueck auf Host kopieren.
    cudaMemcpy(output.data.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);

    // Device-Speicher freigeben.
    cudaFree(d_out);
    cudaFree(d_in);
}

void SobelCudaTiled::apply(const DepthMap& input, GrayImage& output, const SobelParams& params) {
    // Ungueltige Eingabe frueh abfangen.
    if (input.size.width <= 0 || input.size.height <= 0) {
        return;
    }

    // Ausgabe auf Eingabegroesse bringen.
    output.resize(input.size);

    const int w = input.size.width;
    const int h = input.size.height;
    const int in_stride = input.stride;
    const int out_stride = output.stride;
    const size_t in_bytes = static_cast<size_t>(in_stride * h) * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(out_stride * h);

    float* d_in = nullptr;
    unsigned char* d_out = nullptr;

    // Device-Speicher allokieren.
    if (cudaMalloc(&d_in, in_bytes) != cudaSuccess) {
        return;
    }
    if (cudaMalloc(&d_out, out_bytes) != cudaSuccess) {
        cudaFree(d_in);
        return;
    }

    // Input auf Device kopieren.
    cudaMemcpy(d_in, input.data.data(), in_bytes, cudaMemcpyHostToDevice);

    // Klassische 2D-Launch-Konfiguration.
    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((w + block.x - 1) / block.x,
                    (h + block.y - 1) / block.y);

    // Tiled Sobel auf GPU ausfuehren (Depth-Version).
    sobel_depth_tiled_kernel<<<grid, block>>>(d_in, d_out, w, h, in_stride, out_stride,
                                              params.scale, params.threshold, params.clamp);
    // Ergebnis zurueck auf Host kopieren.
    cudaMemcpy(output.data.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);

    // Device-Speicher freigeben.
    cudaFree(d_out);
    cudaFree(d_in);
}

} // namespace mgpu
