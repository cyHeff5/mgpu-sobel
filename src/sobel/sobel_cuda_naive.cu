#include "sobel/sobel_cuda_naive.hpp"

#include <cuda_runtime.h>

namespace mgpu {

namespace {

__device__ __forceinline__ unsigned char clamp_u8(float v) {
    if (v < 0.0f) return 0;
    if (v > 255.0f) return 255;
    return static_cast<unsigned char>(v);
}

__global__ void sobel_gray_kernel(const unsigned char* in,
                                  unsigned char* out,
                                  int w,
                                  int h,
                                  int in_stride,
                                  int out_stride,
                                  float scale,
                                  float threshold,
                                  bool clamp) {
    // Jeder Thread berechnet genau ein Zielpixel.
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
        out[y * out_stride + x] = 0;
        return;
    }

    const int idx0 = (y - 1) * in_stride;
    const int idx1 = y * in_stride;
    const int idx2 = (y + 1) * in_stride;

    const int p00 = in[idx0 + x - 1];
    const int p01 = in[idx0 + x];
    const int p02 = in[idx0 + x + 1];
    const int p10 = in[idx1 + x - 1];
    const int p12 = in[idx1 + x + 1];
    const int p20 = in[idx2 + x - 1];
    const int p21 = in[idx2 + x];
    const int p22 = in[idx2 + x + 1];

    // Klassische Sobel-Kerne fuer X- und Y-Richtung.
    const int gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
    const int gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;

    float mag = sqrtf(static_cast<float>(gx * gx + gy * gy));
    mag *= scale;
    if (mag < threshold) {
        mag = 0.0f;
    }

    out[y * out_stride + x] = clamp ? clamp_u8(mag)
                                    : static_cast<unsigned char>(mag);
}

__global__ void sobel_depth_kernel(const float* in,
                                   unsigned char* out,
                                   int w,
                                   int h,
                                   int in_stride,
                                   int out_stride,
                                   float scale,
                                   float threshold,
                                   bool clamp) {
    // Identisches Mapping wie bei Gray, nur mit float-Input.
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
        out[y * out_stride + x] = 0;
        return;
    }

    const int idx0 = (y - 1) * in_stride;
    const int idx1 = y * in_stride;
    const int idx2 = (y + 1) * in_stride;

    const float p00 = in[idx0 + x - 1];
    const float p01 = in[idx0 + x];
    const float p02 = in[idx0 + x + 1];
    const float p10 = in[idx1 + x - 1];
    const float p12 = in[idx1 + x + 1];
    const float p20 = in[idx2 + x - 1];
    const float p21 = in[idx2 + x];
    const float p22 = in[idx2 + x + 1];

    const float gx = -p00 + p02 - 2.0f * p10 + 2.0f * p12 - p20 + p22;
    const float gy = -p00 - 2.0f * p01 - p02 + p20 + 2.0f * p21 + p22;

    float mag = sqrtf(gx * gx + gy * gy);
    mag *= scale;
    if (mag < threshold) {
        mag = 0.0f;
    }

    out[y * out_stride + x] = clamp ? clamp_u8(mag)
                                    : static_cast<unsigned char>(mag);
}

} // namespace

const char* SobelCudaNaive::name() const {
    return "CUDA Naive";
}

void SobelCudaNaive::apply(const GrayImage& input, GrayImage& output, const SobelParams& params) {
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

    if (cudaMalloc(&d_in, in_bytes) != cudaSuccess) {
        return;
    }
    if (cudaMalloc(&d_out, out_bytes) != cudaSuccess) {
        cudaFree(d_in);
        return;
    }

    // Input auf Device kopieren.
    cudaMemcpy(d_in, input.data.data(), in_bytes, cudaMemcpyHostToDevice);

    // 2D-Grid, damit x/y direkt Bildkoordinaten entsprechen.
    const dim3 block(16, 16);
    const dim3 grid((w + block.x - 1) / block.x,
                    (h + block.y - 1) / block.y);

    // Kernel ausfuehren und Ergebnis zurueckholen.
    sobel_gray_kernel<<<grid, block>>>(d_in, d_out, w, h, in_stride, out_stride,
                                       params.scale, params.threshold, params.clamp);
    cudaMemcpy(output.data.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);

    // Device-Speicher freigeben.
    cudaFree(d_out);
    cudaFree(d_in);
}

void SobelCudaNaive::apply(const DepthMap& input, GrayImage& output, const SobelParams& params) {
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

    if (cudaMalloc(&d_in, in_bytes) != cudaSuccess) {
        return;
    }
    if (cudaMalloc(&d_out, out_bytes) != cudaSuccess) {
        cudaFree(d_in);
        return;
    }

    // Input auf Device kopieren.
    cudaMemcpy(d_in, input.data.data(), in_bytes, cudaMemcpyHostToDevice);

    const dim3 block(16, 16);
    const dim3 grid((w + block.x - 1) / block.x,
                    (h + block.y - 1) / block.y);

    sobel_depth_kernel<<<grid, block>>>(d_in, d_out, w, h, in_stride, out_stride,
                                        params.scale, params.threshold, params.clamp);
    cudaMemcpy(output.data.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);

    // Device-Speicher freigeben.
    cudaFree(d_out);
    cudaFree(d_in);
}

} // namespace mgpu
