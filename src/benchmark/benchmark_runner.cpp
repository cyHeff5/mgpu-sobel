#include "benchmark/benchmark_runner.hpp"

#include "core/error.hpp"
#include "core/timer.hpp"
#include "io/depth_io.hpp"
#include "io/image_io.hpp"
#include "sobel/sobel.hpp"

namespace mgpu {

void BenchmarkRunner::add_backend(Backend backend) {
    backends_.push_back(backend);
}

Status BenchmarkRunner::run(const AppConfig& config,
                            const std::string& input_path,
                            std::vector<BenchmarkResult>& results) {
    if (backends_.empty()) {
        return error(ErrorCode::InvalidArgument, "Keine Backends zum Benchmarken");
    }

    results.clear();

    GrayImage gray_in;
    DepthMap depth_in;

    if (config.input_type == InputType::Depth) {
        const Status st = load_depth(input_path, depth_in);
        if (!st.ok) {
            return st;
        }
    } else if (config.input_type == InputType::Gray) {
        const Status st = load_gray(input_path, gray_in);
        if (!st.ok) {
            return st;
        }
    } else if (config.input_type == InputType::Rgb) {
        RgbImage rgb_in;
        const Status st = load_rgb(input_path, rgb_in);
        if (!st.ok) {
            return st;
        }

        const Status to_gray_status = rgb_to_gray(rgb_in, gray_in);
        if (!to_gray_status.ok) {
            return to_gray_status;
        }
    } else {
        return error(ErrorCode::InvalidArgument, "Unbekannter InputType");
    }

    for (const Backend backend : backends_) {
        AppConfig cfg = config;
        cfg.backend = backend;

        auto sobel = CreateSobelOperator(cfg.backend);
        if (!sobel) {
            return error(ErrorCode::NotSupported, "Unbekanntes Backend");
        }

        GrayImage edges;
        CpuTimer timer;
        timer.start();

        if (cfg.input_type == InputType::Depth) {
            edges.resize(depth_in.size);
            sobel->apply(depth_in, edges, cfg.sobel);
        } else {
            edges.resize(gray_in.size);
            sobel->apply(gray_in, edges, cfg.sobel);
        }

        const double ms = timer.stop_ms();
        results.push_back(BenchmarkResult{sobel->name(), ms});
    }

    return ok();
}

} // namespace mgpu
