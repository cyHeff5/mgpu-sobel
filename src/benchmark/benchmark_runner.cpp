#include "benchmark/benchmark_runner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "core/error.hpp"
#include "core/timer.hpp"
#include "io/depth_io.hpp"
#include "io/image_io.hpp"
#include "sobel/sobel.hpp"

namespace mgpu {

void BenchmarkRunner::add_backend(Backend backend) {
    // Reihenfolge beibehalten: genau so werden Ergebnisse spaeter ausgegeben.
    backends_.push_back(backend);
}

Status BenchmarkRunner::run(const AppConfig& config,
                            const std::string& input_path,
                            std::vector<BenchmarkResult>& results) {
    if (backends_.empty()) {
        return error(ErrorCode::InvalidArgument, "Keine Backends zum Benchmarken");
    }
    if (config.benchmark.warmup_runs < 0) {
        return error(ErrorCode::InvalidArgument, "warmup_runs muss >= 0 sein");
    }
    if (config.benchmark.measured_runs <= 0) {
        return error(ErrorCode::InvalidArgument, "measured_runs muss > 0 sein");
    }

    results.clear();

    // Input nur einmal vorbereiten, damit pro Run wirklich nur Processing gemessen wird.
    GrayImage gray_in;
    DepthMap depth_in;

    if (config.input_type == InputType::Depth) {
        const Status st = load_depth(input_path, depth_in);
        if (!st.ok) {
            return st;
        }
        const Status norm_st = normalize_depth(depth_in);
        if (!norm_st.ok) {
            return norm_st;
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

        // Backend-spezifischen Sobel-Operator erzeugen (polymorphes Interface).
        auto sobel = CreateSobelOperator(cfg.backend);
        if (!sobel) {
            return error(ErrorCode::NotSupported, "Unbekanntes Backend");
        }

        GrayImage edges;
        // Gemeinsamer Apply-Pfad fuer Warmup und Messlaeufe.
        auto run_apply = [&]() {
            if (cfg.input_type == InputType::Depth) {
                edges.resize(depth_in.size);
                sobel->apply(depth_in, edges, cfg.sobel);
            } else {
                edges.resize(gray_in.size);
                sobel->apply(gray_in, edges, cfg.sobel);
            }
        };

        for (int i = 0; i < cfg.benchmark.warmup_runs; ++i) {
            run_apply();
        }

        // Mehrfach messen, um Schwankungen sichtbar zu machen.
        std::vector<double> samples;
        samples.reserve(static_cast<size_t>(cfg.benchmark.measured_runs));
        for (int i = 0; i < cfg.benchmark.measured_runs; ++i) {
#if defined(MGPU_USE_CUDA)
            // CUDA-Backends mit Event-Timer messen, CPU/OMP mit Host-Timer.
            if (cfg.backend == Backend::CudaNaive || cfg.backend == Backend::CudaTiled) {
                CudaTimer timer;
                timer.start();
                run_apply();
                samples.push_back(timer.stop_ms());
            } else
#endif
            {
                CpuTimer timer;
                timer.start();
                run_apply();
                samples.push_back(timer.stop_ms());
            }
        }

        double sum = 0.0;
        double min_ms = std::numeric_limits<double>::infinity();
        double max_ms = 0.0;
        for (double ms : samples) {
            sum += ms;
            min_ms = std::min(min_ms, ms);
            max_ms = std::max(max_ms, ms);
        }
        const double mean_ms = sum / static_cast<double>(samples.size());

        double variance = 0.0;
        for (double ms : samples) {
            const double d = ms - mean_ms;
            variance += d * d;
        }
        variance /= static_cast<double>(samples.size());
        const double stddev_ms = std::sqrt(variance);

        // Pro Backend zusammengefasste Statistik ablegen.
        results.push_back(BenchmarkResult{
            sobel->name(),
            mean_ms,
            min_ms,
            max_ms,
            stddev_ms,
            cfg.benchmark.warmup_runs,
            cfg.benchmark.measured_runs
        });
    }

    return ok();
}

} // namespace mgpu
