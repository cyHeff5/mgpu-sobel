#include <iostream>
#include <string>
#include <stdexcept>

#include "benchmark/benchmark_runner.hpp"
#include "core/config.hpp"
#include "core/error.hpp"
#include "pipeline/edge_pipeline.hpp"

namespace mgpu {

namespace {

void print_usage() {
    std::cout << "Usage:\n"
              << "  mgpu_sobel_app --input <path> --output <path>\n"
              << "               [--backend cpu|omp|cuda_naive|cuda_tiled]\n"
              << "               [--input-type rgb|gray|depth]\n"
              << "               [--benchmark-warmup <int>=2]\n"
              << "               [--benchmark-runs <int>=10]\n"
              << "               [--benchmark]\n";
}

Backend parse_backend(const std::string& value) {
    if (value == "cpu") return Backend::Cpu;
    if (value == "omp") return Backend::Omp;
    if (value == "cuda_naive") return Backend::CudaNaive;
    if (value == "cuda_tiled") return Backend::CudaTiled;
    // Fallback fuer unbekannte Werte (aktuell ohne Hard-Error).
    return Backend::Cpu;
}

InputType parse_input_type(const std::string& value) {
    if (value == "rgb") return InputType::Rgb;
    if (value == "gray") return InputType::Gray;
    if (value == "depth") return InputType::Depth;
    // Fallback fuer unbekannte Werte.
    return InputType::Rgb;
}

} // namespace

} // namespace mgpu

int main(int argc, char** argv) {
    using namespace mgpu;

    if (argc < 3) {
        print_usage();
        return 1;
    }

    AppConfig config;
    std::string input_path;
    std::string output_path;
    bool run_benchmark = false;
    // Zentraler Parser fuer Integer-CLI-Flags mit einheitlicher Fehlermeldung.
    auto parse_int_arg = [](const std::string& value, const char* name, int& out) -> bool {
        try {
            out = std::stoi(value);
            return true;
        } catch (const std::exception&) {
            std::cerr << "Ungueltiger Wert fuer " << name << ": " << value << "\n";
            return false;
        }
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--backend" && i + 1 < argc) {
            config.backend = parse_backend(argv[++i]);
        } else if (arg == "--input-type" && i + 1 < argc) {
            config.input_type = parse_input_type(argv[++i]);
        } else if (arg == "--benchmark-warmup" && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], "--benchmark-warmup", config.benchmark.warmup_runs)) {
                return 1;
            }
        } else if (arg == "--benchmark-runs" && i + 1 < argc) {
            if (!parse_int_arg(argv[++i], "--benchmark-runs", config.benchmark.measured_runs)) {
                return 1;
            }
        } else if (arg == "--benchmark") {
            run_benchmark = true;
        } else if (arg == "--help") {
            print_usage();
            return 0;
        }
    }

    if (input_path.empty()) {
        std::cerr << "Fehlender --input Pfad\n";
        print_usage();
        return 1;
    }

    if (!run_benchmark && output_path.empty()) {
        std::cerr << "Fehlender --output Pfad\n";
        print_usage();
        return 1;
    }

    if (run_benchmark) {
        BenchmarkRunner runner;
        // Alle verfuegbaren Backends in definierter Reihenfolge benchmarken.
        runner.add_backend(Backend::Cpu);
        runner.add_backend(Backend::Omp);
#if defined(MGPU_USE_CUDA)
        runner.add_backend(Backend::CudaNaive);
        runner.add_backend(Backend::CudaTiled);
#endif

        std::vector<BenchmarkResult> results;
        const Status st = runner.run(config, input_path, results);
        if (!st.ok) {
            std::cerr << "Benchmark fehlgeschlagen: " << st.message << "\n";
            return 1;
        }

        for (const auto& r : results) {
            std::cout << r.backend_name
                      << ": mean=" << r.mean_ms << " ms"
                      << ", min=" << r.min_ms << " ms"
                      << ", max=" << r.max_ms << " ms"
                      << ", stddev=" << r.stddev_ms << " ms"
                      << " (warmup=" << r.warmup_runs
                      << ", runs=" << r.measured_runs << ")\n";
        }
        return 0;
    }

    // Normalmodus: genau ein Backend ueber die Pipeline ausfuehren.
    EdgePipeline pipeline;
    const Status st = pipeline.run(config, input_path, output_path);
    if (!st.ok) {
        std::cerr << "Pipeline fehlgeschlagen: " << st.message << "\n";
        return 1;
    }

    std::cout << "Fertig.\n";
    return 0;
}
