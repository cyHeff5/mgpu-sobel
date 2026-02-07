#include "pipeline/edge_pipeline.hpp"

#include "core/error.hpp"
#include "io/depth_io.hpp"
#include "io/image_io.hpp"
#include "sobel/sobel.hpp"

namespace mgpu {

Status EdgePipeline::run(const AppConfig& config,
                         const std::string& input_path,
                         const std::string& output_path) {
    // 1) Backend erzeugen (Factory kapselt Auswahl).
    auto sobel = CreateSobelOperator(config.backend);
    if (!sobel) {
        return error(ErrorCode::NotSupported, "Unbekanntes Backend");
    }

    // 2) Input laden und in gemeinsamen Typ ueberfuehren.
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

    // 3) Ausgabe vorbereiten (graues Kantenbild).
    GrayImage edges;
    if (config.input_type == InputType::Depth) {
        edges.resize(depth_in.size);
        sobel->apply(depth_in, edges, config.sobel);
    } else {
        edges.resize(gray_in.size);
        sobel->apply(gray_in, edges, config.sobel);
    }

    // 4) Ergebnis speichern.
    return save_gray(output_path, edges);
}

} // namespace mgpu
