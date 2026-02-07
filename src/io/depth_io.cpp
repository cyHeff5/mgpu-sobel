#include "io/depth_io.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>

#include "core/error.hpp"

namespace mgpu {

namespace {

bool read_token(std::istream& in, std::string& token) {
    token.clear();
    char c = 0;
    while (in.get(c)) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            continue;
        }
        if (c == '#') {
            std::string line;
            std::getline(in, line);
            continue;
        }
        token.push_back(c);
        break;
    }
    if (token.empty()) {
        return false;
    }
    while (in.get(c)) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            break;
        }
        token.push_back(c);
    }
    return true;
}

bool is_little_endian() {
    std::uint16_t v = 1;
    return *reinterpret_cast<std::uint8_t*>(&v) == 1;
}

void swap_float_endianness(float& v) {
    auto* bytes = reinterpret_cast<std::uint8_t*>(&v);
    std::swap(bytes[0], bytes[3]);
    std::swap(bytes[1], bytes[2]);
}

} // namespace

Status load_depth(const std::string& path, DepthMap& depth) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return error(ErrorCode::IoError, "Datei konnte nicht geoeffnet werden");
    }

    std::string magic;
    std::string token;
    if (!read_token(in, magic)) {
        return error(ErrorCode::IoError, "PFM Header: Magic fehlt");
    }
    if (magic != "Pf") {
        return error(ErrorCode::NotSupported, "Nur Pf (PFM grayscale) wird unterstuetzt");
    }
    if (!read_token(in, token)) {
        return error(ErrorCode::IoError, "PFM Header: Breite fehlt");
    }
    const int width = std::stoi(token);
    if (!read_token(in, token)) {
        return error(ErrorCode::IoError, "PFM Header: Hoehe fehlt");
    }
    const int height = std::stoi(token);
    if (!read_token(in, token)) {
        return error(ErrorCode::IoError, "PFM Header: Scale fehlt");
    }
    const float scale = std::stof(token);
    if (width <= 0 || height <= 0) {
        return error(ErrorCode::InvalidArgument, "Ungueltige Bildgroesse");
    }

    const bool file_little_endian = (scale < 0.0f);
    const bool need_swap = file_little_endian != is_little_endian();

    depth.resize(Size2D{width, height});
    const size_t count = static_cast<size_t>(width * height);
    in.read(reinterpret_cast<char*>(depth.data.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!in) {
        return error(ErrorCode::IoError, "PFM Daten konnten nicht gelesen werden");
    }

    if (need_swap) {
        for (float& v : depth.data) {
            swap_float_endianness(v);
        }
    }

    return ok();
}

Status save_depth(const std::string& path, const DepthMap& depth) {
    if (depth.size.width <= 0 || depth.size.height <= 0) {
        return error(ErrorCode::InvalidArgument, "Ungueltige Bildgroesse");
    }
    if (depth.stride < depth.size.width) {
        return error(ErrorCode::InvalidArgument, "Ungueltiger Depth-Stride");
    }
    if (depth.data.size() < static_cast<size_t>(depth.stride * depth.size.height)) {
        return error(ErrorCode::InvalidArgument, "Depth-Daten zu klein");
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return error(ErrorCode::IoError, "Datei konnte nicht geoeffnet werden");
    }

    // PFM: negative scale bedeutet little-endian.
    out << "Pf\n" << depth.size.width << " " << depth.size.height << "\n-1.0\n";
    for (int y = 0; y < depth.size.height; ++y) {
        const float* row = depth.data.data() + y * depth.stride;
        out.write(reinterpret_cast<const char*>(row),
                  static_cast<std::streamsize>(depth.size.width * sizeof(float)));
    }
    return ok();
}

} // namespace mgpu
