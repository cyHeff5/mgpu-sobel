#include "io/image_io.hpp"

#include <cctype>
#include <fstream>
#include <sstream>

#include "core/error.hpp"
#include "core/math.hpp"

namespace mgpu {

namespace {

// Liest das naechste "Wort" aus dem Stream (ignoriert Whitespace und Kommentare).
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

// Liest den gemeinsamen PNM-Header (P5/P6): Magic, Breite, Hoehe, Maxval.
Status load_pnm_header(std::istream& in, std::string& magic, int& width, int& height, int& maxval) {
    std::string token;
    if (!read_token(in, magic)) {
        return error(ErrorCode::IoError, "PNM Header: Magic fehlt");
    }
    if (!read_token(in, token)) {
        return error(ErrorCode::IoError, "PNM Header: Breite fehlt");
    }
    width = std::stoi(token);
    if (!read_token(in, token)) {
        return error(ErrorCode::IoError, "PNM Header: Hoehe fehlt");
    }
    height = std::stoi(token);
    if (!read_token(in, token)) {
        return error(ErrorCode::IoError, "PNM Header: Maxval fehlt");
    }
    maxval = std::stoi(token);
    return ok();
}

// Schreibt ein Graubild als binaeres PGM (P5).
Status write_pgm(const std::string& path, const GrayImage& image) {
    if (image.size.width <= 0 || image.size.height <= 0) {
        return error(ErrorCode::InvalidArgument, "Ungueltige Bildgroesse");
    }
    const int expected_stride = image.size.width;
    if (image.stride < expected_stride) {
        return error(ErrorCode::InvalidArgument, "Ungueltiger Gray-Stride");
    }
    if (image.data.size() < static_cast<size_t>(image.stride * image.size.height)) {
        return error(ErrorCode::InvalidArgument, "Gray-Daten zu klein");
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return error(ErrorCode::IoError, "Datei konnte nicht geoeffnet werden");
    }

    // PGM-Header: Format, Groesse, Maxval.
    out << "P5\n" << image.size.width << " " << image.size.height << "\n255\n";
    for (int y = 0; y < image.size.height; ++y) {
        const std::uint8_t* row = image.data.data() + y * image.stride;
        // Pro Zeile genau "width" Bytes schreiben (ohne Padding).
        out.write(reinterpret_cast<const char*>(row), image.size.width);
    }
    return ok();
}

// Schreibt ein RGB-Bild als binaeres PPM (P6).
Status write_ppm(const std::string& path, const RgbImage& image) {
    if (image.size.width <= 0 || image.size.height <= 0) {
        return error(ErrorCode::InvalidArgument, "Ungueltige Bildgroesse");
    }
    const int expected_stride = image.size.width * 3;
    if (image.stride < expected_stride) {
        return error(ErrorCode::InvalidArgument, "Ungueltiger RGB-Stride");
    }
    if (image.data.size() < static_cast<size_t>(image.stride * image.size.height)) {
        return error(ErrorCode::InvalidArgument, "RGB-Daten zu klein");
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return error(ErrorCode::IoError, "Datei konnte nicht geoeffnet werden");
    }

    // PPM-Header: Format, Groesse, Maxval.
    out << "P6\n" << image.size.width << " " << image.size.height << "\n255\n";
    for (int y = 0; y < image.size.height; ++y) {
        const std::uint8_t* row = image.data.data() + y * image.stride;
        // Pro Zeile width*3 Bytes schreiben (RGB interleaved).
        out.write(reinterpret_cast<const char*>(row), image.size.width * 3);
    }
    return ok();
}

} // namespace

// Laedt ein RGB-Bild aus einem PPM (P6, 8-bit, maxval=255).
Status load_rgb(const std::string& path, RgbImage& image) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return error(ErrorCode::IoError, "Datei konnte nicht geoeffnet werden");
    }

    std::string magic;
    int width = 0;
    int height = 0;
    int maxval = 0;
    const Status header = load_pnm_header(in, magic, width, height, maxval);
    if (!header.ok) {
        return header;
    }
    if (magic != "P6") {
        return error(ErrorCode::NotSupported, "Nur P6 (PPM) wird unterstuetzt");
    }
    if (maxval != 255) {
        return error(ErrorCode::NotSupported, "Nur maxval=255 wird unterstuetzt");
    }
    if (width <= 0 || height <= 0) {
        return error(ErrorCode::InvalidArgument, "Ungueltige Bildgroesse");
    }

    // Speicher passend zur Bildgroesse vorbereiten.
    image.resize(Size2D{width, height});
    const size_t bytes = static_cast<size_t>(width * height * 3);
    // Rohdaten direkt in den Buffer lesen.
    in.read(reinterpret_cast<char*>(image.data.data()), static_cast<std::streamsize>(bytes));
    if (!in) {
        return error(ErrorCode::IoError, "PPM Daten konnten nicht gelesen werden");
    }
    return ok();
}

// Laedt ein Graubild aus einem PGM (P5, 8-bit, maxval=255).
Status load_gray(const std::string& path, GrayImage& image) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return error(ErrorCode::IoError, "Datei konnte nicht geoeffnet werden");
    }

    std::string magic;
    int width = 0;
    int height = 0;
    int maxval = 0;
    const Status header = load_pnm_header(in, magic, width, height, maxval);
    if (!header.ok) {
        return header;
    }
    if (magic != "P5") {
        return error(ErrorCode::NotSupported, "Nur P5 (PGM) wird unterstuetzt");
    }
    if (maxval != 255) {
        return error(ErrorCode::NotSupported, "Nur maxval=255 wird unterstuetzt");
    }
    if (width <= 0 || height <= 0) {
        return error(ErrorCode::InvalidArgument, "Ungueltige Bildgroesse");
    }

    // Speicher passend zur Bildgroesse vorbereiten.
    image.resize(Size2D{width, height});
    const size_t bytes = static_cast<size_t>(width * height);
    // Rohdaten direkt in den Buffer lesen.
    in.read(reinterpret_cast<char*>(image.data.data()), static_cast<std::streamsize>(bytes));
    if (!in) {
        return error(ErrorCode::IoError, "PGM Daten konnten nicht gelesen werden");
    }
    return ok();
}

// Oeffentliche Save-Funktion fuer Gray -> PGM.
Status save_gray(const std::string& path, const GrayImage& image) {
    return write_pgm(path, image);
}

// Oeffentliche Save-Funktion fuer RGB -> PPM.
Status save_rgb(const std::string& path, const RgbImage& image) {
    return write_ppm(path, image);
}

// Konvertiert RGB nach Gray ueber die Luminanzformel (siehe core/math.hpp).
Status rgb_to_gray(const RgbImage& input, GrayImage& output) {
    if (input.size.width <= 0 || input.size.height <= 0) {
        return error(ErrorCode::InvalidArgument, "Ungueltige Bildgroesse");
    }

    const int expected_stride = input.size.width * 3;
    if (input.stride < expected_stride) {
        return error(ErrorCode::InvalidArgument, "Ungueltiger RGB-Stride");
    }

    // Ausgabe auf gleiche Groesse bringen.
    output.resize(input.size);

    for (int y = 0; y < input.size.height; ++y) {
        const std::uint8_t* src = input.data.data() + y * input.stride;
        std::uint8_t* dst = output.data.data() + y * output.stride;
        for (int x = 0; x < input.size.width; ++x) {
            const std::uint8_t r = src[x * 3 + 0];
            const std::uint8_t g = src[x * 3 + 1];
            const std::uint8_t b = src[x * 3 + 2];
            // Einzelpixel RGB -> Gray.
            dst[x] = to_gray(r, g, b);
        }
    }

    return ok();
}

} // namespace mgpu
