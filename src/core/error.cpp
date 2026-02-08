#include "core/error.hpp"

namespace mgpu {

const char* to_string(ErrorCode code) {
    // Stabiler Text fuer Logs/CLI-Ausgaben.
    switch (code) {
    case ErrorCode::None:
        return "NoError";
    case ErrorCode::InvalidArgument:
        return "InvalidArgument";
    case ErrorCode::IoError:
        return "IoError";
    case ErrorCode::NotSupported:
        return "NotSupported";
    case ErrorCode::CudaError:
        return "CudaError";
    case ErrorCode::OpenMpError:
        return "OpenMpError";
    case ErrorCode::Internal:
        return "Internal";
    default:
        return "UnknownError";
    }
}

Status ok() {
    // Erfolgsstatus ohne Zusatztext.
    return Status{true, std::string{}};
}

Status error(ErrorCode code, const std::string& message) {
    if (message.empty()) {
        return Status{false, to_string(code)};
    }

    std::string text = to_string(code);
    // Fehlercode und Detailtext kombinieren, damit Call-Sites nur einmal formatieren muessen.
    text.append(": ");
    text.append(message);
    return Status{false, text};
}

Status error(ErrorCode code) {
    // Standardfehler ohne Detailtext.
    return Status{false, to_string(code)};
}

} // namespace mgpu
