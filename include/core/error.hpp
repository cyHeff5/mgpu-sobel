#pragma once

#include <string>

#include "core/types.hpp"

namespace mgpu {

// Einheitliche Fehlercodes fuer Pipeline/IO/Backend.
enum class ErrorCode {
    None = 0,
    InvalidArgument,
    IoError,
    NotSupported,
    CudaError,
    OpenMpError,
    Internal
};

// Wandelt Fehlercode in lesbaren Text um.
const char* to_string(ErrorCode code);

// Baut einen Status fuer erfolgreiche Operationen.
Status ok();

// Baut einen Status fuer fehlgeschlagene Operationen.
Status error(ErrorCode code, const std::string& message);

// Baut einen Status nur aus dem Fehlercode.
Status error(ErrorCode code);

} // namespace mgpu
