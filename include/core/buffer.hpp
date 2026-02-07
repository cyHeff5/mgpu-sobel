#pragma once

#include <cstddef>
#include <cstdint>

namespace mgpu {

// Einfacher, typfreier Speicher-Wrapper (Pointer + Groesse in Bytes).
struct BufferView {
    void* data = nullptr;
    std::size_t bytes = 0;
};

// Const-Variante fuer Read-Only Zugriff.
struct ConstBufferView {
    const void* data = nullptr;
    std::size_t bytes = 0;
};

// Helfer fuer Byte-Groessen bei Count * Elementgroesse.
inline std::size_t bytes_for(std::size_t count, std::size_t element_size) {
    return count * element_size;
}

// Pointer-Sicht als const.
inline ConstBufferView as_const(BufferView view) {
    return ConstBufferView{view.data, view.bytes};
}

} // namespace mgpu
