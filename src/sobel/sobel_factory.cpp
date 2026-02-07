#include "sobel/sobel.hpp"

#include <memory>

#include "sobel/sobel_cpu.hpp"
#include "sobel/sobel_omp.hpp"

#if defined(MGPU_USE_CUDA)
#include "sobel/sobel_cuda_naive.hpp"
#include "sobel/sobel_cuda_tiled.hpp"
#endif

namespace mgpu {

std::unique_ptr<ISobelOperator> CreateSobelOperator(Backend backend) {
    switch (backend) {
    case Backend::Cpu:
        return std::make_unique<SobelCpu>();
    case Backend::Omp:
        return std::make_unique<SobelOmp>();
#if defined(MGPU_USE_CUDA)
    case Backend::CudaNaive:
        return std::make_unique<SobelCudaNaive>();
    case Backend::CudaTiled:
        return std::make_unique<SobelCudaTiled>();
#endif
    default:
        return nullptr;
    }
}

} // namespace mgpu
