

#pragma once

#include "cassert"
#include <iostream>

namespace jazzfusion
{

#define CUDA_CHECK(call)                                                                                                                                           \
    {                                                                                                                                                              \
        const cudaError_t error = call;                                                                                                                            \
        if (error != cudaSuccess)                                                                                                                                  \
        {                                                                                                                                                          \
            std::cerr << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " failed with code " << error << ": " << cudaGetErrorString(error) << '\n'; \
            assert(!"CUDA_CHECK fatal");                                                                                                                           \
        }                                                                                                                                                          \
    }

#define OPTIX_CHECK(call)                                                                                                   \
    {                                                                                                                       \
        const OptixResult result = call;                                                                                    \
        if (result != OPTIX_SUCCESS)                                                                                        \
        {                                                                                                                   \
            std::cerr << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " failed with (" << result << ")\n"; \
            assert(!"OPTIX_CHECK fatal");                                                                                   \
        }                                                                                                                   \
    }

}