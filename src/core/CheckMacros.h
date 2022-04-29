

#pragma once

#ifndef CHECK_MACROS_H
#define CHECK_MACROS_H

#include "core/MyAssert.h"

#define CUDA_CHECK(call) \
{ \
  const cudaError_t error = call; \
  if (error != cudaSuccess) \
  { \
    std::cerr << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " failed with code " << error << ": " << cudaGetErrorString(error) << '\n'; \
    MY_ASSERT(!"CUDA_CHECK fatal"); \
  } \
}

#define OPTIX_CHECK(call) \
{ \
  const OptixResult result = call; \
  if (result != OPTIX_SUCCESS) \
  { \
    std::cerr << "ERROR: " << __FILE__ << "(" << __LINE__ << "): " << #call << " failed with (" << result << ")\n"; \
    MY_ASSERT(!"OPTIX_CHECK fatal"); \
  }\
}

#endif // CHECK_MACROS_H
