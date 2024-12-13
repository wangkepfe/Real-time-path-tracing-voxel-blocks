#pragma once

#include "OptixShaderCommon.h"
#include "stdio.h"

#define COMMA ,

#define OPTIX_LEFT_HALF_SCREEN() (optixGetLaunchIndex().x < optixGetLaunchDimensions().x * 0.5f)
#define OPTIX_CENTER_PIXEL() (optixGetLaunchIndex().x == optixGetLaunchDimensions().x * 0.5f) && (optixGetLaunchIndex().y == optixGetLaunchDimensions().y * 0.5f)
#define OPTIX_DEBUG_PRINT(__VALUE__) OptixDebugPrint(__FILE__, __LINE__, #__VALUE__, __VALUE__);

#define CUDA_LEFT_HALF_SCREEN() (blockIdx.x * blockDim.x + threadIdx.x < gridDim.x * blockDim.x * 0.5f)
#define CUDA_CENTER_PIXEL() (blockIdx.x * blockDim.x + threadIdx.x == gridDim.x * blockDim.x * 0.5f && blockIdx.y * blockDim.y + threadIdx.y == gridDim.y * blockDim.y * 0.5f)
#define DEBUG_PRINT(__VALUE__) CudaDebugPrint(__FILE__, __LINE__, #__VALUE__, __VALUE__);

#define OPTIX_DEBUG_PRINT_IMPL(__ARG__, __PRINT_STR__, __PRINT_ARG__)                                                                                                      \
    INL_DEVICE void OptixDebugPrint(const char *file, int line, const char *valueName, __ARG__)                                                                            \
    {                                                                                                                                                                      \
        printf("%s:%d:%s:(%d,%d):" __PRINT_STR__ "\n", file, line, valueName, optixGetLaunchIndex().x, optixGetLaunchIndex().y, __PRINT_ARG__);                            \
    }                                                                                                                                                                      \
    INL_DEVICE void CudaDebugPrint(const char *file, int line, const char *valueName, __ARG__)                                                                             \
    {                                                                                                                                                                      \
        printf("%s:%d:%s:(%d,%d):" __PRINT_STR__ "\n", file, line, valueName, blockIdx.x *blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, __PRINT_ARG__); \
    }

namespace jazzfusion
{

    OPTIX_DEBUG_PRINT_IMPL(float v, "%f", v)
    OPTIX_DEBUG_PRINT_IMPL(const Float2 &v, "(%f,%f)", v.x COMMA v.y)
    OPTIX_DEBUG_PRINT_IMPL(const Float3 &v, "(%f,%f,%f)", v.x COMMA v.y COMMA v.z)
    OPTIX_DEBUG_PRINT_IMPL(const Float4 &v, "(%f,%f,%f,%f)", v.x COMMA v.y COMMA v.z COMMA v.w)
    OPTIX_DEBUG_PRINT_IMPL(int v, "%d", v)
    OPTIX_DEBUG_PRINT_IMPL(const Int2 &v, "(%d,%d)", v.x COMMA v.y)
    OPTIX_DEBUG_PRINT_IMPL(const Int3 &v, "(%d,%d,%d)", v.x COMMA v.y COMMA v.z)
    OPTIX_DEBUG_PRINT_IMPL(const Int4 &v, "(%d,%d,%d,%d)", v.x COMMA v.y COMMA v.z COMMA v.w)
    OPTIX_DEBUG_PRINT_IMPL(uint v, "%d", v)
    OPTIX_DEBUG_PRINT_IMPL(const UInt2 &v, "(%d,%d)", v.x COMMA v.y)
    OPTIX_DEBUG_PRINT_IMPL(const UInt3 &v, "(%d,%d,%d)", v.x COMMA v.y COMMA v.z)

}