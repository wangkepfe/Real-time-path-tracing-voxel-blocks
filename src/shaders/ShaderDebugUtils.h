#pragma once

#include "OptixShaderCommon.h"

#define OPTIX_DEBUG_PRINT(__VALUE__) OptixDebugPrint(__FILE__,__LINE__,#__VALUE__,__VALUE__);

namespace jazzfusion
{


__device__ inline void OptixDebugPrint(const char* file, int line, const char* valueName, float v)
{
    printf("%s:%d:%s:(%d,%d):%f\n", file, line, valueName, optixGetLaunchIndex().x, optixGetLaunchIndex().y, v);
}

__device__ inline void OptixDebugPrint(const char* file, int line, const char* valueName, const Float2& v)
{
    printf("%s:%d:%s:(%d,%d):(%f,%f)\n", file, line, valueName, optixGetLaunchIndex().x, optixGetLaunchIndex().y, v.x, v.y);
}

__device__ inline void OptixDebugPrint(const char* file, int line, const char* valueName, const Float3& v)
{
    printf("%s:%d:%s:(%d,%d):(%f,%f,%f)\n", file, line, valueName, optixGetLaunchIndex().x, optixGetLaunchIndex().y, v.x, v.y, v.z);
}

__device__ inline void OptixDebugPrint(const char* file, int line, const char* valueName, const Float4& v)
{
    printf("%s:%d:%s:(%d,%d):(%f,%f,%f,%f)\n", file, line, valueName, optixGetLaunchIndex().x, optixGetLaunchIndex().y, v.x, v.y, v.z, v.w);
}

__device__ inline void OptixDebugPrint(const char* file, int line, const char* valueName, int v)
{
    printf("%s:%d:%s:(%d,%d):%d\n", file, line, valueName, optixGetLaunchIndex().x, optixGetLaunchIndex().y, v);
}

__device__ inline void OptixDebugPrint(const char* file, int line, const char* valueName, const Int2& v)
{
    printf("%s:%d:%s:(%d,%d):(%d,%d)\n", file, line, valueName, optixGetLaunchIndex().x, optixGetLaunchIndex().y, v.x, v.y);
}

__device__ inline void OptixDebugPrint(const char* file, int line, const char* valueName, const Int3& v)
{
    printf("%s:%d:%s:(%d,%d):(%d,%d,%d)\n", file, line, valueName, optixGetLaunchIndex().x, optixGetLaunchIndex().y, v.x, v.y, v.z);
}

__device__ inline void OptixDebugPrint(const char* file, int line, const char* valueName, const Int4& v)
{
    printf("%s:%d:%s:(%d,%d):(%d,%d,%d,%d)\n", file, line, valueName, optixGetLaunchIndex().x, optixGetLaunchIndex().y, v.x, v.y, v.z, v.w);
}

}