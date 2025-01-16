#pragma once

#include <cuda_runtime.h>

#ifndef INL_HOST_DEVICE
#define INL_HOST_DEVICE __forceinline__ __host__ __device__
#endif

#ifndef INL_DEVICE
#define INL_DEVICE __forceinline__ __device__
#endif

#ifndef SurfObj
#define SurfObj cudaSurfaceObject_t
#endif

#ifndef TexObj
#define TexObj cudaTextureObject_t
#endif

#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>

#define OPTIMIZED_BLUE_NOISE_SPP 4

namespace jazzfusion
{

    static constexpr float RayMax = 1.0e27f;
    static constexpr float RayMaxLowerBound = 1.0e26f;

    // Specular materials only have sample function, they do not have evaluate function
    enum FunctionIndexSpecular
    {
        INDEX_BSDF_SPECULAR_REFLECTION = 0,
        INDEX_BSDF_SPECULAR_REFLECTION_TRANSMISSION = 1,
        NUM_SPECULAR_BSDF = 2,
    };

    // Diffuse materials have sample function at the index, and the evaluate function at index + 1
    enum FunctionIndexDiffuse
    {
        INDEX_BSDF_DIFFUSE_REFLECTION = 2,
        INDEX_BSDF_MICROFACET_REFLECTION = 4,
        INDEX_BSDF_DIFFUSE_REFLECTION_TRANSMISSION_THINFILM = 6,
    };

    static constexpr unsigned int RAY_MAT_FLAG_SKY = 0u;
    static constexpr unsigned int RAY_MAT_FLAG_REFR_AND_REFL = 1u;
    static constexpr unsigned int RAY_MAT_FLAG_REFL_OR_REFR = 2u;
    static constexpr unsigned int RAY_MAT_FLAG_DIFFUSE = 3u;

}
