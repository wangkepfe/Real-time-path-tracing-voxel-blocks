#pragma once

#include "LinearMath.h"

namespace jazzfusion
{

struct BlueNoiseRandGenerator
{
    __host__ __device__ BlueNoiseRandGenerator() {}

    // device copy constructor
    __host__ __device__ BlueNoiseRandGenerator(const BlueNoiseRandGenerator& randGen)
        : sobol_256spp_256d{ randGen.sobol_256spp_256d },
        scramblingTile{ randGen.scramblingTile },
#if OPTIMIZED_BLUE_NOISE_SPP != 1
        rankingTile{ randGen.rankingTile },
#endif
        idx{ randGen.idx },
        sampleIdx{ randGen.sampleIdx }
    {}

    // rand for single float
    __device__ float Rand(int sampleDim)
    {
        return samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_8spp(idx.x, idx.y, sampleIdx, sampleDim);
    }

    // rand for float pair
    __device__ Float2 Rand2(int sampleDim)
    {
        return Float2(Rand(sampleDim),
            Rand(sampleDim + 1));
    }

    // rand for float pair
    __device__ Float4 Rand4(int sampleDim)
    {
        return Float4(Rand(sampleDim), Rand(sampleDim + 1), Rand(sampleDim + 2), Rand(sampleDim + 3));
    }

    // rand implementation
    __device__ float samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_8spp(int pixel_i, int pixel_j, int sampleIndex, int sampleDimension)
    {
        // wrap arguments
        pixel_i = pixel_i & 127;
        pixel_j = pixel_j & 127;
        sampleIndex = sampleIndex & 255;

        // xor index based on optimized ranking
#if OPTIMIZED_BLUE_NOISE_SPP != 1
        int rankedSampleIndex = sampleIndex ^ rankingTile[sampleDimension + (pixel_i + pixel_j * 128) * 8];
#else
        int rankedSampleIndex = sampleIndex;
#endif

        // fetch value in sequence
        int value = sobol_256spp_256d[sampleDimension + rankedSampleIndex * 256];

        // If the dimension is optimized, xor sequence value based on optimized scrambling
        value = value ^ scramblingTile[(sampleDimension % 8) + (pixel_i + pixel_j * 128) * 8];

        // convert to float and return
        float v = (value + 0.5f) / 256.0f;

        // for comparison
        // v = WangHashItoF(sampleIndex * 128 * 128 * 8 + (pixel_i + pixel_j * 128) * 8 + sampleDimension);

        return v;
    }

    unsigned char* sobol_256spp_256d;
    unsigned char* scramblingTile;
#if OPTIMIZED_BLUE_NOISE_SPP != 1
    unsigned char* rankingTile;
#endif

    Int2 idx;
    int sampleIdx;
};

}