#pragma once

#include "LinearMath.h"

namespace jazzfusion
{

struct BlueNoiseRandGenerator
{
    INL_HOST_DEVICE BlueNoiseRandGenerator() {}

    // device copy constructor
    INL_HOST_DEVICE BlueNoiseRandGenerator(const BlueNoiseRandGenerator& randGen)
        : sobol_256spp_256d{ randGen.sobol_256spp_256d },
        scramblingTile{ randGen.scramblingTile },
#if OPTIMIZED_BLUE_NOISE_SPP != 1
        rankingTile{ randGen.rankingTile }
#endif
    {}

    // rand implementation
    INL_DEVICE float rand(int pixel_i, int pixel_j, int sampleIndex, int sampleDimension) const
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

        return v;
    }

    unsigned char* sobol_256spp_256d;
    unsigned char* scramblingTile;
#if OPTIMIZED_BLUE_NOISE_SPP != 1
    unsigned char* rankingTile;
#endif
};

}