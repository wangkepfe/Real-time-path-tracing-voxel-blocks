#pragma once

#include "shaders/RandGen.h"

namespace jazzfusion
{

struct BlueNoiseRandGeneratorHost
{
    BlueNoiseRandGeneratorHost() { init(); }
    ~BlueNoiseRandGeneratorHost() { clear(); }

    void init();
    void clear();

    explicit operator BlueNoiseRandGenerator()
    {
        BlueNoiseRandGenerator randGen{};
        randGen.sobol_256spp_256d = sobol_256spp_256d;
        randGen.scramblingTile = scramblingTile;
#if OPTIMIZED_BLUE_NOISE_SPP != 1
        randGen.rankingTile = rankingTile;
#endif
        return randGen;
    }

    unsigned char* sobol_256spp_256d;
    unsigned char* scramblingTile;

#if OPTIMIZED_BLUE_NOISE_SPP != 1
    unsigned char* rankingTile;
#endif
};

}