#include "Noise.h"
#include "ext/PerlinNoise.hpp"

PerlinNoiseGenerator::PerlinNoiseGenerator(int octaves, unsigned int seed)
    : octaves{octaves}
{
    noiseGenerator = new siv::BasicPerlinNoise<float>();

    static int iter = 0;
    noiseGenerator->reseed(seed + iter * 7);
    ++iter;
}

PerlinNoiseGenerator::~PerlinNoiseGenerator()
{
    delete noiseGenerator;
}

float PerlinNoiseGenerator::getNoise(float x, float y)
{
    return noiseGenerator->octave2D_01(x, y, octaves);
}