#include "Noise.h"
#include "ext/PerlinNoise.hpp"

PerlinNoiseGenerator::PerlinNoiseGenerator(int octaves, unsigned int seed)
    : octaves{octaves}
{
    noiseGenerator = new siv::BasicPerlinNoise<float>();
    noiseGenerator->reseed(seed); // Use the exact seed provided, no auto-increment
}

PerlinNoiseGenerator::~PerlinNoiseGenerator()
{
    delete noiseGenerator;
}

float PerlinNoiseGenerator::getNoise(float x, float y)
{
    return noiseGenerator->octave2D_01(x, y, octaves);
}