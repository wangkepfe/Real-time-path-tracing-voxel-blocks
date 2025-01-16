#pragma once

#include <memory>

namespace siv
{
    template <class T>
    class BasicPerlinNoise;
}

class NoiseGenerator
{
public:
    virtual float getNoise(float x, float y) = 0;
};

class PerlinNoiseGenerator : public NoiseGenerator
{
public:
    PerlinNoiseGenerator(int octaves = 4, unsigned int seed = 124);
    ~PerlinNoiseGenerator();
    float getNoise(float x, float y) override;

private:
    int octaves;

    siv::BasicPerlinNoise<float> *noiseGenerator;
};