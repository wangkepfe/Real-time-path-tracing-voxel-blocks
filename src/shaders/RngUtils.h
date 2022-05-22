#pragma once
#include "ShaderCommon.h"

// Tiny Encryption Algorithm (TEA) to calculate a the seed per launch index and iteration.
// This results in a ton of integer instructions! Use the smallest N necessary.
template<unsigned int N>
INL_DEVICE unsigned int tea(const unsigned int val0, const unsigned int val1)
{
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for (unsigned int n = 0; n < N; ++n)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xA341316C) ^ (v1 + s0) ^ ((v1 >> 5) + 0xC8013EA4);
        v1 += ((v0 << 4) + 0xAD90777D) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7E95761E);
    }
    return v0;
}

// Return a random sample in the range [0, 1) with a simple Linear Congruential Generator.
INL_DEVICE float rng(unsigned int& previous)
{
    previous = previous * 1664525u + 1013904223u;

    return float(previous & 0x00FFFFFF) / float(0x01000000u); // Use the lower 24 bits.
    // return float(previous >> 8) / float(0x01000000u);      // Use the upper 24 bits
}

// Convenience function to generate a 2D unit square sample.
INL_DEVICE float2 rng2(unsigned int& previous)
{
    float2 s;

    previous = previous * 1664525u + 1013904223u;
    s.x = float(previous & 0x00FFFFFF) / float(0x01000000u); // Use the lower 24 bits.
    //s.x = float(previous >> 8) / float(0x01000000u);      // Use the upper 24 bits

    previous = previous * 1664525u + 1013904223u;
    s.y = float(previous & 0x00FFFFFF) / float(0x01000000u); // Use the lower 24 bits.
    //s.y = float(previous >> 8) / float(0x01000000u);      // Use the upper 24 bits

    return s;
}
