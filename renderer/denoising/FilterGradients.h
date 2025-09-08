#pragma once

#include "denoising/DenoiserCommon.h"

__global__ void FilterGradients(
    Int2 screenResolution,
    Float2 invScreenResolution,

    // Input gradient buffer
    SurfObj inputGradientBuffer,
    
    // Output filtered gradient buffer
    SurfObj outputGradientBuffer,

    // Filter parameters
    int filterRadius,
    int stepSize)
{
    Int2 pixelPos;
    pixelPos.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelPos.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelPos.x >= screenResolution.x || pixelPos.y >= screenResolution.y)
        return;

    Float3 filteredGradient = Float3(0.0f, 0.0f, 0.0f);
    float totalWeight = 0.0f;

    // A-trous spatial filter with triangular kernel
    for (int y = -filterRadius; y <= filterRadius; y++)
    {
        for (int x = -filterRadius; x <= filterRadius; x++)
        {
            Int2 sampleOffset = Int2(x, y) * stepSize;
            Int2 samplePos = pixelPos + sampleOffset;

            // Bounds check
            if (samplePos.x < 0 || samplePos.y < 0 || 
                samplePos.x >= screenResolution.x || samplePos.y >= screenResolution.y)
                continue;

            Float4 sampleGradient = Load2DFloat4(inputGradientBuffer, samplePos);
            
            // Triangular filter weight
            float weight = max(0.0f, 1.0f - abs(float(x)) / (filterRadius + 1.0f)) *
                          max(0.0f, 1.0f - abs(float(y)) / (filterRadius + 1.0f));

            filteredGradient += sampleGradient.xyz * weight;
            totalWeight += weight;
        }
    }

    // Normalize by total weight
    if (totalWeight > 0.0f)
        filteredGradient /= totalWeight;

    // Store filtered gradient
    Store2DFloat4(Float4(filteredGradient.x, filteredGradient.y, filteredGradient.z, 1.0f), 
                  outputGradientBuffer, pixelPos);
}