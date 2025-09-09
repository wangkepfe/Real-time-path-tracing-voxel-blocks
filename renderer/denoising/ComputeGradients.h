#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"

__global__ void ComputeGradients(
    Int2 screenResolution,
    Float2 invScreenResolution,

    // Current frame ReSTIR luminance data
    SurfObj restirLuminanceBuffer,
    SurfObj depthBuffer,
    SurfObj materialBuffer,
    SurfObj motionVectorBuffer,

    // Previous frame ReSTIR luminance data  
    SurfObj prevRestirLuminanceBuffer,

    // Output gradient buffers
    SurfObj diffuseGradientBuffer,
    SurfObj specularGradientBuffer,

    // Camera data 
    Camera camera,
    Camera prevCamera,
    unsigned int frameIndex)
{
    Int2 pixelPos;
    pixelPos.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelPos.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelPos.x >= screenResolution.x || pixelPos.y >= screenResolution.y)
        return;

    const int blockSize = 8;  // 8x8 blocks for stratum sampling
    const int log2BlockSize = 3;
    
    Int2 stratumOrigin = Int2((pixelPos.x >> log2BlockSize) << log2BlockSize, (pixelPos.y >> log2BlockSize) << log2BlockSize);

    // Select the pixel most likely to produce the largest gradient within the stratum
    float maxLuminance = 0.0f;
    Int2 selectedPixel = stratumOrigin;
    
    for (int yy = 0; yy < blockSize; ++yy)
    {
        for (int xx = 0; xx < blockSize; ++xx)
        {
            Int2 candidatePixel = stratumOrigin + Int2(xx, yy);
            if (candidatePixel.x >= screenResolution.x || candidatePixel.y >= screenResolution.y)
                continue;
                
            Float2 currentLuminanceVec = Load2DFloat2(restirLuminanceBuffer, candidatePixel);
            float currentLuminance = currentLuminanceVec.x + currentLuminanceVec.y; // Total luminance (diffuse + specular)
            
            if (currentLuminance > maxLuminance)
            {
                maxLuminance = currentLuminance;
                selectedPixel = candidatePixel;
            }
        }
    }

    // Load current frame ReSTIR luminance (float2 for diffuse+specular)
    Float2 currentLuminanceVec = Load2DFloat2(restirLuminanceBuffer, selectedPixel);
    float currentDiffuseLuminance = currentLuminanceVec.x;
    float currentSpecularLuminance = currentLuminanceVec.y;
    
    // Use motion vector to find corresponding previous pixel
    Float2 motionVector = Load2DFloat2(motionVectorBuffer, selectedPixel);
    Int2 prevPixel = Int2(Float2(selectedPixel.x, selectedPixel.y) + motionVector);
    
    float prevDiffuseLuminance = 0.0f;
    float prevSpecularLuminance = 0.0f;
    if (prevPixel.x >= 0 && prevPixel.y >= 0 && 
        prevPixel.x < screenResolution.x && prevPixel.y < screenResolution.y)
    {
        Float2 prevLuminanceVec = Load2DFloat2(prevRestirLuminanceBuffer, prevPixel);
        prevDiffuseLuminance = prevLuminanceVec.x;
        prevSpecularLuminance = prevLuminanceVec.y;
    }
    
    // Calculate gradients as absolute differences for diffuse and specular separately
    float diffuseGradient = abs(currentDiffuseLuminance - prevDiffuseLuminance);
    float specularGradient = abs(currentSpecularLuminance - prevSpecularLuminance);
    
    // Scale and clamp the gradients
    const float gradientScale = 1.0f;
    const float maxGradient = 10.0f;
    diffuseGradient *= gradientScale;
    specularGradient *= gradientScale;
    diffuseGradient = min(diffuseGradient, maxGradient);
    specularGradient = min(specularGradient, maxGradient);
    
    // Store gradients and current luminance for confidence computation
    // Format: (gradient, 0, currentLuminance, 0)
    Store2DFloat4(Float4(diffuseGradient, 0.0f, currentDiffuseLuminance, 0.0f), diffuseGradientBuffer, pixelPos);
    Store2DFloat4(Float4(specularGradient, 0.0f, currentSpecularLuminance, 0.0f), specularGradientBuffer, pixelPos);
}