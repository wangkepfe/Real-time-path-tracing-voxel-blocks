#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"

__global__ void GenerateConfidence(
    Int2 screenResolution,
    Float2 invScreenResolution,

    // Current frame gradient data
    SurfObj diffuseGradientBuffer,
    SurfObj specularGradientBuffer,
    SurfObj prevDiffuseConfidenceBuffer,
    SurfObj prevSpecularConfidenceBuffer,
    SurfObj depthBuffer,
    SurfObj materialBuffer,

    // Output confidence buffers
    SurfObj diffuseConfidenceBuffer,
    SurfObj specularConfidenceBuffer,

    // Camera data 
    Camera camera,
    Camera prevCamera,

    // Parameters
    float gradientScale,
    bool enableTemporalFiltering,
    float temporalWeight,
    unsigned int frameIndex)
{
    Int2 pixelPos;
    pixelPos.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelPos.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelPos.x >= screenResolution.x || pixelPos.y >= screenResolution.y)
        return;

    // Load current frame gradients
    Float4 diffuseGradient = Load2DFloat4(diffuseGradientBuffer, pixelPos);
    Float4 specularGradient = Load2DFloat4(specularGradientBuffer, pixelPos);
    
    // Extract gradient values (stored in x component)
    float currentDiffuseGradient = diffuseGradient.x * gradientScale;
    float currentSpecularGradient = specularGradient.x * gradientScale;
    
    // Compute confidence as inverse of gradient magnitude
    // Higher gradients = lower confidence, lower gradients = higher confidence
    float diffuseConfidence = 1.0f / (1.0f + currentDiffuseGradient);
    float specularConfidence = 1.0f / (1.0f + currentSpecularGradient);
    
    // Temporal filtering with previous confidence if enabled
    if (enableTemporalFiltering && frameIndex > 0)
    {
        // Use motion vectors to find corresponding previous pixel
        Float2 motionVector = Float2(0.0f); // TODO: Load from motion vector buffer if available
        Int2 prevPixel = pixelPos + Int2(motionVector.x, motionVector.y);
        
        if (prevPixel.x >= 0 && prevPixel.y >= 0 && 
            prevPixel.x < screenResolution.x && prevPixel.y < screenResolution.y)
        {
            float prevDiffuseConfidence = Load2DFloat1(prevDiffuseConfidenceBuffer, prevPixel);
            float prevSpecularConfidence = Load2DFloat1(prevSpecularConfidenceBuffer, prevPixel);
            
            // Blend with previous confidence
            diffuseConfidence = lerp(prevDiffuseConfidence, diffuseConfidence, temporalWeight);
            specularConfidence = lerp(prevSpecularConfidence, specularConfidence, temporalWeight);
        }
    }
    
    // Clamp confidence values
    diffuseConfidence = saturate(diffuseConfidence);
    specularConfidence = saturate(specularConfidence);
    
    // Store confidence values
    Store2DFloat1(diffuseConfidence, diffuseConfidenceBuffer, pixelPos);
    Store2DFloat1(specularConfidence, specularConfidenceBuffer, pixelPos);
}