#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"

__global__ void GenerateConfidence(
    Int2 screenResolution,
    Float2 invScreenResolution,

    // Input gradient buffer (filtered)
    SurfObj filteredGradientBuffer,
    
    // Previous confidence for temporal filtering
    SurfObj prevConfidenceBuffer,
    
    // Depth and material for motion validation
    SurfObj depthBuffer,
    SurfObj materialBuffer,

    // Output confidence buffer
    SurfObj confidenceBuffer,

    // Camera data for motion computation  
    Camera camera,
    Camera prevCamera,
    
    // Confidence parameters
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

    // Load current frame data
    float currentViewZ = Load2DFloat1(depthBuffer, pixelPos);
    
    // Early out if no valid surface
    if (currentViewZ > 500000.0f) {
        Store2DFloat1(0.0f, confidenceBuffer, pixelPos);
        return;
    }

    // Load gradient data (following RTXDI format: gradient, 0, currentLuminance, 0)
    Float4 gradientData = Load2DFloat4(filteredGradientBuffer, pixelPos);
    float gradient = gradientData.x;
    float currentLuminance = gradientData.z;
    
    // Add darkness bias to prevent division by zero
    const float darknessBias = 0.01f;
    currentLuminance = max(currentLuminance, darknessBias);
    
    // Convert gradient to confidence (RTXDI style)
    // confidence = 1 - (gradient / (luminance + bias))
    float diffuseConfidence = saturate(1.0f - gradient / currentLuminance);
    
    // Apply sensitivity power to adjust confidence curve
    const float sensitivity = 1.0f;
    diffuseConfidence = saturate(pow(diffuseConfidence, sensitivity));

    // Apply temporal filtering if enabled
    if (enableTemporalFiltering && frameIndex > 0)
    {
        // Compute motion vector using camera transforms (same as gradient computation)
        Float3 currentWorldPos = GetCurrentWorldPosFromPixelPos(camera, pixelPos, currentViewZ);
        Float2 previousUV = prevCamera.worldDirectionToUV(normalize(currentWorldPos - prevCamera.pos));
        Float2 previousPixelPos = previousUV * Float2(screenResolution.x, screenResolution.y) - 0.5f;
        
        // Check if previous position is valid
        if (previousPixelPos.x >= 0.5f && previousPixelPos.x < screenResolution.x - 0.5f &&
            previousPixelPos.y >= 0.5f && previousPixelPos.y < screenResolution.y - 0.5f)
        {
            // Sample previous confidence using bilinear interpolation
            float previousConfidence = BilinearSample2DFloat4(prevConfidenceBuffer, previousPixelPos).x;
            
            // Temporal filtering: blend with previous confidence
            diffuseConfidence = lerp(previousConfidence, diffuseConfidence, temporalWeight);
        }
    }

    // Store final confidence
    Store2DFloat1(diffuseConfidence, confidenceBuffer, pixelPos);
}

