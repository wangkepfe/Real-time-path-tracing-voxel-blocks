#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"

__global__ void BufferCopySky(
    Int2 screenResolution,
    SurfObj inBuffer,
    SurfObj depthBuffer,
    SurfObj outBuffer)
{
    Int2 threadPos;
    threadPos.x = threadIdx.x;
    threadPos.y = threadIdx.y;

    Int2 pixelPos;
    pixelPos.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelPos.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelPos.x >= screenResolution.x || pixelPos.y >= screenResolution.y)
    {
        return;
    }

    float centerViewZ = Load2DFloat1(depthBuffer, pixelPos);

    // Early out
    const float denoisingRange = 500000.0f;
    if (centerViewZ <= denoisingRange)
        return;

    Float4 val = Load2DFloat4(inBuffer, pixelPos);
    Store2DFloat4(val, outBuffer, pixelPos);
}

__global__ void BufferCopyNonSky(
    Int2 screenResolution,
    SurfObj inBuffer,
    SurfObj depthBuffer,
    SurfObj albedoBuffer,
    SurfObj uiBuffer,
    SurfObj outBuffer)
{
    Int2 threadPos;
    threadPos.x = threadIdx.x;
    threadPos.y = threadIdx.y;

    Int2 pixelPos;
    pixelPos.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelPos.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelPos.x >= screenResolution.x || pixelPos.y >= screenResolution.y)
    {
        return;
    }

    float centerViewZ = Load2DFloat1(depthBuffer, pixelPos);

    // Early out
    const float denoisingRange = 500000.0f;
    if (centerViewZ > denoisingRange)
        return;

    if (0)
    {
        Float3 illum = Load2DFloat4(inBuffer, pixelPos).xyz;
        Float3 albedo = Load2DFloat4(albedoBuffer, pixelPos).xyz;
        Float3 uiOverlay = Load2DFloat4(uiBuffer, pixelPos).xyz;
        Float3 uiColor = Float3(0.0f);

        Float3 out = uiOverlay.x > 0.0f ? uiColor : (illum * albedo);

        Store2DFloat4(Float4(out, 0.0f), outBuffer, pixelPos);
    }

    if (1)
    {
        Float3 illum = Load2DFloat4(inBuffer, pixelPos).xyz;
        Float3 albedo = Load2DFloat4(albedoBuffer, pixelPos).xyz;
        Store2DFloat4(Float4(illum * albedo, 0.0f), outBuffer, pixelPos);
    }

    // Visualize albedo
    if (0)
    {
        Float3 illum = Load2DFloat4(inBuffer, pixelPos).xyz;
        Store2DFloat4(Float4(illum, 0.0f), outBuffer, pixelPos);
    }

    // Visualize depth
    if (0)
    {
        float debugVisualization = Load2DFloat1(inBuffer, pixelPos);
        if (CUDA_CENTER_PIXEL())
        {
            DEBUG_PRINT(debugVisualization);
        }
        float depth = debugVisualization;
        float depthMin = 0.0f;
        float depthMax = 10000.0f;
        float normalizedDepth = (depth > depthMin) ? logf(depth - depthMin + 1.0f) / logf(depthMax - depthMin + 1.0f) : 0.0f;
        normalizedDepth = fminf(fmaxf(normalizedDepth, 0.0f), 1.0f);
        Store2DFloat4(Float4(normalizedDepth), outBuffer, pixelPos);
    }

    // Visualize normal
    if (0)
    {
        Float3 debugVisualization = Load2DFloat4(inBuffer, pixelPos).xyz;
        if (CUDA_CENTER_PIXEL())
        {
            DEBUG_PRINT(debugVisualization);
        }
        debugVisualization = (debugVisualization + 1.0f) / 2.0f;
        Store2DFloat4(Float4(debugVisualization, 0.0f), outBuffer, pixelPos);
    }
}