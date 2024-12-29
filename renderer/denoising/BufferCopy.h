#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"

namespace jazzfusion
{
    __global__ void BufferCopyFloat1(
        Int2 screenResolution,
        SurfObj inBuffer,
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

        float val = Load2DFloat1(inBuffer, pixelPos);
        Store2DFloat1(val, outBuffer, pixelPos);
    }

    __global__ void BufferCopyFloat4(
        Int2 screenResolution,
        SurfObj inBuffer,
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

        Float4 val = Load2DFloat4(inBuffer, pixelPos);
        Store2DFloat4(val, outBuffer, pixelPos);
    }

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

        Float3 illum = Load2DFloat4(inBuffer, pixelPos).xyz;
        // Float3 albedo = Float3(1.0f);
        Float3 albedo = Load2DFloat4(albedoBuffer, pixelPos).xyz;
        Float3 uiOverlay = Load2DFloat4(uiBuffer, pixelPos).xyz;
        Float3 uiColor = Float3(0.0f);

        Float3 out = uiOverlay.x > 0.0f ? uiColor : (illum * albedo);

        Store2DFloat4(Float4(out, 0.0f), outBuffer, pixelPos);
    }
}