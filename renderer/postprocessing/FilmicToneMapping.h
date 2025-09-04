#pragma once

#include "shaders/LinearMath.h"
#include "core/GlobalSettings.h"
#include "core/BufferManager.h"

#ifndef SurfObj
#define SurfObj cudaSurfaceObject_t
#endif

// ACES Filmic Tone Mapping (Narkowicz approximation)
__device__ inline Float3 ACESFilm(Float3 x)
{
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp3f(x * (a * x + b) / (x * (c * x + d) + e));
}

// Uncharted 2 Tone Mapping
__device__ inline Float3 Uncharted2Tonemap(Float3 x)
{
    float A = 0.15f;
    float B = 0.50f;
    float C = 0.10f;
    float D = 0.20f;
    float E = 0.02f;
    float F = 0.30f;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

// Simple Reinhard Tone Mapping
__device__ inline Float3 ReinhardTonemap(Float3 x, float whitePoint)
{
    Float3 numerator = x * (1.0f + (x / (whitePoint * whitePoint)));
    return numerator / (1.0f + x);
}

// Helper function for power
__device__ inline Float3 powf3(Float3 v, float e)
{
    return Float3(powf(v.x, e), powf(v.y, e), powf(v.z, e));
}

// sRGB gamma encoding
__device__ inline Float3 LinearToSRGB(Float3 color)
{
    Float3 result;
    result.x = (color.x <= 0.0031308f) ? 12.92f * color.x : 1.055f * powf(color.x, 1.0f / 2.4f) - 0.055f;
    result.y = (color.y <= 0.0031308f) ? 12.92f * color.y : 1.055f * powf(color.y, 1.0f / 2.4f) - 0.055f;
    result.z = (color.z <= 0.0031308f) ? 12.92f * color.z : 1.055f * powf(color.z, 1.0f / 2.4f) - 0.055f;
    return result;
}

// Main filmic tone mapping kernel
__global__ void FilmicToneMapping(
    SurfObj colorBuffer,
    Int2 size,
    PostProcessingPipelineParams pipelineParams,
    ToneMappingParams toneMappingParams,
    float computedExposure)
{
    Int2 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx.x >= size.x || idx.y >= size.y)
        return;

    Float3 color = Load2DFloat4(colorBuffer, idx).xyz;
    
    // Apply exposure (either auto-computed or manual)
    float finalExposure = pipelineParams.enableAutoExposure ? computedExposure : toneMappingParams.manualExposure;
    color *= finalExposure;
    
    // Apply tone mapping curve based on selection
    Float3 toneMapped;
    switch (toneMappingParams.curve)
    {
    case ToneMappingParams::CURVE_NARKOWICZ_ACES:
        toneMapped = ACESFilm(color);
        break;
    case ToneMappingParams::CURVE_UNCHARTED2:
        {
            float W = toneMappingParams.whitePoint;
            Float3 whiteScale = Float3(1.0f) / Uncharted2Tonemap(Float3(W));
            toneMapped = Uncharted2Tonemap(color * 2.0f) * whiteScale;
        }
        break;
    case ToneMappingParams::CURVE_REINHARD:
        toneMapped = ReinhardTonemap(color, toneMappingParams.whitePoint);
        break;
    default:
        toneMapped = ACESFilm(color);
        break;
    }
    
    // Apply color grading
    toneMapped = clamp3f(toneMapped);
    
    // Contrast adjustment
    toneMapped = powf3(toneMapped, toneMappingParams.contrast);
    
    // Saturation adjustment
    float luminance = dot(toneMapped, Float3(0.2126f, 0.7152f, 0.0722f));
    toneMapped = lerp(Float3(luminance), toneMapped, toneMappingParams.saturation);
    
    // Lift and gain
    toneMapped = clamp3f(toneMapped * toneMappingParams.gain + toneMappingParams.lift);
    
    // Always output sRGB (no HDR output modes)
    toneMapped = LinearToSRGB(toneMapped);
    
    Store2DFloat4(Float4(toneMapped, 1.0f), colorBuffer, idx);
}