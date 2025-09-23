#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"

// Anti-firefly data structure for bilateral filtering
struct AntiFireflyData
{
    Float3 normal;
    float roughness;
    float materialID;
    float viewZ;
};

// Cross Bilateral Rank-Conditioned Rank-Selection (RCRS) filter implementation
// Based on NRD RELAX anti-firefly pass for removing temporal fireflies
__device__ Float3 RELAX_AntiFirefly(
    Int2 pixelPos,
    Float3 centerColor,
    AntiFireflyData centerData,
    SurfObj illuminationBuffer,
    SurfObj normalRoughnessBuffer,
    SurfObj materialBuffer,
    SurfObj depthBuffer,
    Int2 screenResolution,
    Camera camera,
    unsigned int maxSamples = 15)
{
    // Early out for zero radiance
    if (dot(centerColor, Float3(1.0f)) == 0.0f)
        return centerColor;

    const Float2 poissonDisk[16] = {
        Float2(-0.94201624f, -0.39906216f),
        Float2( 0.94558609f, -0.76890725f),
        Float2(-0.09418410f, -0.92938870f),
        Float2( 0.34495938f,  0.29387760f),
        Float2(-0.91588581f,  0.45771432f),
        Float2(-0.81544232f, -0.87912464f),
        Float2(-0.38277543f,  0.27676845f),
        Float2( 0.97484398f,  0.75648379f),
        Float2( 0.44323325f, -0.97511554f),
        Float2( 0.53742981f, -0.47373420f),
        Float2(-0.26496911f, -0.41893023f),
        Float2( 0.79197514f,  0.19090188f),
        Float2(-0.24188840f,  0.99706507f),
        Float2(-0.81409955f,  0.91437590f),
        Float2( 0.19984126f,  0.78641367f),
        Float2( 0.14383161f, -0.14100790f)
    };

    // Bilateral filter parameters
    float bilateralSigmaDepth = 0.1f;
    float bilateralSigmaNormal = 0.25f;
    float bilateralSigmaLuma = 0.25f;
    float filterRadius = 4.0f;

    float centerLuminance = luminance(centerColor);
    Float3 centerWorldPos = GetCurrentWorldPosFromPixelPos(camera, pixelPos, centerData.viewZ);

    // Collect samples for bilateral filtering
    Float3 samples[16];
    float weights[16];
    int validSampleCount = 0;

    // Add center sample
    samples[0] = centerColor;
    weights[0] = 1.0f;
    validSampleCount = 1;

    // Sample neighborhood using Poisson disk
    int numSamples = min((int)maxSamples, 15);
    for (int i = 0; i < numSamples && validSampleCount < 16; ++i)
    {
        Float2 offset = poissonDisk[i] * filterRadius;
        Int2 samplePos = pixelPos + Int2(offset.x, offset.y);

        // Check bounds
        if (samplePos.x < 0 || samplePos.y < 0 || 
            samplePos.x >= screenResolution.x || samplePos.y >= screenResolution.y)
            continue;

        // Load sample data
        float sampleViewZ = Load2DFloat1(depthBuffer, samplePos);
        float sampleMaterialID = Load2DFloat1(materialBuffer, samplePos);
        Float4 sampleNormalRoughness = Load2DFloat4(normalRoughnessBuffer, samplePos);
        Float3 sampleNormal = sampleNormalRoughness.xyz;
        float sampleRoughness = sampleNormalRoughness.w;

        // Early rejection for different materials
        if (abs(sampleMaterialID - centerData.materialID) > 0.1f)
            continue;

        Float3 sampleWorldPos = GetCurrentWorldPosFromPixelPos(camera, samplePos, sampleViewZ);
        
        // Calculate bilateral weights
        
        // Geometry/depth weight
        float depthDiff = abs(sampleViewZ - centerData.viewZ);
        float depthWeight = exp(-depthDiff / (bilateralSigmaDepth * centerData.viewZ));
        
        // Normal weight  
        float normalSimilarity = saturate(dot(centerData.normal, sampleNormal));
        float normalWeight = pow(normalSimilarity, 1.0f / bilateralSigmaNormal);
        
        // Plane distance weight (additional geometry constraint)
        float planeDistance = abs(dot(sampleWorldPos - centerWorldPos, centerData.normal));
        float planeWeight = exp(-planeDistance / (filterRadius * 0.1f));

        // Roughness weight
        float roughnessDiff = abs(sampleRoughness - centerData.roughness);
        float roughnessWeight = exp(-roughnessDiff * 8.0f);

        // Combine weights
        float totalWeight = depthWeight * normalWeight * planeWeight * roughnessWeight;
        
        if (totalWeight > 0.01f)
        {
            // Load actual sample color from illumination buffer
            Float4 sampleIllumination = Load2DFloat4(illuminationBuffer, samplePos);
            samples[validSampleCount] = sampleIllumination.xyz;
            weights[validSampleCount] = totalWeight;
            validSampleCount++;
        }
    }

    // For RCRS, we need the actual sample colors - in practice, this would sample from 
    // the illumination buffer. For now, we'll implement a luminance-based bilateral filter
    
    // Calculate luminance-based weights and perform filtering
    Float3 filteredColor = Float3(0.0f);
    float totalWeight = 0.0f;

    for (int i = 0; i < validSampleCount; ++i)
    {
        float sampleLuminance = luminance(samples[i]);
        
        // Luminance weight
        float lumaDiff = abs(sampleLuminance - centerLuminance);
        float lumaWeight = exp(-lumaDiff / (bilateralSigmaLuma * (centerLuminance + 1e-3f)));
        
        float finalWeight = weights[i] * lumaWeight;
        
        filteredColor += samples[i] * finalWeight;
        totalWeight += finalWeight;
    }

    // Normalize and apply
    if (totalWeight > 1e-6f)
    {
        filteredColor /= totalWeight;
        
        // Preserve energy - don't let the filter significantly darken the image
        float energyRatio = centerLuminance / max(luminance(filteredColor), 1e-6f);
        if (energyRatio > 1.2f || energyRatio < 0.8f)
        {
            // Blend with original if energy preservation is violated
            float preservationBlend = saturate((abs(energyRatio - 1.0f) - 0.2f) / 0.6f);
            filteredColor = lerp(filteredColor, centerColor, preservationBlend);
        }
    }
    else
    {
        filteredColor = centerColor;
    }

    return filteredColor;
}

// Main anti-firefly kernel
__global__ void AntiFirefly(
    Int2 screenResolution,
    Float2 invScreenResolution,

    SurfObj diffuseIlluminationInputBuffer,
    SurfObj specularIlluminationInputBuffer,

    SurfObj normalRoughnessBuffer,
    SurfObj materialBuffer, 
    SurfObj depthBuffer,

    SurfObj diffuseIlluminationOutputBuffer,
    SurfObj specularIlluminationOutputBuffer,

    Camera camera,
    
    // Anti-firefly parameters
    float denoisingRange,
    unsigned int maxSamples)
{
    Int2 pixelPos;
    pixelPos.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelPos.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelPos.x >= screenResolution.x || pixelPos.y >= screenResolution.y)
        return;

    float centerViewZ = Load2DFloat1(depthBuffer, pixelPos);

    // Early out - beyond denoising range
    if (centerViewZ > denoisingRange)
        return;

    // Early out - sky pixels
    if (centerViewZ == 0.0f)
        return;

    // Load center pixel data
    float centerMaterialID = Load2DFloat1(materialBuffer, pixelPos);
    Float4 centerNormalRoughness = Load2DFloat4(normalRoughnessBuffer, pixelPos);
    Float3 centerNormal = centerNormalRoughness.xyz;
    float centerRoughness = centerNormalRoughness.w;

    // Setup anti-firefly data
    AntiFireflyData antiFireflyData;
    antiFireflyData.normal = centerNormal;
    antiFireflyData.roughness = centerRoughness;
    antiFireflyData.materialID = centerMaterialID;
    antiFireflyData.viewZ = centerViewZ;

    // Load input illumination
    Float4 diffuseIllumination = Load2DFloat4(diffuseIlluminationInputBuffer, pixelPos);
    Float4 specularIllumination = Load2DFloat4(specularIlluminationInputBuffer, pixelPos);

    // Apply anti-firefly filter to both diffuse and specular
    Float3 filteredDiffuse = RELAX_AntiFirefly(
        pixelPos, 
        diffuseIllumination.xyz, 
        antiFireflyData,
        diffuseIlluminationInputBuffer,
        normalRoughnessBuffer,
        materialBuffer,
        depthBuffer,
        screenResolution,
        camera,
        maxSamples
    );

    Float3 filteredSpecular = RELAX_AntiFirefly(
        pixelPos, 
        specularIllumination.xyz, 
        antiFireflyData,
        specularIlluminationInputBuffer,
        normalRoughnessBuffer,
        materialBuffer,
        depthBuffer,
        screenResolution,
        camera,
        maxSamples
    );

    // Store results (preserve variance in .w channel)
    Store2DFloat4(Float4(filteredDiffuse, diffuseIllumination.w), diffuseIlluminationOutputBuffer, pixelPos);
    Store2DFloat4(Float4(filteredSpecular, specularIllumination.w), specularIlluminationOutputBuffer, pixelPos);
}