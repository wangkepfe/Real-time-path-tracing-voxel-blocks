#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"

__device__ float getDiffuseNormalWeight(Float3 centerNormal, Float3 pointNormal)
{
    float historyFixEdgeStoppingNormalPower = 8.0f;
    return pow(max(0.01f, dot(centerNormal, pointNormal)), max(historyFixEdgeStoppingNormalPower, 0.01f));
}

__device__ float getSpecularNormalWeight(Float3 centerNormal, Float3 pointNormal, float roughness)
{
    // Specular requires stricter normal weighting, especially for low roughness
    float historyFixEdgeStoppingNormalPower = lerp(32.0f, 8.0f, roughness);
    return pow(max(0.001f, dot(centerNormal, pointNormal)), max(historyFixEdgeStoppingNormalPower, 0.01f));
}

__device__ float getSpecularRoughnessWeight(float centerRoughness, float sampleRoughness)
{
    // Weight based on roughness similarity for specular
    float roughnessDiff = abs(centerRoughness - sampleRoughness);
    return exp(-16.0f * roughnessDiff);
}

__device__ float getRadius(float historyLength)
{
    constexpr float gHistoryFixFrameNum = 4.0f;
    // IMPORTANT: progression is "{8, 4, 2, 1} + 1". "+1" is important to better break blobs
    return exp2(gHistoryFixFrameNum - historyLength) + 1.0f;
}

// Enhanced HistoryFix supporting both diffuse and specular
__global__ void HistoryFix(
    Int2 screenResolution,
    Float2 invScreenResolution,

    SurfObj depthBuffer,
    SurfObj materialBuffer,
    SurfObj normalRoughnessBuffer,
    SurfObj historyLengthBuffer,

    // Separate diffuse buffers
    SurfObj diffuseIlluminationPingBuffer,
    SurfObj diffuseIlluminationPongBuffer,

    // Separate specular buffers
    SurfObj specularIlluminationPingBuffer,
    SurfObj specularIlluminationPongBuffer,
    SurfObj specularHitDistBuffer,

    Camera camera,

    // RELAX parameters
    float historyFixEdgeStoppingNormalPower,
    float historyFixStrideBetweenSamples)
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

    // Early out if linearZ is beyond denoising range
    float centerViewZ = Load2DFloat1(depthBuffer, pixelPos);
    const float denoisingRange = 500000.0f;
    if (centerViewZ > denoisingRange)
        return;

    float historyLength = Load2DFloat1(historyLengthBuffer, pixelPos);
    constexpr float historyFixFrameNum = 4.0f;

    // If no disocclusion detected, just copy Ping to Pong
    if (historyLength > historyFixFrameNum)
    {
        // Copy diffuse data from Ping to Pong
        Float4 diffuseData = Load2DFloat4(diffuseIlluminationPingBuffer, pixelPos);
        Store2DFloat4(diffuseData, diffuseIlluminationPongBuffer, pixelPos);

        // Copy specular data from Ping to Pong
        Float4 specularData = Load2DFloat4(specularIlluminationPingBuffer, pixelPos);
        Store2DFloat4(specularData, specularIlluminationPongBuffer, pixelPos);

        // Note: specularHitDistBuffer is shared between passes, no copy needed

        return;
    }

    // Loading center data
    float centerMaterialID = Load2DUshort1(materialBuffer, pixelPos);
    Float4 centerNormalRoughness = Load2DFloat4(normalRoughnessBuffer, pixelPos);
    Float3 centerNormal = centerNormalRoughness.xyz;
    float centerRoughness = centerNormalRoughness.w;
    Float3 centerWorldPos = GetCurrentWorldPosFromPixelPos(camera, pixelPos, centerViewZ);
    constexpr float depthThresholdConstant = 0.003f;
    float depthThreshold = depthThresholdConstant * centerViewZ;

    // Load both diffuse and specular data
    Float4 diffuseIlluminationAnd2ndMomentSum = Load2DFloat4(diffuseIlluminationPingBuffer, pixelPos);
    Float4 specularIlluminationAnd2ndMomentSum = Load2DFloat4(specularIlluminationPingBuffer, pixelPos);
    float specularHitDistSum = Load2DFloat1(specularHitDistBuffer, pixelPos);

    float diffuseWSum = 1.0f;
    float specularWSum = 1.0f;

    // Running sparse cross-bilateral filter
    float r = getRadius(historyLength) * historyFixStrideBetweenSamples;
    for (int j = -2; j <= 2; j++)
    {
        for (int i = -2; i <= 2; i++)
        {
            int dx = (int)(i * r);
            int dy = (int)(j * r);

            Int2 samplePosInt = pixelPos + Int2(dx, dy);

            bool isInside = samplePosInt.x >= 0 && samplePosInt.y >= 0 && samplePosInt.x < screenResolution.x && samplePosInt.y < screenResolution.y;
            if ((i == 0) && (j == 0))
                continue;

            float sampleMaterialID = Load2DUshort1(materialBuffer, samplePosInt);
            Float4 sampleNormalRoughness = Load2DFloat4(normalRoughnessBuffer, samplePosInt);
            Float3 sampleNormal = sampleNormalRoughness.xyz;
            float sampleRoughness = sampleNormalRoughness.w;
            float sampleViewZ = Load2DFloat1(depthBuffer, samplePosInt);
            Float3 sampleWorldPos = GetCurrentWorldPosFromPixelPos(camera, samplePosInt, sampleViewZ);

            float geometryWeight = GetPlaneDistanceWeight_Atrous(
                centerWorldPos,
                centerNormal,
                sampleWorldPos,
                depthThreshold);

            // Common weights
            float materialWeight = (sampleMaterialID == centerMaterialID) ? 1.0f : 0.0f;
            float insideWeight = isInside ? 1.0f : 0.0f;

            // Summing up diffuse result
            float diffuseW = geometryWeight;
            diffuseW *= getDiffuseNormalWeight(centerNormal, sampleNormal);
            diffuseW *= insideWeight * materialWeight;

            if (diffuseW > 1e-4f)
            {
                Float4 sampleDiffuseIlluminationAnd2ndMoment = Load2DFloat4(diffuseIlluminationPingBuffer, samplePosInt);
                diffuseIlluminationAnd2ndMomentSum += sampleDiffuseIlluminationAnd2ndMoment * diffuseW;
                diffuseWSum += diffuseW;
            }

            // Summing up specular result
            float specularW = geometryWeight;
            specularW *= getSpecularNormalWeight(centerNormal, sampleNormal, centerRoughness);
            specularW *= getSpecularRoughnessWeight(centerRoughness, sampleRoughness);
            specularW *= insideWeight * materialWeight;

            if (specularW > 1e-4f)
            {
                Float4 sampleSpecularIlluminationAnd2ndMoment = Load2DFloat4(specularIlluminationPingBuffer, samplePosInt);
                float sampleSpecularHitDist = Load2DFloat1(specularHitDistBuffer, samplePosInt);

                specularIlluminationAnd2ndMomentSum += sampleSpecularIlluminationAnd2ndMoment * specularW;
                specularHitDistSum += sampleSpecularHitDist * specularW;
                specularWSum += specularW;
            }
        }
    }

    // Output buffers will hold the pixels with disocclusion processed by history fix.
    // The next shader will have to copy these areas to normal and responsive history buffers.

    Float4 outDiffuseIlluminationAnd2ndMoment = diffuseIlluminationAnd2ndMomentSum / diffuseWSum;
    Float4 outSpecularIlluminationAnd2ndMoment = specularIlluminationAnd2ndMomentSum / specularWSum;
    float outSpecularHitDist = specularHitDistSum / specularWSum;

    Store2DFloat4(outDiffuseIlluminationAnd2ndMoment, diffuseIlluminationPongBuffer, pixelPos);
    Store2DFloat4(outSpecularIlluminationAnd2ndMoment, specularIlluminationPongBuffer, pixelPos);
    Store2DFloat1(outSpecularHitDist, specularHitDistBuffer, pixelPos);
}