#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"
#include "shaders/RestirCommon.h"
#include <float.h>
#include <cuda_runtime.h>

__global__ void FireflyBoilingFilter(
    Int2 screenResolution,
    SurfObj illuminationBuffer,
    SurfObj normalRoughnessBuffer,
    SurfObj depthBuffer,
    SurfObj materialBuffer,
    DIReservoir *reservoirBuffer,
    int reservoirStride,
    int reservoirParity,
    float weightThreshold,
    float minWeight,
    float normalThreshold,
    float depthSigma,
    float phiLuminance,
    Camera camera)
{
    Int2 pixelPos;
    pixelPos.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelPos.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelPos.x >= screenResolution.x || pixelPos.y >= screenResolution.y)
    {
        return;
    }

    constexpr float kDenoisingRange = 500000.0f;

    const float centerDepth = Load2DFloat1(depthBuffer, pixelPos);
    if (centerDepth > kDenoisingRange)
    {
        // Sky pixel, skip any processing.
        return;
    }

    const int pixelIndex = pixelPos.y * screenResolution.x + pixelPos.x;
    const int reservoirIndex = reservoirParity * reservoirStride + pixelIndex;

    DIReservoir reservoir = reservoirBuffer[reservoirIndex];
    const float currentWeight = reservoir.weightSum;
    const bool reservoirValid = (reservoir.lightData != 0) && isfinite(currentWeight) && currentWeight > 0.0f;

    // Compute warp-wide statistics over the 8x4 tile (single warp per block).
    const unsigned int activeMask = __activemask();

    float warpWeightSum = reservoirValid ? currentWeight : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        warpWeightSum += __shfl_down_sync(activeMask, warpWeightSum, offset);
    }
    warpWeightSum = __shfl_sync(activeMask, warpWeightSum, 0);

    unsigned int warpValidCount = reservoirValid ? 1u : 0u;
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        warpValidCount += __shfl_down_sync(activeMask, warpValidCount, offset);
    }
    warpValidCount = __shfl_sync(activeMask, warpValidCount, 0);

    if (!reservoirValid)
    {
        return;
    }

    const float neighborWeightSum = warpWeightSum - currentWeight;
    const int neighborValidCount = static_cast<int>(warpValidCount) - 1;

    bool isFirefly = false;
    if (currentWeight >= minWeight)
    {
        if (neighborValidCount <= 0)
        {
            isFirefly = true;
        }
        else
        {
            const float neighborAverage = neighborWeightSum / float(neighborValidCount);
            if (neighborAverage > 0.0f && currentWeight > neighborAverage * weightThreshold)
            {
                isFirefly = true;
            }
        }
    }

    if (!isFirefly)
    {
        return;
    }

    const Float4 centerColor4 = Load2DFloat4(illuminationBuffer, pixelPos);
    const Float3 centerColor = centerColor4.xyz;
    const float centerLuminance = luminance(centerColor);

    Float4 centerNormalRoughness = Load2DFloat4(normalRoughnessBuffer, pixelPos);
    Float3 centerNormal = centerNormalRoughness.xyz;
    const float centerNormalLen = length(centerNormal);
    if (centerNormalLen > 0.0f)
    {
        centerNormal /= centerNormalLen;
    }
    else
    {
        centerNormal = Float3(0.0f, 1.0f, 0.0f);
    }

    const float centerMaterial = Load2DFloat1(materialBuffer, pixelPos);
    const Float3 centerWorldPos = GetCurrentWorldPosFromPixelPos(camera, pixelPos, centerDepth);

    const float gaussian[3] = {1.0f, 2.0f, 1.0f};

    Float4 filteredColor = centerColor4;
    float filteredWeight = 1.0f;

    Float4 fallbackColor = centerColor4 * (gaussian[0] * gaussian[0]);
    float fallbackWeight = gaussian[0] * gaussian[0];

    const float depthScale = fmaxf(fabsf(centerDepth), 1.0f);
    const float normalWeightParam = GetNormalWeightParam2(1.0f, 0.25f);

    DIReservoir bestReservoir = reservoir;
    float bestScore = FLT_MAX;
    bool hasReplacement = false;

    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0)
            {
                continue;
            }

            const int sampleX = pixelPos.x + dx;
            const int sampleY = pixelPos.y + dy;

            if (sampleX < 0 || sampleY < 0 || sampleX >= screenResolution.x || sampleY >= screenResolution.y)
            {
                continue;
            }

            const float gaussianWeight = gaussian[abs(dx)] * gaussian[abs(dy)];
            const Int2 samplePos(sampleX, sampleY);

            const Float4 sampleColor4 = Load2DFloat4(illuminationBuffer, samplePos);
            fallbackColor += sampleColor4 * gaussianWeight;
            fallbackWeight += gaussianWeight;

            const float sampleDepth = Load2DFloat1(depthBuffer, samplePos);
            if (sampleDepth > kDenoisingRange)
            {
                continue;
            }

            Float4 sampleNormalRoughness = Load2DFloat4(normalRoughnessBuffer, samplePos);
            Float3 sampleNormal = sampleNormalRoughness.xyz;
            const float sampleNormalLen = length(sampleNormal);
            if (sampleNormalLen <= 0.0f)
            {
                continue;
            }
            sampleNormal /= sampleNormalLen;

            const float normalDot = dot(centerNormal, sampleNormal);
            if (normalDot < normalThreshold)
            {
                continue;
            }

            const float sampleMaterial = Load2DFloat1(materialBuffer, samplePos);
            if (fabsf(sampleMaterial - centerMaterial) > 0.5f)
            {
                continue;
            }

            const Float3 sampleWorldPos = GetCurrentWorldPosFromPixelPos(camera, samplePos, sampleDepth);

            const float geometryWeight = GetPlaneDistanceWeight_Atrous(centerWorldPos, centerNormal, sampleWorldPos, depthSigma * depthScale);
            if (geometryWeight <= 0.0f)
            {
                continue;
            }

            const float normalWeight = ComputeNonExponentialWeight(AcosApprox(clampf(normalDot, -1.0f, 1.0f)), normalWeightParam, 0.0f);
            const float depthWeight = expf(-fabsf(sampleDepth - centerDepth) / (depthScale * depthSigma + 1e-6f));
            const float luminanceWeight = expf(-fabsf(luminance(sampleColor4.xyz) - centerLuminance) * phiLuminance);

            const float totalWeight = gaussianWeight * geometryWeight * normalWeight * depthWeight * luminanceWeight;
            if (totalWeight > 1e-5f)
            {
                filteredColor += sampleColor4 * totalWeight;
                filteredWeight += totalWeight;
            }

            const int neighborIndex = reservoirParity * reservoirStride + sampleY * screenResolution.x + sampleX;
            const DIReservoir neighborReservoir = reservoirBuffer[neighborIndex];
            const bool neighborValid = (neighborReservoir.lightData != 0) && isfinite(neighborReservoir.weightSum) && neighborReservoir.weightSum > 0.0f && neighborReservoir.weightSum < currentWeight;

            if (neighborValid)
            {
                const float depthTerm = fabsf(sampleDepth - centerDepth) / (depthScale + 1e-6f);
                const float normalTerm = 1.0f - clampf(normalDot, 0.0f, 1.0f);
                const float weightDiff = fabsf(neighborReservoir.weightSum - currentWeight);
                const float score = depthTerm + normalTerm + 0.25f * weightDiff;

                if (score < bestScore)
                {
                    bestScore = score;
                    bestReservoir = neighborReservoir;
                    hasReplacement = true;
                }
            }
        }
    }

    Float4 outputColor;
    if (filteredWeight > 0.0f)
    {
        outputColor = filteredColor / filteredWeight;
    }
    else if (fallbackWeight > 0.0f)
    {
        outputColor = fallbackColor / fallbackWeight;
    }
    else
    {
        outputColor = centerColor4;
    }

    Store2DFloat4(outputColor, illuminationBuffer, pixelPos);

    if (hasReplacement)
    {
        reservoirBuffer[reservoirIndex] = bestReservoir;
    }
    else
    {
        DIReservoir clamped = reservoir;
        float neighborAverage = (neighborValidCount > 0) ? (neighborWeightSum / float(neighborValidCount)) : minWeight;
        float clampTarget = (neighborValidCount > 0) ? (neighborAverage * weightThreshold) : minWeight;
        clampTarget = fmaxf(clampTarget, minWeight);
        clamped.weightSum = fminf(clamped.weightSum, clampTarget);
        reservoirBuffer[reservoirIndex] = clamped;
    }
}





