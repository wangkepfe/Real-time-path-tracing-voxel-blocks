#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"

namespace jazzfusion
{
    __device__ Float2 GetCoarseRoughnessWeightParams(float roughness)
    {
        return Float2(1.0, -roughness);
    }

    __device__ float GetNormalWeightParams(float nonLinearAccumSpeed, float fraction, float roughness = 1.0)
    {
        float angle = GetSpecularLobeHalfAngle(roughness);
        angle *= lerp(saturate(fraction), 1.0, nonLinearAccumSpeed);

        return 1.0 / max(angle, 0.5f * M_PI / 180.0f);
    }

    __device__ void Preload(
        Int2 sharedPos,
        Int2 globalPos,
        Int2 screenResolution,
        SurfObj normalRoughnessBuffer,
        SurfObj depthBuffer,
        SurfObj illuminationBuffer,
        Float4 *sharedNormalRoughness,
        Float3 *sharedHitdistViewZ,
        int bufferSize)
    {
        globalPos = clamp2i(globalPos, Int2(0), Int2(screenResolution.x - 1, screenResolution.y - 1));

        // It's ok that we don't use materialID in Hitdist reconstruction
        Float4 normalRoughness = Load2DFloat4(normalRoughnessBuffer, globalPos);
        float viewZ = abs(Load2DFloat1(depthBuffer, globalPos));

        // if (CUDA_CENTER_PIXEL())
        // {
        //     DEBUG_PRINT(viewZ);
        // }

        const float denoisingRange = 500000.0f;
        Float2 hitDist = Float2(denoisingRange);

        hitDist.y = Load2DFloat4(illuminationBuffer, globalPos).w;

        sharedNormalRoughness[sharedPos.y * bufferSize + sharedPos.x] = normalRoughness;
        sharedHitdistViewZ[sharedPos.y * bufferSize + sharedPos.x] = Float3(hitDist, viewZ);
    }

    template <int groupSize, int border>
    __global__ void HitDistReconstruction(
        Int2 screenResolution,
        Float2 invScreenResolution,
        SurfObj normalRoughnessBuffer,
        SurfObj depthBuffer,
        SurfObj illuminationBuffer,
        SurfObj illuminationPingBuffer)
    {
        Int2 threadPos;
        threadPos.x = threadIdx.x;
        threadPos.y = threadIdx.y;

        Int2 pixelPos;
        pixelPos.x = blockIdx.x * blockDim.x + threadIdx.x;
        pixelPos.y = blockIdx.y * blockDim.y + threadIdx.y;

        unsigned int threadIndex = threadIdx.z * (blockDim.x * blockDim.y) +
                                   threadIdx.y * blockDim.x +
                                   threadIdx.x;

        constexpr int bufferSize = (groupSize + border * 2);

        __shared__ Float4 sharedNormalRoughness[bufferSize * bufferSize];
        __shared__ Float3 sharedHitdistViewZ[bufferSize * bufferSize];

        Float2 pixelUv = Float2(Float2(pixelPos.x, pixelPos.y) + 0.5f) * invScreenResolution;

        Int2 groupBase = pixelPos - threadPos - border;
        unsigned int stageNum = (bufferSize * bufferSize + groupSize * groupSize - 1) / (groupSize * groupSize);
        for (unsigned int stage = 0; stage < stageNum; stage++)
        {
            unsigned int virtualIndex = threadIndex + stage * groupSize * groupSize;
            Int2 newId = Int2(virtualIndex % bufferSize, virtualIndex / bufferSize);
            if (stage == 0 || virtualIndex < bufferSize * bufferSize)
                Preload(newId, groupBase + newId, screenResolution, normalRoughnessBuffer, depthBuffer, illuminationBuffer, sharedNormalRoughness, sharedHitdistViewZ, bufferSize);
        }
        __syncthreads();

        if (pixelPos.x >= screenResolution.x || pixelPos.y >= screenResolution.y)
        {
            return;
        }

        Int2 smemPos = threadPos + border;
        Float3 centerHitdistViewZ = Float3(sharedHitdistViewZ[smemPos.y * bufferSize + smemPos.x]);
        float centerViewZ = centerHitdistViewZ.z;

        // Early out
        const float denoisingRange = 500000.0f;
        if (centerViewZ > denoisingRange)
            return;

        // Center data
        Float4 normalAndRoughness = Load2DFloat4(normalRoughnessBuffer, pixelPos);
        Float3 centerNormal = normalAndRoughness.xyz;
        float centerRoughness = normalAndRoughness.w;

        // Hit distance reconstruction
        Float3 centerDiffuseIllumination = Load2DFloat4(illuminationBuffer, pixelPos).xyz;
        float centerDiffuseHitDist = centerHitdistViewZ.y;
        float diffuseNormalWeightParam = GetNormalWeightParams(1.0f, 1.0f, 1.0f);

        float sumDiffuseWeight = 1000.0f * float(centerDiffuseHitDist != 0.0f);
        float sumDiffuseHitDist = centerDiffuseHitDist * sumDiffuseWeight;

        for (int dy = 0; dy <= border * 2; dy++)
        {
            for (int dx = 0; dx <= border * 2; dx++)
            {
                Int2 o = Int2(dx, dy) - border;

                if (o.x == 0 && o.y == 0)
                    continue;

                Int2 pos = threadPos + Int2(dx, dy);
                Float4 sampleNormalRoughness = sharedNormalRoughness[pos.y * bufferSize + pos.x];
                Float3 sampleNormal = sampleNormalRoughness.xyz;
                Float3 sampleRoughness = Float3(sampleNormalRoughness.w);
                Float3 sampleHitdistViewZ = sharedHitdistViewZ[pos.y * bufferSize + pos.x];
                float sampleViewZ = sampleHitdistViewZ.z;
                float cosa = saturate(dot(centerNormal, sampleNormal));
                float angle = AcosApprox(cosa);

                float w = IsInScreen(pixelUv + Float2(o.x, o.y) * invScreenResolution);
                w *= GetGaussianWeight(length(Float2(o.x, o.y)) * 0.5f);
                w *= GetBilateralWeight(sampleViewZ, centerViewZ);

                float sampleDiffuseHitDist = sampleHitdistViewZ.y;
                // if (CUDA_CENTER_PIXEL())
                // {
                //     DEBUG_PRINT(sampleDiffuseHitDist);
                // }
                float diffuseWeight = w;

                constexpr float expWeightScale = 3.0f;
                diffuseWeight *= ComputeExponentialWeight(angle, diffuseNormalWeightParam, 0.0f, expWeightScale);

                sampleDiffuseHitDist = Denanify(diffuseWeight, sampleDiffuseHitDist);
                diffuseWeight *= float(sampleDiffuseHitDist != 0.0f);

                sumDiffuseHitDist += sampleDiffuseHitDist * diffuseWeight;
                sumDiffuseWeight += diffuseWeight;
            }
        }

        sumDiffuseHitDist /= max(sumDiffuseWeight, 1e-6f);
        // if (CUDA_CENTER_PIXEL())
        // {
        //     DEBUG_PRINT(sumDiffuseHitDist);
        // }
        Store2DFloat4(Float4(centerDiffuseIllumination, sumDiffuseHitDist), illuminationPingBuffer, pixelPos);
    }
}