#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"

namespace jazzfusion
{
    __global__ void Atrous(
        Int2 screenResolution,
        Float2 invScreenResolution,

        SurfObj illuminationInputBuffer,

        SurfObj normalRoughnessBuffer,
        SurfObj materialBuffer,
        SurfObj depthBuffer,
        SurfObj historyLengthBuffer,

        SurfObj illuminationOutputBuffer,

        Camera camera,
        unsigned int frameIndex,
        unsigned int stepSize)
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

        float centerViewZ = Load2DFloat1(depthBuffer, pixelPos);

        // Early out
        const float denoisingRange = 500000.0f;
        if (centerViewZ > denoisingRange)
            return;

        float centerMaterialID = Load2DUshort1(materialBuffer, pixelPos);
        Float4 centerNormalRoughness = Load2DFloat4(normalRoughnessBuffer, pixelPos);
        Float3 centerNormal = centerNormalRoughness.xyz;
        float centerRoughness = centerNormalRoughness.w;
        Float3 centerWorldPos = GetCurrentWorldPosFromPixelPos(camera, pixelPos, centerViewZ);

        float historyLength = Load2DFloat1(historyLengthBuffer, pixelPos);

        // Diffuse normal weight is used for diffuse and can be used for specular depending on settings.
        // Weight strictness is higher as the Atrous step size increases.
        float lobeAngleFraction = 0.5f;
        float diffuseLobeAngleFraction = lobeAngleFraction / sqrtf((float)stepSize);

        diffuseLobeAngleFraction = lerp(0.99f, diffuseLobeAngleFraction, saturate(historyLength / 5.0f));

        Float4 centerDiffuseIlluminationAndVariance = Load2DFloat4(illuminationInputBuffer, pixelPos);
        float centerDiffuseLuminance = luminance(centerDiffuseIlluminationAndVariance.xyz);
        float centerDiffuseVar = centerDiffuseIlluminationAndVariance.w;
        float diffusePhiLuminance = 2.0f;
        float diffusePhiLIlluminationInv = 1.0f / max(1.0e-4f, diffusePhiLuminance * sqrt(centerDiffuseVar));

        float diffuseLuminanceWeightRelaxation = 1.0f;
        float diffuseNormalWeightParam = GetNormalWeightParam2(1.0f, diffuseLobeAngleFraction);

        float sumWDiffuse = 0.44198f * 0.44198f;
        Float4 sumDiffuseIlluminationAndVariance = centerDiffuseIlluminationAndVariance * Float4(Float3(sumWDiffuse), sumWDiffuse * sumWDiffuse);

        const float kernelWeightGaussian3x3[2] = {0.44198f, 0.27901f};

        constexpr float depthThresholdConstant = 0.003f;
        float depthThreshold = depthThresholdConstant * centerViewZ;

        // Adding random offsets to minimize "ringing" at large A-Trous steps
        unsigned int rngHashState;
        Int2 offset = Int2(0);
        if (stepSize > 4)
        {
            RngHashInitialize(rngHashState, UInt2(pixelPos.x, pixelPos.y), frameIndex);
            Float2 offsetF = Float2(stepSize) * 0.5f * (RngHashGetFloat2(rngHashState) - 0.5f);
            offset = Int2(offsetF.x, offsetF.y);
        }

        float diffuseMinLuminanceWeight = 0.0f;
        float maxDiffuseLuminanceRelativeDifference = -log(saturate(diffuseMinLuminanceWeight));

        // if (CUDA_PIXEL(0.5f, 0.5f))
        // {
        //     DEBUG_PRINT(centerDiffuseIlluminationAndVariance);
        // }

        for (int yy = -1; yy <= 1; yy++)
        {
            for (int xx = -1; xx <= 1; xx++)
            {
                int sampleIndexDebug = (yy + 1) * 3 + (xx + 1);

                Int2 p = pixelPos + offset + Int2(xx, yy) * stepSize;
                bool isCenter = ((xx == 0) && (yy == 0));
                if (isCenter)
                    continue;

                bool isInside = p.x >= 0 && p.y >= 0 && p.x < screenResolution.x && p.y < screenResolution.y;
                float kernel = kernelWeightGaussian3x3[abs(xx)] * kernelWeightGaussian3x3[abs(yy)];

                // Fetching normal, roughness, linear Z
                // Calculating sample world position
                float sampleMaterialID = Load2DUshort1(materialBuffer, p);
                Float4 sampleNormalRoughnes = Load2DFloat4(normalRoughnessBuffer, p);
                Float3 sampleNormal = sampleNormalRoughnes.xyz;
                float sampleRoughness = sampleNormalRoughnes.w;
                float sampleViewZ = Load2DFloat1(depthBuffer, p);
                Float3 sampleWorldPos = GetCurrentWorldPosFromPixelPos(camera, p, sampleViewZ);

                // Calculating geometry weight for diffuse and specular
                float geometryW = GetPlaneDistanceWeight_Atrous(centerWorldPos, centerNormal, sampleWorldPos, depthThreshold);
                geometryW *= kernel;
                geometryW *= float(isInside && sampleViewZ < denoisingRange);

                // Calculating weights for diffuse
                float angled = AcosApprox(dot(centerNormal, sampleNormal));
                float normalWDiffuse = ComputeNonExponentialWeight(angled, diffuseNormalWeightParam, 0.0f);

                // Summing up diffuse
                float weight = geometryW * normalWDiffuse;
                weight *= (sampleMaterialID == centerMaterialID);
                if (weight > 1e-4f)
                {
                    Float4 sampleDiffuseIlluminationAndVariance = Load2DFloat4(illuminationInputBuffer, p);
                    float sampleDiffuseLuminance = luminance(sampleDiffuseIlluminationAndVariance.xyz);

                    float diffuseLuminanceW = abs(centerDiffuseLuminance - sampleDiffuseLuminance) * diffusePhiLIlluminationInv;
                    diffuseLuminanceW = min(maxDiffuseLuminanceRelativeDifference, diffuseLuminanceW);
                    weight *= exp(-diffuseLuminanceW);

                    sumWDiffuse += weight;
                    sumDiffuseIlluminationAndVariance += Float4(Float3(weight), weight * weight) * sampleDiffuseIlluminationAndVariance;

                    // if (CUDA_PIXEL(0.5f, 0.5f))
                    // {
                    //     DEBUG_PRINT(Float4(weight, geometryW, normalWDiffuse, sampleIndexDebug));
                    //     DEBUG_PRINT(Float4(sampleDiffuseIlluminationAndVariance.xyz, sampleIndexDebug));
                    // }
                }
            }
        }
        Float4 filteredDiffuseIlluminationAndVariance = Float4(sumDiffuseIlluminationAndVariance / Float4(Float3(sumWDiffuse), sumWDiffuse * sumWDiffuse));

        // if (CUDA_PIXEL(0.5f, 0.5f))
        // {
        //     Store2DFloat4(Float4(1.0f, 0.0f, 0.0f, 0.0f), illuminationOutputBuffer, pixelPos);
        // }
        // else
        // {
        Store2DFloat4(filteredDiffuseIlluminationAndVariance, illuminationOutputBuffer, pixelPos);
        // }
    }
}