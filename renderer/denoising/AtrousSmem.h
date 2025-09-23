#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"

// Helper functions
// computes a 3x3 gaussian blur of the variance, centered around
// the current pixel
__device__ void computeVariance(
    Int2 threadPos,
    float &diffuseVariance,
    float &specularVariance,
    int border,
    Float4 *sharedDiffuseIllum,
    Float4 *sharedSpecularIllum,
    int bufferSize)
{
    Float4 diffuseSum = Float4(0.0f);
    Float4 specularSum = Float4(0.0f);

    const float kernel[4] =
        {1.0f / 4.0f, 1.0f / 8.0f,
         1.0f / 8.0f, 1.0f / 16.0f};

    Int2 sharedMemoryIndex = threadPos + Int2(border, border);
    for (int dx = -1; dx <= 1; dx++)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            Int2 sharedMemoryIndexP = sharedMemoryIndex + Int2(dx, dy);
            float k = kernel[abs(dx) * 2 + abs(dy)];

            Float4 diffuse = sharedDiffuseIllum[sharedMemoryIndexP.y * bufferSize + sharedMemoryIndexP.x];
            Float4 specular = sharedSpecularIllum[sharedMemoryIndexP.y * bufferSize + sharedMemoryIndexP.x];
            diffuseSum += diffuse * k;
            specularSum += specular * k;
        }
    }

    float diffuse1stMoment = luminance(diffuseSum.xyz);
    float diffuse2ndMoment = diffuseSum.w;
    diffuseVariance = max(0.0f, diffuse2ndMoment - diffuse1stMoment * diffuse1stMoment);
    
    float specular1stMoment = luminance(specularSum.xyz);
    float specular2ndMoment = specularSum.w;
    specularVariance = max(0.0f, specular2ndMoment - specular1stMoment * specular1stMoment);
}

__device__ void Preload(Int2 sharedPos,
                        Int2 globalPos,
                        Int2 screenResolution,
                        SurfObj diffuseIlluminationInputBuffer,
                        SurfObj specularIlluminationInputBuffer,
                        SurfObj normalRoughnessBuffer,
                        SurfObj materialBuffer,
                        SurfObj depthBuffer,
                        Float4 *sharedDiffuseIllum,
                        Float4 *sharedSpecularIllum,
                        Float4 *sharedNormalRoughness,
                        Float4 *sharedWorldPosMaterialID,
                        int bufferSize,
                        Camera camera)
{
    globalPos = clamp2i(globalPos, Int2(0), Int2(screenResolution.x - 1, screenResolution.y - 1));

    sharedDiffuseIllum[sharedPos.y * bufferSize + sharedPos.x] = Load2DFloat4(diffuseIlluminationInputBuffer, globalPos);
    sharedSpecularIllum[sharedPos.y * bufferSize + sharedPos.x] = Load2DFloat4(specularIlluminationInputBuffer, globalPos);
    sharedNormalRoughness[sharedPos.y * bufferSize + sharedPos.x] = Load2DFloat4(normalRoughnessBuffer, globalPos);

    float viewZ = Load2DFloat1(depthBuffer, globalPos);
    float materialID = Load2DFloat1(materialBuffer, globalPos);
    Float3 worldPos = GetCurrentWorldPosFromPixelPos(camera, globalPos, viewZ);
    sharedWorldPosMaterialID[sharedPos.y * bufferSize + sharedPos.x] = Float4(worldPos, materialID);
}

// Note: Using functions from DenoiserCommon.h to avoid duplicates

template <int groupSize, int border>
__global__ void AtrousSmem(
    Int2 screenResolution,
    Float2 invScreenResolution,

    SurfObj diffuseIlluminationInputBuffer,
    SurfObj specularIlluminationInputBuffer,

    SurfObj normalRoughnessBuffer,
    SurfObj materialBuffer,
    SurfObj depthBuffer,
    SurfObj historyLengthBuffer,

    SurfObj diffuseIlluminationOutputBuffer,
    SurfObj specularIlluminationOutputBuffer,

    Camera camera,

    // Denoising parameters
    float phiLuminance,
    float specularPhiLuminance,
    float depthThreshold,
    float roughnessFraction,
    float diffuseLobeAngleFraction,
    float specularLobeAngleFraction)
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

    __shared__ Float4 sharedDiffuseIllum[bufferSize * bufferSize];
    __shared__ Float4 sharedSpecularIllum[bufferSize * bufferSize];
    __shared__ Float4 sharedNormalRoughness[bufferSize * bufferSize];
    __shared__ Float4 sharedWorldPosMaterialID[bufferSize * bufferSize];

    Int2 groupBase = pixelPos - threadPos - border;
    unsigned int stageNum = (bufferSize * bufferSize + groupSize * groupSize - 1) / (groupSize * groupSize);
    for (unsigned int stage = 0.0f; stage < stageNum; stage++)
    {
        unsigned int virtualIndex = threadIndex + stage * groupSize * groupSize;
        Int2 newId = Int2(virtualIndex % bufferSize, virtualIndex / bufferSize);
        if (stage == 0 || virtualIndex < bufferSize * bufferSize)
            Preload(
                newId,
                groupBase + newId,
                screenResolution,
                diffuseIlluminationInputBuffer,
                specularIlluminationInputBuffer,
                normalRoughnessBuffer,
                materialBuffer,
                depthBuffer,
                sharedDiffuseIllum,
                sharedSpecularIllum,
                sharedNormalRoughness,
                sharedWorldPosMaterialID,
                bufferSize,
                camera);
    }
    __syncthreads();

    if (pixelPos.x >= screenResolution.x || pixelPos.y >= screenResolution.y)
    {
        return;
    }

    float centerViewZ = Load2DFloat1(depthBuffer, pixelPos);

    // Early out - using parameter
    if (centerViewZ > 500000.0f) // TODO: Pass denoisingRange parameter
        return;

    Int2 sharedMemoryIndex = threadPos + Int2(border, border);
    Float4 normalRoughness = sharedNormalRoughness[sharedMemoryIndex.y * bufferSize + sharedMemoryIndex.x];
    Float3 centerNormal = normalRoughness.xyz;
    float centerRoughness = normalRoughness.w;
    Float4 centerWorldPosMaterialID = sharedWorldPosMaterialID[sharedMemoryIndex.y * bufferSize + sharedMemoryIndex.x];
    Float3 centerWorldPos = centerWorldPosMaterialID.xyz;
    float centerMaterialID = centerWorldPosMaterialID.w;

    float historyLength = Load2DFloat1(historyLengthBuffer, pixelPos);

    uint32_t spatialVarianceEstimationHistoryThreshold = 3; // TODO: Make parameter
    float diffusePhiLuminance = phiLuminance;
    float specularPhiLuminanceAdapted = getSpecularLuminanceWeightFromRoughness(centerRoughness, specularPhiLuminance);

    if (historyLength >= spatialVarianceEstimationHistoryThreshold) // Running Atrous 3x3
    {
        // Calculating variance, filtered using 3x3 gaussin blur
        float centerDiffuseVar;
        float centerSpecularVar;

        computeVariance(
            threadPos,
            centerDiffuseVar,
            centerSpecularVar,
            border,
            sharedDiffuseIllum,
            sharedSpecularIllum,
            bufferSize);

        // Separate weight parameters for diffuse and specular
        float centerDiffuseLuminance = luminance(sharedDiffuseIllum[sharedMemoryIndex.y * bufferSize + sharedMemoryIndex.x].xyz);
        float centerSpecularLuminance = luminance(sharedSpecularIllum[sharedMemoryIndex.y * bufferSize + sharedMemoryIndex.x].xyz);
        
        float diffusePhiLIlluminationInv = 1.0f / max(1.0e-4f, diffusePhiLuminance * sqrt(centerDiffuseVar));
        float specularPhiLIlluminationInv = 1.0f / max(1.0e-4f, specularPhiLuminanceAdapted * sqrt(centerSpecularVar));

        float diffuseNormalWeightParam = GetNormalWeightParam2(1.0f, diffuseLobeAngleFraction);
        float specularNormalWeightParam = GetNormalWeightParam2(1.0f, specularLobeAngleFraction);
        float specularNormalWeightFromRoughness = getSpecularNormalWeightFromRoughness(centerRoughness);

        float sumWDiffuse = 0.0f;
        Float4 sumDiffuseIlluminationAnd2ndMoment = Float4(0.0f);
        float sumWSpecular = 0.0f;
        Float4 sumSpecularIlluminationAnd2ndMoment = Float4(0.0f);

        const float kernelWeightGaussian3x3[2] = {0.44198f, 0.27901f};
        float finalDepthThreshold = depthThreshold * centerViewZ;

        // float diffuseMinLuminanceWeight = 0.0f;
        // float maxDiffuseLuminanceRelativeDifference = -log(saturate(diffuseMinLuminanceWeight));

        for (int cx = -1; cx <= 1; cx++)
        {
            for (int cy = -1; cy <= 1; cy++)
            {
                const Int2 p = pixelPos + Int2(cx, cy);
                const bool isCenter = ((cx == 0) && (cy == 0));
                const bool isInside = p.x >= 0 && p.y >= 0 && p.x < screenResolution.x && p.y < screenResolution.y;
                const float kernel = isInside ? kernelWeightGaussian3x3[abs(cx)] * kernelWeightGaussian3x3[abs(cy)] : 0.0f;

                Int2 sharedMemoryIndexP = sharedMemoryIndex + Int2(cx, cy);

                Float4 sampleNormalRoughness = sharedNormalRoughness[sharedMemoryIndexP.y * bufferSize + sharedMemoryIndexP.x];
                Float3 sampleNormal = sampleNormalRoughness.xyz;
                float sampleRoughness = sampleNormalRoughness.w;
                Float4 sampleWorldPosMaterialID = sharedWorldPosMaterialID[sharedMemoryIndexP.y * bufferSize + sharedMemoryIndexP.x];
                Float3 sampleWorldPos = sampleWorldPosMaterialID.xyz;
                float sampleMaterialID = sampleWorldPosMaterialID.w;

                // Calculating geometry weight for diffuse and specular
                float geometryW = GetPlaneDistanceWeight_Atrous(
                    centerWorldPos,
                    centerNormal,
                    sampleWorldPos,
                    finalDepthThreshold);

                geometryW *= kernel;

                // Calculating weights for diffuse
                float angled = AcosApprox(dot(centerNormal, sampleNormal));
                float normalWDiffuse = ComputeNonExponentialWeight(angled, diffuseNormalWeightParam, 0.0f);

                // Summing up diffuse
                Float4 sampleDiffuseIlluminationAnd2ndMoment = sharedDiffuseIllum[sharedMemoryIndexP.y * bufferSize + sharedMemoryIndexP.x];
                float sampleDiffuseLuminance = luminance(sampleDiffuseIlluminationAnd2ndMoment.xyz);

                float diffuseLuminanceW = abs(centerDiffuseLuminance - sampleDiffuseLuminance) * diffusePhiLIlluminationInv;

                float wDiffuse = geometryW * normalWDiffuse * exp(-diffuseLuminanceW);
                wDiffuse = isCenter ? kernel : wDiffuse;
                wDiffuse *= (sampleMaterialID == centerMaterialID);

                sumWDiffuse += wDiffuse;
                sumDiffuseIlluminationAnd2ndMoment += wDiffuse * sampleDiffuseIlluminationAnd2ndMoment;
                
                // Summing up specular
                Float4 sampleSpecularIlluminationAnd2ndMoment = sharedSpecularIllum[sharedMemoryIndexP.y * bufferSize + sharedMemoryIndexP.x];
                float sampleSpecularLuminance = luminance(sampleSpecularIlluminationAnd2ndMoment.xyz);
                
                // Enhanced specular normal weight considering roughness
                float specularNormalW = ComputeNonExponentialWeight(angled, specularNormalWeightParam, 0.0f);
                specularNormalW = pow(specularNormalW, specularNormalWeightFromRoughness);
                
                // Roughness weight for specular
                float roughnessW = exp(-16.0f * abs(centerRoughness - sampleRoughness));
                
                float specularLuminanceW = abs(centerSpecularLuminance - sampleSpecularLuminance) * specularPhiLIlluminationInv;
                
                float wSpecular = geometryW * specularNormalW * roughnessW * exp(-specularLuminanceW);
                wSpecular = isCenter ? kernel : wSpecular;
                wSpecular *= (sampleMaterialID == centerMaterialID);
                
                sumWSpecular += wSpecular;
                sumSpecularIlluminationAnd2ndMoment += wSpecular * sampleSpecularIlluminationAnd2ndMoment;
            }
        }

        sumWDiffuse = max(sumWDiffuse, 1e-6f);
        sumWSpecular = max(sumWSpecular, 1e-6f);
        
        sumDiffuseIlluminationAnd2ndMoment /= sumWDiffuse;
        sumSpecularIlluminationAnd2ndMoment /= sumWSpecular;
        
        float diffuse1stMoment = luminance(sumDiffuseIlluminationAnd2ndMoment.xyz);
        float diffuse2ndMoment = sumDiffuseIlluminationAnd2ndMoment.w;
        float diffuseVariance = max(0.0f, diffuse2ndMoment - diffuse1stMoment * diffuse1stMoment);
        Float4 filteredDiffuseIlluminationAndVariance = Float4(sumDiffuseIlluminationAnd2ndMoment.xyz, diffuseVariance);
        
        float specular1stMoment = luminance(sumSpecularIlluminationAnd2ndMoment.xyz);
        float specular2ndMoment = sumSpecularIlluminationAnd2ndMoment.w;
        float specularVariance = max(0.0f, specular2ndMoment - specular1stMoment * specular1stMoment);
        Float4 filteredSpecularIlluminationAndVariance = Float4(sumSpecularIlluminationAnd2ndMoment.xyz, specularVariance);

        Store2DFloat4(filteredDiffuseIlluminationAndVariance, diffuseIlluminationOutputBuffer, pixelPos);
        Store2DFloat4(filteredSpecularIlluminationAndVariance, specularIlluminationOutputBuffer, pixelPos);
    }
    else
    // Running spatial variance estimation
    {

        float sumWDiffuseIllumination = 0.0f;
        Float3 sumDiffuseIllumination = Float3(0.0f);
        float sumDiffuse1stMoment = 0.0f;
        float sumDiffuse2ndMoment = 0.0f;
        
        float sumWSpecularIllumination = 0.0f;
        Float3 sumSpecularIllumination = Float3(0.0f);
        float sumSpecular1stMoment = 0.0f;
        float sumSpecular2ndMoment = 0.0f;

        // Separate normal weight parameters for diffuse and specular
        float diffuseNormalWeightParam = GetNormalWeightParam2(1.0f, diffuseLobeAngleFraction);
        float specularNormalWeightParam = GetNormalWeightParam2(1.0f, specularLobeAngleFraction);
        float specularNormalWeightFromRoughness = getSpecularNormalWeightFromRoughness(centerRoughness);

        // Compute first and second moment spatially. This code also applies cross-bilateral
        // filtering on the input illumination.
        for (int cx = -2; cx <= 2; cx++)
        {
            for (int cy = -2; cy <= 2; cy++)
            {
                Int2 sharedMemoryIndexP = sharedMemoryIndex + Int2(cx, cy);

                Float4 sampleNormalRoughness = sharedNormalRoughness[sharedMemoryIndexP.y * bufferSize + sharedMemoryIndexP.x];
                Float3 sampleNormal = sampleNormalRoughness.xyz;
                float sampleRoughness = sampleNormalRoughness.w;
                float sampleMaterialID = sharedWorldPosMaterialID[sharedMemoryIndexP.y * bufferSize + sharedMemoryIndexP.x].w;

                // Calculating weights
                float depthW = 1.0f; // TODO: should we take in account depth here?
                float angle = AcosApprox(dot(centerNormal, sampleNormal));
                float diffuseNormalW = ComputeNonExponentialWeight(angle, diffuseNormalWeightParam, 0.0f);
                float specularNormalW = ComputeNonExponentialWeight(angle, specularNormalWeightParam, 0.0f);
                specularNormalW = pow(specularNormalW, specularNormalWeightFromRoughness);
                
                // Roughness weight for specular
                float roughnessW = exp(-16.0f * abs(centerRoughness - sampleRoughness));

                Float4 sampleDiffuse = sharedDiffuseIllum[sharedMemoryIndexP.y * bufferSize + sharedMemoryIndexP.x];
                Float3 sampleDiffuseIllumination = sampleDiffuse.xyz;
                float sampleDiffuse1stMoment = luminance(sampleDiffuseIllumination);
                float sampleDiffuse2ndMoment = sampleDiffuse.w;
                float diffuseW = diffuseNormalW * depthW;
                diffuseW *= (sampleMaterialID == centerMaterialID);

                sumWDiffuseIllumination += diffuseW;
                sumDiffuseIllumination += sampleDiffuseIllumination * diffuseW;
                sumDiffuse1stMoment += sampleDiffuse1stMoment * diffuseW;
                sumDiffuse2ndMoment += sampleDiffuse2ndMoment * diffuseW;
                
                Float4 sampleSpecular = sharedSpecularIllum[sharedMemoryIndexP.y * bufferSize + sharedMemoryIndexP.x];
                Float3 sampleSpecularIllumination = sampleSpecular.xyz;
                float sampleSpecular1stMoment = luminance(sampleSpecularIllumination);
                float sampleSpecular2ndMoment = sampleSpecular.w;
                float specularW = specularNormalW * roughnessW * depthW;
                specularW *= (sampleMaterialID == centerMaterialID);
                
                sumWSpecularIllumination += specularW;
                sumSpecularIllumination += sampleSpecularIllumination * specularW;
                sumSpecular1stMoment += sampleSpecular1stMoment * specularW;
                sumSpecular2ndMoment += sampleSpecular2ndMoment * specularW;
            }
        }

        float boost = max(1.0f, 4.0f / (historyLength + 1.0f));

        sumWDiffuseIllumination = max(sumWDiffuseIllumination, 1e-6f);
        sumWSpecularIllumination = max(sumWSpecularIllumination, 1e-6f);
        
        sumDiffuseIllumination /= sumWDiffuseIllumination;
        sumDiffuse1stMoment /= sumWDiffuseIllumination;
        sumDiffuse2ndMoment /= sumWDiffuseIllumination;
        float diffuseVariance = max(0.0f, sumDiffuse2ndMoment - sumDiffuse1stMoment * sumDiffuse1stMoment);
        diffuseVariance *= boost;
        
        sumSpecularIllumination /= sumWSpecularIllumination;
        sumSpecular1stMoment /= sumWSpecularIllumination;
        sumSpecular2ndMoment /= sumWSpecularIllumination;
        float specularVariance = max(0.0f, sumSpecular2ndMoment - sumSpecular1stMoment * sumSpecular1stMoment);
        specularVariance *= boost;

        Store2DFloat4(Float4(sumDiffuseIllumination, diffuseVariance), diffuseIlluminationOutputBuffer, pixelPos);
        Store2DFloat4(Float4(sumSpecularIllumination, specularVariance), specularIlluminationOutputBuffer, pixelPos);
    }
}