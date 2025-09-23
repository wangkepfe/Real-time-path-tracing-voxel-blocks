#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"

// Note: Using functions from DenoiserCommon.h to avoid duplicates

__global__ void Atrous(
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
    
    // Confidence buffers
    SurfObj diffuseConfidenceBuffer,
    SurfObj specularConfidenceBuffer,
    SurfObj specularReprojectionConfidenceBuffer,

    Camera camera,
    unsigned int frameIndex,
    unsigned int stepSize,

    // Denoising parameters
    float phiLuminance,
    float specularPhiLuminance,
    float depthThreshold,
    float roughnessFraction,
    float diffuseLobeAngleFraction,
    float specularLobeAngleFraction,
    float specularLobeAngleSlack,
    
    // Confidence parameters  
    bool useConfidenceAdaptation,
    float confidenceDrivenRelaxationMultiplier,
    float confidenceDrivenLuminanceEdgeStoppingRelaxation,
    float normalEdgeStoppingRelaxation)
{
    Int2 threadPos;
    threadPos.x = threadIdx.x;
    threadPos.y = threadIdx.y;

    Int2 pixelPos;
    pixelPos.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelPos.y = blockIdx.y * blockDim.y + threadIdx.y;

    //unsigned int threadIndex = threadIdx.z * (blockDim.x * blockDim.y) +
    //                           threadIdx.y * blockDim.x +
    //                           threadIdx.x;

    float centerViewZ = Load2DFloat1(depthBuffer, pixelPos);

    // Early out - using hardcoded value for now
    if (centerViewZ > 500000.0f) // TODO: Pass denoisingRange parameter
        return;

    float centerMaterialID = Load2DUshort1(materialBuffer, pixelPos);
    Float4 centerNormalRoughness = Load2DFloat4(normalRoughnessBuffer, pixelPos);
    Float3 centerNormal = centerNormalRoughness.xyz;
    float centerRoughness = centerNormalRoughness.w;
    Float3 centerWorldPos = GetCurrentWorldPosFromPixelPos(camera, pixelPos, centerViewZ);
    Float3 centerViewVector = -normalize(centerWorldPos - camera.pos);

    float historyLength = Load2DFloat1(historyLengthBuffer, pixelPos);

    // Weight strictness increases with A-trous step size for both diffuse and specular
    float adaptedDiffuseLobeAngleFraction = diffuseLobeAngleFraction / sqrtf((float)stepSize);
    adaptedDiffuseLobeAngleFraction = lerp(0.99f, adaptedDiffuseLobeAngleFraction, saturate(historyLength / 5.0f));
    
    float adaptedSpecularLobeAngleFraction = specularLobeAngleFraction / sqrtf((float)stepSize);
    adaptedSpecularLobeAngleFraction = lerp(0.99f, adaptedSpecularLobeAngleFraction, saturate(historyLength / 5.0f));

    Float4 centerDiffuseIlluminationAndVariance = Load2DFloat4(diffuseIlluminationInputBuffer, pixelPos);
    float centerDiffuseLuminance = luminance(centerDiffuseIlluminationAndVariance.xyz);
    float centerDiffuseVar = centerDiffuseIlluminationAndVariance.w;
    float diffusePhiLIlluminationInv = 1.0f / max(1.0e-4f, phiLuminance * sqrt(centerDiffuseVar));
    
    Float4 centerSpecularIlluminationAndVariance = Load2DFloat4(specularIlluminationInputBuffer, pixelPos);
    float centerSpecularLuminance = luminance(centerSpecularIlluminationAndVariance.xyz);
    float centerSpecularVar = centerSpecularIlluminationAndVariance.w;
    float specularPhiLuminanceAdapted = getSpecularLuminanceWeightFromRoughness(centerRoughness, specularPhiLuminance);
    float specularPhiLIlluminationInv = 1.0f / max(1.0e-4f, specularPhiLuminanceAdapted * sqrt(centerSpecularVar));

    // Load specular reprojection confidence for enhanced normal weighting
    float specularReprojectionConfidence = 1.0f;
    if (specularReprojectionConfidenceBuffer != 0)
    {
        specularReprojectionConfidence = Load2DFloat1(specularReprojectionConfidenceBuffer, pixelPos);
    }

    // Confidence-driven luminance weight relaxation for both diffuse and specular
    float diffuseLuminanceWeightRelaxation = 1.0f;
    float specularLuminanceWeightRelaxation = 1.0f;
    if (useConfidenceAdaptation)
    {
        if (diffuseConfidenceBuffer != 0)
        {
            float diffuseConfidence = Load2DFloat1(diffuseConfidenceBuffer, pixelPos);
            float confidenceDrivenRelaxation = saturate(confidenceDrivenRelaxationMultiplier * (1.0f - diffuseConfidence));
            diffuseLuminanceWeightRelaxation = 1.0f - saturate(confidenceDrivenRelaxation * confidenceDrivenLuminanceEdgeStoppingRelaxation);
        }
        if (specularConfidenceBuffer != 0)
        {
            float specularConfidence = Load2DFloat1(specularConfidenceBuffer, pixelPos);
            float confidenceDrivenRelaxation = saturate(confidenceDrivenRelaxationMultiplier * (1.0f - specularConfidence));
            specularLuminanceWeightRelaxation = 1.0f - saturate(confidenceDrivenRelaxation * confidenceDrivenLuminanceEdgeStoppingRelaxation);
        }
    }
    
    float diffuseNormalWeightParam = GetNormalWeightParam2(1.0f, adaptedDiffuseLobeAngleFraction);
    
    // Enhanced specular normal weight calculation with view-vector dependency
    Float2 specularNormalWeightParams = GetSpecularNormalWeightParams(
        centerRoughness,
        historyLength,
        specularReprojectionConfidence,
        normalEdgeStoppingRelaxation,
        adaptedSpecularLobeAngleFraction,
        specularLobeAngleSlack
    );
    
    // Fallback simplified specular normal weight for compatibility
    float specularNormalWeightParamSimplified = GetNormalWeightParam2(1.0f, adaptedSpecularLobeAngleFraction);
    float specularNormalWeightFromRoughness = getSpecularNormalWeightFromRoughness(centerRoughness);
    
    // Roughness weight parameters for specular
    Float2 roughnessWeightParams = GetRoughnessWeightParams(centerRoughness, roughnessFraction);

    float sumWDiffuse = 0.44198f * 0.44198f;
    Float4 sumDiffuseIlluminationAndVariance = centerDiffuseIlluminationAndVariance * Float4(sumWDiffuse, sumWDiffuse, sumWDiffuse, sumWDiffuse * sumWDiffuse);
    float sumWSpecular = 0.44198f * 0.44198f;
    Float4 sumSpecularIlluminationAndVariance = centerSpecularIlluminationAndVariance * Float4(sumWSpecular, sumWSpecular, sumWSpecular, sumWSpecular * sumWSpecular);

    const float kernelWeightGaussian3x3[2] = {0.44198f, 0.27901f};

    float finalDepthThreshold = depthThreshold * centerViewZ;

    // Adding random offsets to minimize "ringing" at large A-Trous steps
    unsigned int rngHashState;
    Int2 offset = Int2(0);
    if (stepSize > 4)
    {
        RngHashInitialize(rngHashState, UInt2(pixelPos.x, pixelPos.y), frameIndex);
        Float2 offsetF = Float2(stepSize, stepSize) * 0.5f * (RngHashGetFloat2(rngHashState) - 0.5f);
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
            //int sampleIndexDebug = (yy + 1) * 3 + (xx + 1);

            Int2 p = pixelPos + offset + Int2(xx, yy) * stepSize;
            bool isCenter = ((xx == 0) && (yy == 0));
            if (isCenter)
                continue;

            bool isInside = p.x >= 0 && p.y >= 0 && p.x < screenResolution.x && p.y < screenResolution.y;
            float kernel = kernelWeightGaussian3x3[abs(xx)] * kernelWeightGaussian3x3[abs(yy)];

            // Fetching normal, roughness, linear Z
            // Calculating sample world position
            float sampleMaterialID = Load2DUshort1(materialBuffer, p);
            Float4 sampleNormalRoughness = Load2DFloat4(normalRoughnessBuffer, p);
            Float3 sampleNormal = sampleNormalRoughness.xyz;
            float sampleRoughness = sampleNormalRoughness.w;
            float sampleViewZ = Load2DFloat1(depthBuffer, p);
            Float3 sampleWorldPos = GetCurrentWorldPosFromPixelPos(camera, p, sampleViewZ);
            Float3 sampleViewVector = -normalize(sampleWorldPos - camera.pos);

            // Calculating geometry weight for diffuse and specular
            float geometryW = GetPlaneDistanceWeight_Atrous(centerWorldPos, centerNormal, sampleWorldPos, finalDepthThreshold);
            geometryW *= kernel;
            geometryW *= float(isInside && sampleViewZ < 500000.0f); // TODO: Use denoisingRange parameter

            // Calculating weights for diffuse (simple angle-based)
            float angled = AcosApprox(dot(centerNormal, sampleNormal));
            float normalWDiffuse = ComputeNonExponentialWeight(angled, diffuseNormalWeightParam, 0.0f);
            
            // Enhanced specular normal weight with view-vector dependency
            float normalWSpecular = GetSpecularNormalWeightWithViewVector(
                specularNormalWeightParams, 
                centerNormal, sampleNormal, 
                centerViewVector, sampleViewVector
            );
            
            // Fallback to simplified calculation if needed
            if (normalWSpecular == 0.0f) {
                float angles = AcosApprox(dot(centerNormal, sampleNormal));
                normalWSpecular = ComputeNonExponentialWeight(angles, specularNormalWeightParamSimplified, 0.0f);
                normalWSpecular = pow(normalWSpecular, specularNormalWeightFromRoughness);
            }
            
            // Material ID check (common for both)
            float materialWeight = (sampleMaterialID == centerMaterialID) ? 1.0f : 0.0f;

            // Summing up diffuse
            float wDiffuse = geometryW * normalWDiffuse * materialWeight;
            if (wDiffuse > 1e-4f)
            {
                Float4 sampleDiffuseIlluminationAndVariance = Load2DFloat4(diffuseIlluminationInputBuffer, p);
                float sampleDiffuseLuminance = luminance(sampleDiffuseIlluminationAndVariance.xyz);

                float diffuseLuminanceW = abs(centerDiffuseLuminance - sampleDiffuseLuminance) * diffusePhiLIlluminationInv;
                diffuseLuminanceW = min(maxDiffuseLuminanceRelativeDifference, diffuseLuminanceW);
                diffuseLuminanceW *= diffuseLuminanceWeightRelaxation;

                wDiffuse *= exp(-diffuseLuminanceW);

                sumWDiffuse += wDiffuse;
                sumDiffuseIlluminationAndVariance += Float4(wDiffuse, wDiffuse, wDiffuse, wDiffuse * wDiffuse) * sampleDiffuseIlluminationAndVariance;
            }
            
            // Summing up specular with enhanced roughness weighting
            float roughnessW = ComputeWeight(sampleRoughness, roughnessWeightParams.x, roughnessWeightParams.y);
            float wSpecular = geometryW * normalWSpecular * roughnessW * materialWeight;
            if (wSpecular > 1e-4f)
            {
                Float4 sampleSpecularIlluminationAndVariance = Load2DFloat4(specularIlluminationInputBuffer, p);
                float sampleSpecularLuminance = luminance(sampleSpecularIlluminationAndVariance.xyz);

                float specularLuminanceW = abs(centerSpecularLuminance - sampleSpecularLuminance) * specularPhiLIlluminationInv;
                specularLuminanceW *= specularLuminanceWeightRelaxation;

                wSpecular *= exp(-specularLuminanceW);

                sumWSpecular += wSpecular;
                sumSpecularIlluminationAndVariance += Float4(wSpecular, wSpecular, wSpecular, wSpecular * wSpecular) * sampleSpecularIlluminationAndVariance;
            }
        }
    }
    Float4 filteredDiffuseIlluminationAndVariance = Float4(sumDiffuseIlluminationAndVariance.xyz / sumWDiffuse, sumDiffuseIlluminationAndVariance.w / (sumWDiffuse * sumWDiffuse));
    Float4 filteredSpecularIlluminationAndVariance = Float4(sumSpecularIlluminationAndVariance.xyz / sumWSpecular, sumSpecularIlluminationAndVariance.w / (sumWSpecular * sumWSpecular));

    Store2DFloat4(filteredDiffuseIlluminationAndVariance, diffuseIlluminationOutputBuffer, pixelPos);
    Store2DFloat4(filteredSpecularIlluminationAndVariance, specularIlluminationOutputBuffer, pixelPos);
}