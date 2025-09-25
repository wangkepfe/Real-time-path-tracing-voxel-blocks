#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"
#include "shaders/ShaderDebugUtils.h"
#include "core/GlobalSettings.h"

// RELAX denoiser configuration
#define RELAX_MAX_ACCUM_FRAME_NUM 63.0f
#define RELAX_NORMAL_ULP 0.001f
#define RELAX_USE_BICUBIC_FOR_VIRTUAL_MOTION 1
#define RELAX_USE_CATROM_FOR_SURFACE_MOTION_IN_TA 1
#define RELAX_USE_HIGH_PARALLAX_CURVATURE 1
#define RELAX_USE_HIGH_PARALLAX_CURVATURE_SILHOUETTE_FIX 1
#define RELAX_USE_SPECULAR_MOTION_V2 1
#define RELAX_MAX_ALLOWED_VIRTUAL_MOTION_ACCELERATION 10.0f
#define RELAX_CURVATURE_Z_THRESHOLD 0.01f

// Advanced NRD parameters structure for denoising
struct RELAXDenoisingParams
{
    bool resetHistory;
    uint32_t frameIndex;
    float framerateScale;
    float roughnessFraction;
    uint32_t strandMaterialID;
    float strandThickness;
    uint32_t cameraAttachedReflectionMaterialID;
    float specVarianceBoost;
    bool hasHistoryConfidence;
    bool hasDisocclusionThresholdMix;
};

__device__ float ComputeParallaxInPixels(Float3 X, Float2 uvForZeroParallax, Camera camera, Float2 rectSize)
{
    // Both produce same results, but behavior is different on objects attached to the camera:
    // non-0 parallax on pure translations, 0 parallax on pure rotations
    //      ComputeParallaxInPixels( Xprev + gCameraDelta, gOrthoMode == 0.0 ? smbPixelUv : pixelUv, gWorldToClipPrev, gRectSize );
    // 0 parallax on translations, non-0 parallax on pure rotations
    //      ComputeParallaxInPixels( Xprev - gCameraDelta, gOrthoMode == 0.0 ? pixelUv : smbPixelUv, gWorldToClip, gRectSize );

    Float2 uv = camera.worldDirectionToUV(normalize(X - camera.pos));
    Float2 parallaxInUv = uv - uvForZeroParallax;
    float parallaxInPixels = length(parallaxInUv * rectSize);

    return parallaxInPixels;
}

__device__ float isReprojectionTapValid(Float3 currentWorldPos, Float3 previousWorldPos, Float3 currentNormal, float disocclusionThreshold)
{
    // Check if plane distance is acceptable
    Float3 posDiff = currentWorldPos - previousWorldPos;
    float maxPlaneDistance = abs(dot(posDiff, currentNormal));

    return maxPlaneDistance > disocclusionThreshold ? 0.0f : 1.0f;
}

__device__ float ApplyThinLensEquation(float hitDist, float curvature)
{
    return hitDist / max(1.0f + hitDist * curvature, 0.001f);
}

__device__ Float3 GetXvirtual(float hitDist, float curvature, Float3 currentWorldPos, Float3 prevWorldPos, Float3 currentNormal, Float3 V, float roughness)
{
    // Apply thin lens equation for virtual position calculation
    float focusedHitDist = ApplyThinLensEquation(hitDist, curvature);

    // Calculate reflection direction
    Float3 reflectionDir = reflect3f(-V, currentNormal);

    // Virtual position along reflection direction
    Float3 virtualWorldPos = currentWorldPos + reflectionDir * focusedHitDist;

    return virtualWorldPos;
}

// GetSpecLobeTanHalfAngle is already defined in DenoiserCommon.h

__device__ float GetEncodingAwareNormalWeight(Float3 n1, Float3 n2, float angle, float curvatureAngle = 0.0f, float relaxation = 0.0f, bool useAngleBased = false)
{
    float cosa = saturate(dot(n1, n2));
    float angle1 = acos(cosa);
    float w = 1.0f;

    if (useAngleBased)
    {
        w = saturate(1.0f - angle1 / max(angle + curvatureAngle + relaxation, 1e-6f));
    }
    else
    {
        float threshold = cos(angle + curvatureAngle + relaxation);
        w = cosa > threshold ? 1.0f : 0.0f;
    }

    return w;
}

__device__ float ComputeWeight(float value, float referenceValue, float sensitivity)
{
    return exp(-abs(value - referenceValue) * sensitivity);
}

__device__ Float2 GetRelaxedRoughnessWeightParams(float roughness2, float fraction)
{
    float a = lerp(0.0f, 0.99f, saturate(roughness2 * 2.0f));
    float b = 1.0f / (lerp(0.01f, 1.0f, a) * fraction + 1e-6f);
    return Float2(a, b);
}

__device__ Float3 GetPreviousWorldPosFromPixelPos(Camera prevCamera, Int2 pixelPos, float linearZ)
{
    // Similar to GetCurrentWorldPosFromPixelPos but for previous camera
    Float2 uv = (Float2(pixelPos.x, pixelPos.y) + 0.5f) * prevCamera.inversedResolution;
    Float3 viewVector = prevCamera.uvToWorldDirection(uv);
    return prevCamera.pos + viewVector * linearZ;
}

__device__ float GetSpecMagicCurve(float roughness)
{
    // Magic curve for specular accumulation based on roughness
    float f = 1.0f - exp(-200.0f * roughness * roughness);
    return f * f;
}

__device__ Float4 GetSpecularDominantDirection(Float3 N, Float3 V, float roughness)
{
    // Simplified specular dominant direction calculation
    // Based on GGX distribution dominant direction
    float m = roughness * roughness;
    float f = (1.0f - m) / (1.0f + m);

    Float3 R = reflect3f(-V, N);
    Float3 dominantDir = normalize(lerp(R, N, roughness));

    // Return direction and confidence weight
    float confidence = f * f;
    return Float4(dominantDir, confidence);
}

__device__ float ComputeVirtualHistoryAmount(Float3 currentNormal, Float3 prevNormal, Float3 currentNormalAveraged,
                                             float currentRoughness, float prevRoughness,
                                             Float2 uvDiff, Float2 screenSize, float parallaxInPixels,
                                             float curvatureAngle, bool isOrthoMode)
{
    // Amount of virtual motion - dominant factor
    Float4 D = GetSpecularDominantDirection(currentNormal, normalize(currentNormal), currentRoughness);
    float virtualHistoryAmount = D.w;

    // Decreasing virtual history amount for ortho case
    virtualHistoryAmount *= isOrthoMode ? 0.75f : 1.0f;

    // Virtual motion amount - back-facing check
    virtualHistoryAmount *= (dot(prevNormal, currentNormalAveraged) > 0.0f) ? 1.0f : 0.0f;

    // Normal weight for virtual motion based reprojection
    float uvDiffLengthInPixels = length(uvDiff * screenSize);
    float lobeHalfAngle = max(atan(GetSpecLobeTanHalfAngle(currentRoughness)), RELAX_NORMAL_ULP);
    float normalWeight = GetEncodingAwareNormalWeight(currentNormal, prevNormal, lobeHalfAngle, curvatureAngle, RELAX_NORMAL_ULP, true);
    virtualHistoryAmount *= lerp(1.0f - saturate(uvDiffLengthInPixels), 1.0f, normalWeight);

    // Roughness weight for virtual motion based reprojection
    Float2 relaxedRoughnessWeightParams = GetRelaxedRoughnessWeightParams(currentRoughness * currentRoughness, 0.99f);
    float virtualRoughnessWeight = ComputeWeight(prevRoughness * prevRoughness, relaxedRoughnessWeightParams.x, relaxedRoughnessWeightParams.y);
    virtualRoughnessWeight = lerp(1.0f - saturate(uvDiffLengthInPixels), 1.0f, virtualRoughnessWeight);
    virtualHistoryAmount *= isOrthoMode ? 1.0f : virtualRoughnessWeight;

    return virtualHistoryAmount;
}

__device__ float ComputeVirtualHistoryHitDistConfidence(float currentHitDist, float prevHitDist,
                                                        float currentLinearZ, float curvature,
                                                        float currentRoughness)
{
    // Virtual history confidence - hit distance
    float SMC = GetSpecMagicCurve(currentRoughness);
    float hitDist1 = ApplyThinLensEquation(currentHitDist, curvature);
    float hitDist2 = ApplyThinLensEquation(prevHitDist, curvature);
    float maxDist = max(hitDist1, hitDist2);
    float dHitT = abs(hitDist1 - hitDist2);
    float dHitTMultiplier = lerp(20.0f, 0.0f, SMC);
    float virtualHistoryHitDistConfidence = 1.0f - saturate(dHitTMultiplier * dHitT / (currentLinearZ + maxDist));
    virtualHistoryHitDistConfidence = lerp(virtualHistoryHitDistConfidence, 1.0f, SMC);

    return virtualHistoryHitDistConfidence;
}

__device__ float ComputeLookingBackValidation(Float2 prevUVVMB, Float2 uvDiff, Int2 screenResolution,
                                              SurfObj prevNormalRoughnessBuffer,
                                              Quat worldPrevToWorldRotation,
                                              Float3 prevNormalVMB, float currentRoughness,
                                              float lobeHalfAngle, float curvatureAngle)
{
    // "Looking back" 1 and 2 frames and applying normal weight to decrease lags
    Float2 normalizedUvDiff = normalize(uvDiff);
    float uvDiffLength = length(uvDiff);
    normalizedUvDiff *= saturate(uvDiffLength / 0.1f) + uvDiffLength / 2.0f;

    Float2 backUV1 = prevUVVMB + 1.0f * normalizedUvDiff / Float2(screenResolution.x, screenResolution.y);
    Float2 backUV2 = prevUVVMB + 2.0f * normalizedUvDiff / Float2(screenResolution.x, screenResolution.y);

    // Sample normals from 1 and 2 frames back
    Float4 backNormalRoughness1 = Load2DFloat4(prevNormalRoughnessBuffer, Int2(backUV1.x * screenResolution.x, backUV1.y * screenResolution.y));
    Float4 backNormalRoughness2 = Load2DFloat4(prevNormalRoughnessBuffer, Int2(backUV2.x * screenResolution.x, backUV2.y * screenResolution.y));

    // Transform normals from previous frame to current frame
    backNormalRoughness1.xyz = normalize(rotate(worldPrevToWorldRotation, backNormalRoughness1.xyz).v);
    backNormalRoughness2.xyz = normalize(rotate(worldPrevToWorldRotation, backNormalRoughness2.xyz).v);

    // Check if UVs are in screen bounds
    bool inScreen1 = (backUV1.x >= 0.0f && backUV1.x <= 1.0f && backUV1.y >= 0.0f && backUV1.y <= 1.0f);
    bool inScreen2 = (backUV2.x >= 0.0f && backUV2.x <= 1.0f && backUV2.y >= 0.0f && backUV2.y <= 1.0f);

    // Compute normal weights
    float prevPrevNormalWeight1 = inScreen1 ? GetEncodingAwareNormalWeight(prevNormalVMB, backNormalRoughness1.xyz, lobeHalfAngle, curvatureAngle * 2.0f, RELAX_NORMAL_ULP, true) : 1.0f;
    float prevPrevNormalWeight2 = inScreen2 ? GetEncodingAwareNormalWeight(prevNormalVMB, backNormalRoughness2.xyz, lobeHalfAngle, curvatureAngle * 3.0f, RELAX_NORMAL_ULP, true) : 1.0f;

    float prevPrevNormalWeight = prevPrevNormalWeight1 * prevPrevNormalWeight2;

    // Taking into account roughness 1 and 2 frames back helps cleaning up surfaces with varying roughness
    Float2 relaxedRoughnessWeightParams = GetRelaxedRoughnessWeightParams(currentRoughness * currentRoughness, 0.99f);
    float rw1 = ComputeWeight(backNormalRoughness1.w * backNormalRoughness1.w, relaxedRoughnessWeightParams.x, relaxedRoughnessWeightParams.y);
    float rw2 = ComputeWeight(backNormalRoughness2.w * backNormalRoughness2.w, relaxedRoughnessWeightParams.x, relaxedRoughnessWeightParams.y);
    float roughnessWeight = rw1 * rw2;

    return (0.33f + 0.67f * prevPrevNormalWeight) * (roughnessWeight * 0.9f + 0.1f);
}

// Compute specular dominant direction weight (based on NRD RELAX ImportanceSampling::GetSpecularDominantDirection)
__device__ float GetSpecularDominantDirectionWeight(Float3 N, Float3 V, float roughness, float NoV)
{
    // Approximation of the dominant direction weight from NRD RELAX
    // This accounts for the viewing angle and roughness to determine how much
    // the specular lobe contributes to temporal reprojection
    float alpha = roughness * roughness;

    // Energy-based weight approximation
    // For smooth surfaces (low roughness), the dominant direction is strong
    // For rough surfaces (high roughness), the dominant direction is weak
    float roughnessFactor = 1.0f - alpha;

    // Viewing angle factor - grazing angles reduce dominant direction strength
    float NoVFactor = lerp(0.5f, 1.0f, NoV);

    // Combined weight (matching NRD's D.w behavior)
    return roughnessFactor * NoVFactor;
}

__device__ void Preload(Int2 sharedPos,
                        Int2 globalPos,
                        Int2 screenResolution,
                        SurfObj normalRoughnessBuffer,
                        Float4 *sharedNormalRoughness,
                        int bufferSize)
{
    globalPos = clamp2i(globalPos, Int2(0), Int2(screenResolution.x - 1, screenResolution.y - 1));
    sharedNormalRoughness[sharedPos.y * bufferSize + sharedPos.x] = Load2DFloat4(normalRoughnessBuffer, globalPos);
}

template <int groupSize, int border>
__global__ void TemporalAccumulation(
    Int2 screenResolution,
    Float2 invScreenResolution,

    SurfObj prevDepthBuffer,
    SurfObj prevMatBuffer,
    SurfObj prevDiffuseIllumBuffer,
    SurfObj prevDiffuseFastIllumBuffer,
    SurfObj prevSpecularIllumBuffer,
    SurfObj prevSpecularFastIllumBuffer,
    SurfObj prevSpecularHitDistBuffer,
    SurfObj prevHistoryLengthBuffer,
    SurfObj prevNormalRoughnessBuffer,

    SurfObj diffuseIlluBuffer,
    SurfObj specularIlluBuffer,
    SurfObj depthBuffer,
    SurfObj normalRoughnessBuffer,
    SurfObj materialBuffer,

    SurfObj diffuseIlluminationPingBuffer,
    SurfObj diffuseIlluminationPongBuffer,
    SurfObj specularIlluminationPingBuffer,
    SurfObj specularIlluminationPongBuffer,
    SurfObj specularHitDistBuffer,
    SurfObj specularReprojectionConfidenceBuffer,
    SurfObj historyLengthBuffer,

    // History confidence buffers
    SurfObj diffuseHistoryConfidenceBuffer,
    SurfObj prevDiffuseHistoryConfidenceBuffer,
    SurfObj specularHistoryConfidenceBuffer,
    SurfObj prevSpecularHistoryConfidenceBuffer,
    SurfObj disocclusionThresholdMixBuffer,

    SurfObj debugBuffer,

    Camera camera,
    Camera prevCamera,

    // Denoising parameters
    float denoisingRange,
    float disocclusionThreshold,
    float disocclusionThresholdAlternate,
    float diffuseMaxAccumulatedFrameNum,
    float diffuseMaxFastAccumulatedFrameNum,
    float specularMaxAccumulatedFrameNum,
    float specularMaxFastAccumulatedFrameNum,
    float specularVarianceBoost,
    float roughnessFraction,
    RELAXDenoisingParams relaxParams)
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

    Float2 pixelUv = Float2(Float2(pixelPos.x, pixelPos.y) + 0.5f) * invScreenResolution;

    Int2 groupBase = pixelPos - threadPos - border;
    unsigned int stageNum = (bufferSize * bufferSize + groupSize * groupSize - 1) / (groupSize * groupSize);
    for (unsigned int stage = 0; stage < stageNum; stage++)
    {
        unsigned int virtualIndex = threadIndex + stage * groupSize * groupSize;
        Int2 newId = Int2(virtualIndex % bufferSize, virtualIndex / bufferSize);
        if (stage == 0 || virtualIndex < bufferSize * bufferSize)
        {
            Preload(newId, groupBase + newId, screenResolution, normalRoughnessBuffer, sharedNormalRoughness, bufferSize);
        }
    }
    __syncthreads();

    if (pixelPos.x >= screenResolution.x || pixelPos.y >= screenResolution.y)
    {
        return;
    }

    float currentLinearZ = Load2DFloat1(depthBuffer, pixelPos);

    // Early out - use parameter passed to kernel
    if (currentLinearZ > denoisingRange)
        return;

    Int2 sharedMemoryIndex = threadPos + border;

    // Reading current GBuffer data
    float currentMaterialID = Load2DFloat1(materialBuffer, pixelPos);
    Float4 currentNormalRoughness = Load2DFloat4(normalRoughnessBuffer, pixelPos);
    Float3 currentNormal = currentNormalRoughness.xyz;
    // float currentRoughness = currentNormalRoughness.w;

    // Getting current position and view vector for current pixel
    Float2 currentUV = (Float2(pixelPos.x, pixelPos.y) + 0.5f) * camera.inversedResolution;
    Float3 currentViewVector = camera.uvToWorldDirection(currentUV);
    Float3 currentWorldPos = GetCurrentWorldPosFromPixelPos(camera, pixelPos, currentLinearZ);
    Float3 V = -normalize(currentViewVector);
    float NoV = abs(dot(currentNormal, V));

    // Getting previous position
    Float3 prevWorldPos = currentWorldPos; // TODO: movement of the world
    Float2 prevUVSMB = prevCamera.worldDirectionToUV(normalize(prevWorldPos - prevCamera.pos));

    // if (CUDA_PIXEL(0.5f, 0.75f))
    // {
    //     DEBUG_PRINT(currentUV);
    //     DEBUG_PRINT(prevUVSMB);
    //     DEBUG_PRINT(pixelPos);
    // }

    // Input noisy data
    Float3 diffuseIllumination = Load2DFloat4(diffuseIlluBuffer, pixelPos).xyz;
    Float4 specularIllumination = Load2DFloat4(specularIlluBuffer, pixelPos);

    // Calculating average normal, minHitDist and specular sigma
    Float3 currentNormalAveraged = currentNormal;

    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            // Skipping center pixel
            if ((i == 0) && (j == 0))
                continue;

            Float4 normalRoughness = sharedNormalRoughness[(sharedMemoryIndex.y + j) * bufferSize + (sharedMemoryIndex.x + i)];
            currentNormalAveraged += normalRoughness.xyz;

            // For specular, we need to track hit distance from specular buffer
            // This would require loading from specular buffer in the shared memory preload
        }
    }
    currentNormalAveraged /= 9.0f;

    // Computing 2nd moments of input noisy luminance
    float diffuse1stMoment = luminance(diffuseIllumination);
    float diffuse2ndMoment = diffuse1stMoment * diffuse1stMoment;

    float specular1stMoment = luminance(specularIllumination.xyz);
    float specular2ndMoment = specular1stMoment * specular1stMoment;

    float currentRoughness = currentNormalRoughness.w;
    float currentRoughnessModified = currentRoughness; // Can add normal variance modification later

    // Calculating surface parallax
    Float3 cameraDelta = prevCamera.pos - camera.pos;
    float smbParallaxInPixels1 = ComputeParallaxInPixels(prevWorldPos + cameraDelta, pixelUv, prevCamera, Float2(screenResolution.x, screenResolution.y));
    float smbParallaxInPixels2 = ComputeParallaxInPixels(prevWorldPos - cameraDelta, prevUVSMB, camera, Float2(screenResolution.x, screenResolution.y));

    float smbParallaxInPixelsMax = max(smbParallaxInPixels1, smbParallaxInPixels2);
    float smbParallaxInPixelsMin = min(smbParallaxInPixels1, smbParallaxInPixels2);

    // Calculating disocclusion threshold using passed parameters
    float disocclusionThresholdMix = 0.0f;

    // Check for strand material or disocclusion threshold mix buffer
    if (currentMaterialID == relaxParams.strandMaterialID)
    {
        float pixelSize = camera.getPixelWorldSizeScaleToDepth() * currentLinearZ;
        disocclusionThresholdMix = saturate(relaxParams.strandThickness / pixelSize);
    }
    else if (relaxParams.hasDisocclusionThresholdMix)
    {
        disocclusionThresholdMix = Load2DFloat1(disocclusionThresholdMixBuffer, pixelPos);
    }

    float disocclusionThresholdBonus = disocclusionThreshold + (1.5f / screenResolution.y);
    float disocclusionThresholdAlternateBonus = disocclusionThresholdAlternate + (1.5f / screenResolution.y);

    float finalDisocclusionThreshold = lerp(disocclusionThresholdBonus, disocclusionThresholdAlternateBonus, disocclusionThresholdMix);

    // Loading previous data based on surface motion vectors - inline implementation
    float footprintQuality;
    Float4 prevDiffuseIlluminationAnd2ndMomentSMB;
    Float3 prevDiffuseIlluminationAnd2ndMomentSMBResponsive;
    Float4 prevSpecularIlluminationAnd2ndMomentSMB;
    Float3 prevSpecularIlluminationAnd2ndMomentSMBResponsive;
    float prevReflectionHitTSMB;
    float historyLength;

    Quat worldPrevToWorldRotation = rotationBetween(prevCamera.dir, camera.dir);

    float SMBReprojectionFound = 0.0f;
    {
        float estimatedPrevDepth = length(prevWorldPos - prevCamera.pos);

        // Calculating previous pixel position
        Float2 prevPixelPosFloat = prevUVSMB * screenResolution;

        // Calculating footprint origin and weights
        Float2 bilinearOriginF = floor(prevPixelPosFloat - 0.5f);
        Int2 bilinearOriginSMB = Int2(bilinearOriginF.x, bilinearOriginF.y);
        Float2 bilinearWeights = fract(prevPixelPosFloat - 0.5f);

        // Calculating disocclusion threshold
        float pixelSize = camera.getPixelWorldSizeScaleToDepth() * currentLinearZ;
        float frustumSize = pixelSize * min(screenResolution.x, screenResolution.y);
        float disocclusionThresholdSlopeScale = 1.0f / lerp(lerp(0.05f, 1.0f, NoV), 1.0f, saturate(smbParallaxInPixelsMax / 30.0f));
        Float4 smbDisocclusionThreshold = Float4(saturate(finalDisocclusionThreshold * disocclusionThresholdSlopeScale) * frustumSize);
        smbDisocclusionThreshold *= IsInScreenBilinear(bilinearOriginSMB, screenResolution);
        smbDisocclusionThreshold -= 1e-6f;

        // Checking bicubic footprint (with cut corners)
        Int2 bicubicOffsets[4][2] = {
            {{0, -1}, {-1, 0}},
            {{1, -1}, {2, 0}},
            {{-1, 1}, {0, 2}},
            {{2, 1}, {1, 2}}};

        Int2 bilinearOffsetsSMB[4] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};

        float bicubicFootprintValid = 1.0f;
        Float4 bilinearTapsValidSMB = Float4(0.0f);

        // Check bicubic taps
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                Int2 tapPos = bilinearOriginSMB + bicubicOffsets[i][j];
                float prevViewZInTap = Load2DFloat1(prevDepthBuffer, tapPos);
                float reprojectionTapValid = abs(prevViewZInTap - estimatedPrevDepth) > smbDisocclusionThreshold[i] ? 0.0f : 1.0f;
                bicubicFootprintValid *= reprojectionTapValid;
            }
        }

        // Check bilinear taps
        for (int i = 0; i < 4; ++i)
        {
            Int2 tapPos = bilinearOriginSMB + bilinearOffsetsSMB[i];
            float prevViewZInTap = Load2DFloat1(prevDepthBuffer, tapPos);
            float reprojectionTapValid = abs(prevViewZInTap - estimatedPrevDepth) > smbDisocclusionThreshold[i] ? 0.0f : 1.0f;
            bicubicFootprintValid *= reprojectionTapValid;
            bilinearTapsValidSMB[i] = reprojectionTapValid;
        }

        // Sample and check normal
        Float3 prevNormalFlat = normalize(SampleBicubicSmoothStep<Load2DFuncFloat4<Float3>, Float3, BoundaryFuncClamp>(
            prevNormalRoughnessBuffer, prevUVSMB, screenResolution));
        Float3 prevNormalFlatRotated = normalize(rotate(worldPrevToWorldRotation, prevNormalFlat).v);

        // Reject backfacing history
        float dotNormal = dot(normalize(currentNormalAveraged), prevNormalFlatRotated);
        if (dotNormal < 0.0f)
        {
            bilinearTapsValidSMB = Float4(0.0f);
            bicubicFootprintValid = 0.0f;
        }
        else
        {
            bool useBicubicSMB = (bicubicFootprintValid > 0);

            // Sample diffuse history
            if (useBicubicSMB)
            {
                prevDiffuseIlluminationAnd2ndMomentSMB = SampleBicubic12Taps<Load2DFuncFloat4<Float4>, Float4, BoundaryFuncClamp>(
                    prevDiffuseIllumBuffer, prevUVSMB, screenResolution);
            }
            else
            {
                prevDiffuseIlluminationAnd2ndMomentSMB = SampleBilinearCustomFloat4(
                    prevDiffuseIllumBuffer, prevUVSMB, screenResolution, bilinearTapsValidSMB);
            }

            if (useBicubicSMB)
            {
                prevDiffuseIlluminationAnd2ndMomentSMBResponsive = SampleBicubic12Taps<Load2DFuncFloat4<Float3>, Float3, BoundaryFuncClamp>(
                    prevDiffuseFastIllumBuffer, prevUVSMB, screenResolution);
            }
            else
            {
                prevDiffuseIlluminationAnd2ndMomentSMBResponsive = SampleBilinearCustomFloat4(
                                                                       prevDiffuseFastIllumBuffer, prevUVSMB, screenResolution, bilinearTapsValidSMB)
                                                                       .xyz;
            }

            prevDiffuseIlluminationAnd2ndMomentSMB = max4f(prevDiffuseIlluminationAnd2ndMomentSMB, Float4(0.0f));
            prevDiffuseIlluminationAnd2ndMomentSMBResponsive = max3f(prevDiffuseIlluminationAnd2ndMomentSMBResponsive, Float3(0.0f));

            // Sample specular history
            if (useBicubicSMB)
            {
                prevSpecularIlluminationAnd2ndMomentSMB = SampleBicubic12Taps<Load2DFuncFloat4<Float4>, Float4, BoundaryFuncClamp>(
                    prevSpecularIllumBuffer, prevUVSMB, screenResolution);
            }
            else
            {
                prevSpecularIlluminationAnd2ndMomentSMB = SampleBilinearCustomFloat4(
                    prevSpecularIllumBuffer, prevUVSMB, screenResolution, bilinearTapsValidSMB);
            }

            if (useBicubicSMB)
            {
                prevSpecularIlluminationAnd2ndMomentSMBResponsive = SampleBicubic12Taps<Load2DFuncFloat4<Float3>, Float3, BoundaryFuncClamp>(
                    prevSpecularFastIllumBuffer, prevUVSMB, screenResolution);
            }
            else
            {
                prevSpecularIlluminationAnd2ndMomentSMBResponsive = SampleBilinearCustomFloat4(
                                                                        prevSpecularFastIllumBuffer, prevUVSMB, screenResolution, bilinearTapsValidSMB)
                                                                        .xyz;
            }

            prevSpecularIlluminationAnd2ndMomentSMB = max4f(prevSpecularIlluminationAnd2ndMomentSMB, Float4(0.0f));
            prevSpecularIlluminationAnd2ndMomentSMBResponsive = max3f(prevSpecularIlluminationAnd2ndMomentSMBResponsive, Float3(0.0f));

            // Sample specular hit distance
            prevReflectionHitTSMB = SampleBilinearCustomFloat1(prevSpecularHitDistBuffer, prevUVSMB, screenResolution, bilinearTapsValidSMB);
            prevReflectionHitTSMB = max(0.001f, prevReflectionHitTSMB);

            SMBReprojectionFound = (bicubicFootprintValid > 0.0f) ? 2.0f : 1.0f;
            Float4 bilinearCustomWeights = GetBilinearWeight(prevUVSMB, screenResolution);
            footprintQuality = (bicubicFootprintValid > 0) ? 1.0f : dot(bilinearCustomWeights, Float4(1.0f));

            if (dot(bilinearTapsValidSMB, Float4(1.0f)) == 0.0f)
            {
                SMBReprojectionFound = 0.0f;
                footprintQuality = 0.0f;
                historyLength = 0.0f;
            }
            else
            {
                historyLength = SampleBilinearCustomFloat1(prevHistoryLengthBuffer, prevUVSMB, screenResolution, bilinearTapsValidSMB);
            }
        }
    }

    // History length is based on surface motion based disocclusion
    historyLength = historyLength + 1.0f;

    // Avoid footprint momentary stretching due to changed viewing angle
    Float3 Vprev = normalize(prevWorldPos - prevCamera.pos);
    float NoVprev = abs(dot(currentNormal, Vprev));
    float sizeQuality = (NoVprev + 1e-3f) / (NoV + 1e-3f); // this order because we need to fix stretching only, shrinking is OK
    sizeQuality *= sizeQuality;
    sizeQuality *= sizeQuality;
    footprintQuality *= lerp(0.1f, 1.0f, saturate(sizeQuality));

    // Minimize "getting stuck in history" effect when only fraction of bilinear footprint is valid
    // by shortening the history length
    if (footprintQuality < 1.0f)
    {
        historyLength *= sqrtf(footprintQuality);
        historyLength = max(historyLength, 1.0f);
    }

    // Handling history reset if needed
    historyLength = relaxParams.resetHistory ? 1.0f : historyLength;

    // Limiting history length based on max values for diffuse and specular
    float maxAccumulatedFrameNum = 1.0f + max(diffuseMaxAccumulatedFrameNum, specularMaxAccumulatedFrameNum);
    historyLength = min(historyLength, maxAccumulatedFrameNum);

    // Temporal accumulation of diffuse illumination with history confidence support
    float diffHistoryLength = historyLength;
    float diffMaxAccumulatedFrameNumAdj = diffuseMaxAccumulatedFrameNum;
    float diffMaxFastAccumulatedFrameNumAdj = diffuseMaxFastAccumulatedFrameNum;

    if (relaxParams.hasHistoryConfidence)
    {
        float inDiffConfidence = Load2DFloat1(diffuseHistoryConfidenceBuffer, pixelPos);
        diffMaxAccumulatedFrameNumAdj *= inDiffConfidence;
        diffMaxFastAccumulatedFrameNumAdj *= inDiffConfidence;
    }

    float diffuseAlpha = (SMBReprojectionFound > 0) ? max(1.0f / (diffMaxAccumulatedFrameNumAdj + 1.0f), 1.0f / diffHistoryLength) : 1.0f;
    float diffuseAlphaResponsive = (SMBReprojectionFound > 0) ? max(1.0f / (diffMaxFastAccumulatedFrameNumAdj + 1.0f), 1.0f / diffHistoryLength) : 1.0f;

    Float4 accumulatedDiffuseIlluminationAnd2ndMoment = lerp(prevDiffuseIlluminationAnd2ndMomentSMB, Float4(diffuseIllumination, diffuse2ndMoment), diffuseAlpha);
    Float3 accumulatedDiffuseIlluminationResponsive = lerp(prevDiffuseIlluminationAnd2ndMomentSMBResponsive, diffuseIllumination, diffuseAlphaResponsive);

    // Store diffuse results
    Store2DFloat4(accumulatedDiffuseIlluminationAnd2ndMoment, diffuseIlluminationPingBuffer, pixelPos);
    Store2DFloat4(Float4(accumulatedDiffuseIlluminationResponsive, 0), diffuseIlluminationPongBuffer, pixelPos);
    Store2DFloat1(historyLength, historyLengthBuffer, pixelPos);

    // === SPECULAR TEMPORAL ACCUMULATION ===

    float specHistoryLength = historyLength;
    float specMaxAccumulatedFrameNumAdj = specularMaxAccumulatedFrameNum;
    float specMaxFastAccumulatedFrameNumAdj = specularMaxFastAccumulatedFrameNum;

    if (relaxParams.hasHistoryConfidence)
    {
        float inSpecConfidence = Load2DFloat1(specularHistoryConfidenceBuffer, pixelPos);
        specMaxAccumulatedFrameNumAdj *= inSpecConfidence;
        specMaxFastAccumulatedFrameNumAdj *= inSpecConfidence;
    }

    float specHistoryFrames = min(specMaxAccumulatedFrameNumAdj, specHistoryLength);
    float specHistoryResponsiveFrames = min(specMaxFastAccumulatedFrameNumAdj, specHistoryLength);

    // Calculate curvature (simplified version)
    float curvature = 0.0f; // Can implement full curvature calculation later

    // Thin lens equation for adjusting reflection HitT
    float hitDistFocused = ApplyThinLensEquation(currentLinearZ, curvature);

    // Loading specular data based on virtual motion
    Float4 prevSpecularIlluminationAnd2ndMomentVMB;
    Float3 prevSpecularIlluminationAnd2ndMomentVMBResponsive;
    Float3 prevNormalVMB;
    Float2 prevUVVMB;
    float prevRoughnessVMB;
    float prevReflectionHitTVMB;

    // Virtual motion based reprojection - inline implementation
    float VMBReprojectionFound = 0.0f;
    {
        // Calculate virtual motion based on NRD RELAX approach
        // For specular, the virtual position is along the reflection direction
        Float3 reflectionDir = reflect3f(-V, currentNormal);
        Float3 virtualWorldPos = currentWorldPos + reflectionDir * hitDistFocused;

        // Transform virtual world position to previous frame
        Float3 cameraDelta = prevCamera.pos - camera.pos;
        Float3 prevVirtualWorldPos = virtualWorldPos + cameraDelta;

        // Project virtual position to previous frame screen space
        prevUVVMB = prevCamera.worldDirectionToUV(normalize(prevVirtualWorldPos - prevCamera.pos));

        // Check if the virtual UV is within screen bounds
        if (prevUVVMB.x < 0.0f || prevUVVMB.x > 1.0f || prevUVVMB.y < 0.0f || prevUVVMB.y > 1.0f)
        {
            prevSpecularIlluminationAnd2ndMomentVMB = Float4(0.0f);
            prevSpecularIlluminationAnd2ndMomentVMBResponsive = Float3(0.0f);
            prevNormalVMB = currentNormal;
            prevRoughnessVMB = 0.0f;
            prevReflectionHitTVMB = currentLinearZ;
        }
        else
        {
            Float2 prevVirtualPixelPosFloat = prevUVVMB * Float2(screenResolution.x, screenResolution.y);

            // Calculating footprint origin and weights for bilinear sampling
            Float2 bilinearOriginF = floor(prevVirtualPixelPosFloat - 0.5f);
            Int2 bilinearOriginVMB = Int2(bilinearOriginF.x, bilinearOriginF.y);

            // Calculate estimated depth at virtual position for disocclusion checking
            float estimatedVirtualDepth = length(prevVirtualWorldPos - prevCamera.pos);

            // Calculating disocclusion threshold for virtual motion
            Float4 vmbDisocclusionThreshold = Float4(finalDisocclusionThreshold * currentLinearZ);
            vmbDisocclusionThreshold *= IsInScreenBilinear(bilinearOriginVMB, screenResolution);
            vmbDisocclusionThreshold -= 1e-6f;

            // Checking bilinear footprint validity for virtual motion based specular reprojection
            Int2 bilinearOffsetsVMB[4] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
            Float4 bilinearTapsValidVMB = Float4(0.0f);

            for (int i = 0; i < 4; ++i)
            {
                Int2 tapPos = bilinearOriginVMB + bilinearOffsetsVMB[i];

                // Check screen bounds
                if (tapPos.x < 0 || tapPos.y < 0 || tapPos.x >= screenResolution.x || tapPos.y >= screenResolution.y)
                {
                    bilinearTapsValidVMB[i] = 0.0f;
                    continue;
                }

                float prevViewZInTap = Load2DFloat1(prevDepthBuffer, tapPos);

                // Use plane-based disocclusion test
                float depthDiff = abs(prevViewZInTap - estimatedVirtualDepth);
                bilinearTapsValidVMB[i] = depthDiff < vmbDisocclusionThreshold[i] ? 1.0f : 0.0f;

                // Additional material ID consistency check
                if (bilinearTapsValidVMB[i] > 0.0f)
                {
                    float prevMaterialID = Load2DFloat1(prevMatBuffer, tapPos);
                    if (abs(prevMaterialID - currentMaterialID) > 0.1f)
                    {
                        bilinearTapsValidVMB[i] = 0.0f;
                    }
                }
            }

            // Initialize outputs
            prevSpecularIlluminationAnd2ndMomentVMB = Float4(0.0f);
            prevSpecularIlluminationAnd2ndMomentVMBResponsive = Float3(0.0f);
            prevNormalVMB = currentNormal;
            prevRoughnessVMB = 0.0f;
            prevReflectionHitTVMB = currentLinearZ;

            // Sample previous data if any taps are valid
            if (dot(bilinearTapsValidVMB, Float4(1.0f)) > 0.0f)
            {
                // Use bicubic sampling if surface motion was bicubic and all taps are valid
                bool useBicubicVMB = (SMBReprojectionFound == 2.0f) && (dot(bilinearTapsValidVMB, Float4(1.0f)) == 4.0f);

                // Sample specular illumination and variance
                if (useBicubicVMB)
                {
                    prevSpecularIlluminationAnd2ndMomentVMB = SampleBicubic12Taps<Load2DFuncFloat4<Float4>, Float4, BoundaryFuncClamp>(
                        prevSpecularIllumBuffer, prevUVVMB, Float2(screenResolution.x, screenResolution.y));
                }
                else
                {
                    prevSpecularIlluminationAnd2ndMomentVMB = SampleBilinearCustomFloat4(
                        prevSpecularIllumBuffer, prevUVVMB, Float2(screenResolution.x, screenResolution.y), bilinearTapsValidVMB);
                }
                prevSpecularIlluminationAnd2ndMomentVMB = max4f(prevSpecularIlluminationAnd2ndMomentVMB, Float4(0.0f));

                // Sample fast/responsive specular illumination
                if (useBicubicVMB)
                {
                    prevSpecularIlluminationAnd2ndMomentVMBResponsive = SampleBicubic12Taps<Load2DFuncFloat4<Float3>, Float3, BoundaryFuncClamp>(
                        prevSpecularFastIllumBuffer, prevUVVMB, Float2(screenResolution.x, screenResolution.y));
                }
                else
                {
                    prevSpecularIlluminationAnd2ndMomentVMBResponsive = SampleBilinearCustomFloat4(
                                                                            prevSpecularFastIllumBuffer, prevUVVMB, Float2(screenResolution.x, screenResolution.y), bilinearTapsValidVMB)
                                                                            .xyz;
                }
                prevSpecularIlluminationAnd2ndMomentVMBResponsive = max3f(prevSpecularIlluminationAnd2ndMomentVMBResponsive, Float3(0.0f));

                // Sample previous hit distance
                prevReflectionHitTVMB = SampleBilinearCustomFloat1(
                    prevSpecularHitDistBuffer, prevUVVMB, Float2(screenResolution.x, screenResolution.y), bilinearTapsValidVMB);
                prevReflectionHitTVMB = max(0.001f, prevReflectionHitTVMB);

                // Sample previous normal and roughness, transforming normal to current frame
                Float4 prevNormalRoughnessVMB = SampleBilinearCustomFloat4(
                    prevNormalRoughnessBuffer, prevUVVMB, Float2(screenResolution.x, screenResolution.y), bilinearTapsValidVMB);
                prevNormalVMB = normalize(rotate(worldPrevToWorldRotation, prevNormalRoughnessVMB.xyz).v);
                prevRoughnessVMB = prevNormalRoughnessVMB.w;
            }

            // Return success only if all 4 taps are valid
            VMBReprojectionFound = (dot(bilinearTapsValidVMB, Float4(1.0f)) == 4.0f) ? 1.0f : 0.0f;
        }
    }

    // Calculate UV difference for confidence calculations
    Float2 uvDiff = prevUVVMB - prevUVSMB;
    float curvatureAngle = 0.0f; // Will be enhanced when full curvature is implemented

    // Virtual history confidence calculations
    float virtualHistoryAmount = ComputeVirtualHistoryAmount(
        currentNormal, prevNormalVMB, normalize(currentNormalAveraged),
        currentRoughness, prevRoughnessVMB,
        uvDiff, Float2(screenResolution.x, screenResolution.y), smbParallaxInPixelsMax,
        curvatureAngle, false // isOrthoMode - assume perspective for now
    );

    // Hit distance confidence
    float virtualHistoryHitDistConfidence = ComputeVirtualHistoryHitDistConfidence(
        specularIllumination.w, prevReflectionHitTVMB, currentLinearZ, curvature, currentRoughness);

    // Apply VMB reprojection success and specular dominant direction factor (like NRD RELAX)
    // The dominant direction factor accounts for roughness and viewing angle
    Float3 viewDir = normalize(camera.pos - currentWorldPos);
    float NoVDominant = abs(dot(currentNormal, viewDir));
    float dominantFactor = GetSpecularDominantDirectionWeight(currentNormal, viewDir, currentRoughness, NoVDominant);
    virtualHistoryAmount *= VMBReprojectionFound * dominantFactor * virtualHistoryHitDistConfidence;

    // "Looking back" validation to reduce lags
    if (VMBReprojectionFound > 0.0f && length(uvDiff) > 0.1f)
    {
        float lobeHalfAngle = max(atan(GetSpecLobeTanHalfAngle(currentRoughness)), RELAX_NORMAL_ULP);
        float lookingBackValidation = ComputeLookingBackValidation(
            prevUVVMB, uvDiff, screenResolution,
            prevNormalRoughnessBuffer, worldPrevToWorldRotation,
            prevNormalVMB, currentRoughness, lobeHalfAngle, curvatureAngle);
        virtualHistoryAmount *= lookingBackValidation;
    }

    // Surface motion based accumulation with enhanced confidence
    Float3 VprevSMB = normalize(prevWorldPos - prevCamera.pos);
    float lobeHalfAngle = max(atan(GetSpecLobeTanHalfAngle(currentRoughness)), RELAX_NORMAL_ULP);
    float specSMBConfidence = (SMBReprojectionFound > 0.0f ? 1.0f : 0.0f) *
                              GetEncodingAwareNormalWeight(V, VprevSMB, lobeHalfAngle * NoV / max(relaxParams.framerateScale, 1.0f), 0.0f, 0.0f, false);

    float specSMBAlpha = 1.0f - specSMBConfidence;
    float specSMBResponsiveAlpha = 1.0f - specSMBConfidence;
    specSMBAlpha = max(specSMBAlpha, 1.0f / (1.0f + specHistoryFrames));
    specSMBResponsiveAlpha = max(specSMBResponsiveAlpha, 1.0f / (1.0f + specHistoryResponsiveFrames));

    Float4 accumulatedSpecularSMB;
    accumulatedSpecularSMB.xyz = lerp(prevSpecularIlluminationAnd2ndMomentSMB.xyz, specularIllumination.xyz, specSMBAlpha);
    accumulatedSpecularSMB.w = lerp(prevReflectionHitTSMB, specularIllumination.w, max(specSMBAlpha, 0.1f));
    float accumulatedSpecularM2SMB = lerp(prevSpecularIlluminationAnd2ndMomentSMB.w, specular2ndMoment, specSMBAlpha);
    Float3 accumulatedSpecularSMBResponsive = lerp(prevSpecularIlluminationAnd2ndMomentSMBResponsive, specularIllumination.xyz, specSMBResponsiveAlpha);

    // Virtual motion based accumulation with enhanced confidence
    // Recompute virtualRoughnessWeight for confidence calculation (matching NRD approach)
    Float2 relaxedRoughnessWeightParams = GetRelaxedRoughnessWeightParams(currentRoughness * currentRoughness, relaxParams.roughnessFraction);
    float virtualRoughnessWeight = ComputeWeight(prevRoughnessVMB * prevRoughnessVMB, relaxedRoughnessWeightParams.x, relaxedRoughnessWeightParams.y);
    float uvDiffLengthInPixels = length(uvDiff * Float2(screenResolution.x, screenResolution.y));
    virtualRoughnessWeight = lerp(1.0f - saturate(uvDiffLengthInPixels), 1.0f, virtualRoughnessWeight);
    float specVMBConfidence = virtualRoughnessWeight * 0.9f + 0.1f;
    float specVMBAlpha = 1.0f - specVMBConfidence;
    float specVMBResponsiveAlpha = specVMBAlpha;
    float specVMBHitTAlpha = specVMBResponsiveAlpha;

    specVMBAlpha = max(specVMBAlpha, 1.0f / (1.0f + specHistoryFrames));
    specVMBResponsiveAlpha = max(specVMBResponsiveAlpha, 1.0f / (1.0f + specHistoryResponsiveFrames));
    specVMBHitTAlpha = max(specVMBHitTAlpha, 1.0f / (1.0f + specHistoryFrames));

    Float4 accumulatedSpecularVMB;
    accumulatedSpecularVMB.xyz = lerp(prevSpecularIlluminationAnd2ndMomentVMB.xyz, specularIllumination.xyz, specVMBAlpha);
    accumulatedSpecularVMB.w = lerp(prevReflectionHitTVMB, specularIllumination.w, max(specVMBHitTAlpha, 0.1f));
    float accumulatedSpecularM2VMB = lerp(prevSpecularIlluminationAnd2ndMomentVMB.w, specular2ndMoment, specVMBAlpha);
    Float3 accumulatedSpecularVMBResponsive = lerp(prevSpecularIlluminationAnd2ndMomentVMBResponsive, specularIllumination.xyz, specVMBResponsiveAlpha);

    // Fallback to surface motion if virtual motion doesn't go well
    // For very smooth surfaces (low roughness), prefer virtual motion
    // For rough surfaces, surface motion becomes more reliable
    float virtualPreference = 1.0f - GetSpecMagicCurve(currentRoughness);
    virtualHistoryAmount *= lerp(1.0f, saturate(specVMBConfidence / (specSMBConfidence + 1e-6f)), virtualPreference);

    // Final temporal accumulation of specular data
    float accumulatedReflectionHitT = lerp(accumulatedSpecularSMB.w, accumulatedSpecularVMB.w, virtualHistoryAmount);
    Float3 accumulatedSpecularIllumination = lerp(accumulatedSpecularSMB.xyz, accumulatedSpecularVMB.xyz, virtualHistoryAmount);
    Float3 accumulatedSpecularIlluminationResponsive = lerp(accumulatedSpecularSMBResponsive, accumulatedSpecularVMBResponsive, virtualHistoryAmount);
    float accumulatedSpecular2ndMoment = lerp(accumulatedSpecularM2SMB, accumulatedSpecularM2VMB, virtualHistoryAmount);

    // If zero specular sample (color = 0), artificially adding variance for pixels with low reprojection confidence
    float specularHistoryConfidence = lerp(specSMBConfidence, specVMBConfidence, virtualHistoryAmount);
    if (accumulatedSpecular2ndMoment == 0.0f)
        accumulatedSpecular2ndMoment = relaxParams.specVarianceBoost * (1.0f - specularHistoryConfidence);

    // Store specular results - ensure hit distance is properly propagated
    Store2DFloat4(Float4(accumulatedSpecularIllumination, accumulatedSpecular2ndMoment), specularIlluminationPingBuffer, pixelPos);
    Store2DFloat4(Float4(accumulatedSpecularIlluminationResponsive, accumulatedReflectionHitT), specularIlluminationPongBuffer, pixelPos);
    Store2DFloat1(accumulatedReflectionHitT, specularHitDistBuffer, pixelPos);
    Store2DFloat1(specularHistoryConfidence, specularReprojectionConfidenceBuffer, pixelPos);
}