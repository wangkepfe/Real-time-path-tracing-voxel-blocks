#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"
#include "shaders/ShaderDebugUtils.h"
#include "core/GlobalSettings.h"

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

// Returns reprojected data from previous frame calculated using filtering based on filters above.
// Returns reprojection search result based on surface motion:
// 2 - reprojection found, bicubic footprint was used
// 1 - reprojection found, bilinear footprint was used
// 0 - reprojection not found

__device__ float loadSurfaceMotionBasedPrevData(
    Int2 screenResolution,
    Float2 invScreenResolution,

    SurfObj prevDepthBuffer,
    SurfObj prevMatBuffer,
    SurfObj prevIllumBuffer,
    SurfObj prevFastIllumBuffer,
    SurfObj prevHistoryLengthBuffer,
    SurfObj prevNormalRoughnessBuffer,

    float pixelWorldSizeScaleToDepth,
    Quat worldPrevToWorldRotation,
    Camera prevCamera,

    Float3 prevWorldPos,
    Float2 prevUVSMB,
    float currentLinearZ,
    Float3 currentNormal,
    float NoV,
    float smbParallaxInPixelsMax,
    float currentMaterialID,
    float disocclusionThreshold,

    float &footprintQuality,
    float &historyLength,
    Float4 &prevDiffuseIllumAnd2ndMoment,
    Float3 &prevDiffuseResponsiveIllum)
{
    float estimatedPrevDepth = length(prevWorldPos - prevCamera.pos);

    // if (CUDA_CENTER_PIXEL())
    // {
    //     DEBUG_PRINT(prevWorldPos);
    //     DEBUG_PRINT(prevCamera.pos);
    // }

    // Calculating previous pixel position
    Float2 prevPixelPosFloat = prevUVSMB * screenResolution;

    // Calculating footprint origin and weights
    Float2 bilinearOriginF = floor(prevPixelPosFloat - 0.5f);
    Int2 bilinearOrigin = Int2(bilinearOriginF.x, bilinearOriginF.y);
    Float2 bilinearWeights = fract(prevPixelPosFloat - 0.5f);

    // if (CUDA_PIXEL(0.5f, 0.75f))
    // {
    //     DEBUG_PRINT(bilinearOrigin);
    // }

    // Calculating disocclusion threshold
    float pixelSize = pixelWorldSizeScaleToDepth * currentLinearZ;
    float frustumSize = pixelSize * min(screenResolution.x, screenResolution.y);
    float disocclusionThresholdSlopeScale = 1.0 / lerp(lerp(0.05f, 1.0f, NoV), 1.0f, saturate(smbParallaxInPixelsMax / 30.0f));
    Float4 smbDisocclusionThreshold = Float4(saturate(disocclusionThreshold * disocclusionThresholdSlopeScale) * frustumSize);
    smbDisocclusionThreshold *= IsInScreenBilinear(bilinearOrigin, screenResolution);
    smbDisocclusionThreshold -= 1e-6f;

    // Checking bicubic footprint (with cut corners)
    // remembering bilinear taps validity and worldspace position along the way,
    // for faster weighted bilinear and for calculating previous worldspace position
    // bc - bicubic tap,
    // bl - bicubic & bilinear tap
    //
    // -- bc bc --
    // bc bl bl bc
    // bc bl bl bc
    // -- bc bc --
    Int2 bicubicOffsets[4][2] = {
        {
            {0, -1},
            {-1, 0},
        },
        {
            {1, -1},
            {2, 0},
        },
        {
            {-1, 1},
            {0, 2},
        },
        {{2, 1},
         {1, 2}}};

    Int2 bilinearOffsets[4] = {
        {0, 0},
        {1, 0},
        {0, 1},
        {1, 1}};

    float bicubicFootprintValid = 1.0f;
    Float4 bilinearTapsValid = Float4(0.0f);

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            Int2 tapPos = bilinearOrigin + bicubicOffsets[i][j];
            float prevViewZInTap = Load2DFloat1(prevDepthBuffer, tapPos);
            float reprojectionTapValid = abs(prevViewZInTap - estimatedPrevDepth) > smbDisocclusionThreshold[i] ? 0.0f : 1.0f;
            bicubicFootprintValid *= reprojectionTapValid;
        }
    }

    for (int i = 0; i < 4; ++i)
    {
        Int2 tapPos = bilinearOrigin + bilinearOffsets[i];
        float prevViewZInTap = Load2DFloat1(prevDepthBuffer, tapPos);
        float reprojectionTapValid = abs(prevViewZInTap - estimatedPrevDepth) > smbDisocclusionThreshold[i] ? 0.0f : 1.0f;
        bicubicFootprintValid *= reprojectionTapValid;
        bilinearTapsValid[i] = reprojectionTapValid;

        // if (CUDA_PIXEL(0.5f, 0.1f))
        // {
        //     DEBUG_PRINT(tapPos);
        //     DEBUG_PRINT(prevWorldPos);
        //     DEBUG_PRINT(Float4(prevViewZInTap, estimatedPrevDepth, smbDisocclusionThreshold[i], reprojectionTapValid));
        // }
    }

    // 4 normal samples
    // Float3 prevNormalFlat = Load2DFloat4(prevNormalRoughnessBuffer, bilinearOrigin).xyz;
    Float3 prevNormalFlat = normalize(SampleBicubicSmoothStep<Load2DFuncFloat4<Float3>, Float3, BoundaryFuncClamp>(prevNormalRoughnessBuffer, prevUVSMB, screenResolution));
    Float3 prevNormalFlatRotated = normalize(rotate(worldPrevToWorldRotation, prevNormalFlat).v);

    // Reject backfacing history: if angle between current normal and previous normal is larger than 90 deg
    float dotNormal = dot(currentNormal, prevNormalFlatRotated);
    // if (CUDA_PIXEL(0.5f, 0.5f))
    // {
    //     DEBUG_PRINT(dotNormal);
    //     DEBUG_PRINT(currentNormal);
    //     DEBUG_PRINT(prevNormalFlat);
    //     DEBUG_PRINT(prevNormalFlatRotated);
    // }
    if (dotNormal < 0.0f)
    {
        bilinearTapsValid = Float4(0.0f);
        bicubicFootprintValid = 0.0f;
    }

    bool useBicubic = (bicubicFootprintValid > 0);

    if (useBicubic)
    {
        prevDiffuseIllumAnd2ndMoment = SampleBicubic12Taps<Load2DFuncFloat4<Float4>, Float4, BoundaryFuncClamp>(prevIllumBuffer, prevUVSMB, screenResolution);
    }
    else
    {
        prevDiffuseIllumAnd2ndMoment = SampleBilinearCustomFloat4(prevIllumBuffer, prevUVSMB, screenResolution, bilinearTapsValid);
    }

    if (useBicubic)
    {
        prevDiffuseResponsiveIllum = SampleBicubic12Taps<Load2DFuncFloat4<Float3>, Float3, BoundaryFuncClamp>(prevFastIllumBuffer, prevUVSMB, screenResolution);
    }
    else
    {
        prevDiffuseResponsiveIllum = SampleBilinearCustomFloat4(prevFastIllumBuffer, prevUVSMB, screenResolution, bilinearTapsValid).xyz;
    }

    prevDiffuseIllumAnd2ndMoment = max4f(prevDiffuseIllumAnd2ndMoment, Float4(0.0f));
    prevDiffuseResponsiveIllum = max3f(prevDiffuseResponsiveIllum, Float3(0.0f));

    float reprojectionFound = (bicubicFootprintValid > 0.0f) ? 2.0f : 1.0f;
    Float4 bilinearCustomWeights = GetBilinearWeight(prevUVSMB, screenResolution);
    footprintQuality = (bicubicFootprintValid > 0) ? 1.0f : dot(bilinearCustomWeights, Float4(1.0f));

    if (dot(bilinearTapsValid, Float4(1.0f)) == 0.0f)
    {
        reprojectionFound = 0.0f;
        footprintQuality = 0.0f;
        historyLength = 0.0f;
    }
    else
    {
        // historyLength = Load2DFloat1(prevHistoryLengthBuffer, bilinearOrigin);
        historyLength = SampleBilinearCustomFloat1(prevHistoryLengthBuffer, prevUVSMB, screenResolution, bilinearTapsValid);
    }

    // if (CUDA_PIXEL(0.5f, 0.5f))
    // {
    //     DEBUG_PRINT(bilinearTapsValid);
    //     DEBUG_PRINT(historyLength);
    // }

    return reprojectionFound;
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
    SurfObj prevIllumBuffer,
    SurfObj prevFastIllumBuffer,
    SurfObj prevHistoryLengthBuffer,
    SurfObj prevNormalRoughnessBuffer,
    SurfObj motionVectorBuffer,

    SurfObj illuBuffer,
    SurfObj depthBuffer,
    SurfObj normalRoughnessBuffer,
    SurfObj materialBuffer,

    SurfObj illuminationPingBuffer,
    SurfObj illuminationPongBuffer,
    SurfObj historyLengthBuffer,

    SurfObj debugBuffer,

    Camera camera,
    Camera prevCamera,

    // Denoising parameters
    float denoisingRange,
    float disocclusionThreshold,
    float disocclusionThresholdAlternate,
    float maxAccumulatedFrameNum,
    float maxFastAccumulatedFrameNum)
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
            Preload(newId, groupBase + newId, screenResolution, normalRoughnessBuffer, sharedNormalRoughness, bufferSize);
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

    // Reproject using world-space motion vectors
    Float4 motionSample = Load2DFloat4(motionVectorBuffer, pixelPos);
    Float3 motionWS = motionSample.xyz;

    Float3 prevWorldPos = currentWorldPos + motionWS;
    Float2 prevUVSMB = prevCamera.worldDirectionToUV(normalize(prevWorldPos - prevCamera.pos));

    // Input noisy data
    Float3 diffuseIllumination = Load2DFloat4(illuBuffer, pixelPos).xyz;

    // Calculating average normal, minHitDist and specular sigma
    Float3 currentNormalAveraged = currentNormal;
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            // Skipping center pixel
            if ((i == 0) && (j == 0))
                continue;

            currentNormalAveraged += sharedNormalRoughness[(sharedMemoryIndex.y + j) * bufferSize + (sharedMemoryIndex.x + i)].xyz;
        }
    }
    currentNormalAveraged /= 9.0f;

    float diffuse1stMoment = luminance(diffuseIllumination);
    float diffuse2ndMoment = diffuse1stMoment * diffuse1stMoment;

    // Calculating surface parallax
    Float3 cameraDelta = prevCamera.pos - camera.pos;
    float smbParallaxInPixels1 = ComputeParallaxInPixels(prevWorldPos + cameraDelta, pixelUv, prevCamera, Float2(screenResolution.x, screenResolution.y));
    float smbParallaxInPixels2 = ComputeParallaxInPixels(prevWorldPos - cameraDelta, prevUVSMB, camera, Float2(screenResolution.x, screenResolution.y));

    float smbParallaxInPixelsMax = max(smbParallaxInPixels1, smbParallaxInPixels2);
    float smbParallaxInPixelsMin = min(smbParallaxInPixels1, smbParallaxInPixels2);

    // Calculating disocclusion threshold using passed parameters
    float disocclusionThresholdMix = 0.0f;

    float disocclusionThresholdBonus = disocclusionThreshold + (1.5f / screenResolution.y);
    float disocclusionThresholdAlternateBonus = disocclusionThresholdAlternate + (1.5f / screenResolution.y);

    float finalDisocclusionThreshold = lerp(disocclusionThresholdBonus, disocclusionThresholdAlternateBonus, disocclusionThresholdMix);

    // Loading previous data based on surface motion vectors
    float footprintQuality;

    Float4 prevDiffuseIlluminationAnd2ndMomentSMB;
    Float3 prevDiffuseIlluminationAnd2ndMomentSMBResponsive;

    float historyLength;

    Quat worldPrevToWorldRotation = rotationBetween(prevCamera.dir, camera.dir);

    float SMBReprojectionFound = loadSurfaceMotionBasedPrevData(
        screenResolution,
        invScreenResolution,

        prevDepthBuffer,
        prevMatBuffer,
        prevIllumBuffer,
        prevFastIllumBuffer,
        prevHistoryLengthBuffer,
        prevNormalRoughnessBuffer,

        camera.getPixelWorldSizeScaleToDepth(),
        worldPrevToWorldRotation,
        prevCamera,

        prevWorldPos,
        prevUVSMB,
        currentLinearZ,
        normalize(currentNormalAveraged),
        NoV,
        smbParallaxInPixelsMax,
        currentMaterialID,
        finalDisocclusionThreshold,

        footprintQuality,
        historyLength,
        prevDiffuseIlluminationAnd2ndMomentSMB,
        prevDiffuseIlluminationAnd2ndMomentSMBResponsive

    );

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

    // Limiting history length: HistoryFix must be invoked if history length <= gHistoryFixFrameNum
    // Use parameters passed to kernel instead of hardcoded values
    float diffMaxAccumulatedFrameNum = maxAccumulatedFrameNum;

    // Temporal accumulation of diffuse illumination
    float diffMaxFastAccumulatedFrameNum = maxFastAccumulatedFrameNum;

    historyLength = min(historyLength, diffMaxAccumulatedFrameNum);

    float diffHistoryLength = historyLength;

    float diffuseAlpha = (SMBReprojectionFound > 0) ? max(1.0f / (diffMaxAccumulatedFrameNum + 1.0f), 1.0f / diffHistoryLength) : 1.0f;
    float diffuseAlphaResponsive = (SMBReprojectionFound > 0) ? max(1.0f / (diffMaxFastAccumulatedFrameNum + 1.0f), 1.0f / diffHistoryLength) : 1.0f;

    Float4 accumulatedDiffuseIlluminationAnd2ndMoment = lerp(prevDiffuseIlluminationAnd2ndMomentSMB, Float4(diffuseIllumination, diffuse2ndMoment), diffuseAlpha);
    Float3 accumulatedDiffuseIlluminationResponsive = lerp(prevDiffuseIlluminationAnd2ndMomentSMBResponsive, diffuseIllumination, diffuseAlphaResponsive);

    // if (CUDA_PIXEL(0.5f, 0.5f))
    // {
    //     DEBUG_PRINT(historyLength);
    // }

    Store2DFloat4(accumulatedDiffuseIlluminationAnd2ndMoment, illuminationPingBuffer, pixelPos);
    Store2DFloat4(Float4(accumulatedDiffuseIlluminationResponsive, 0), illuminationPongBuffer, pixelPos);
    Store2DFloat1(historyLength, historyLengthBuffer, pixelPos);

    // Store2DFloat4(Float4(SMBReprojectionFound / 2.0f), debugBuffer, pixelPos);
    // Store2DFloat4(Float4(historyLength / 30.0f), debugBuffer, pixelPos);
}