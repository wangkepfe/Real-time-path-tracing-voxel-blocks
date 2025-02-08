#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"

__global__ void PrePass(
    Int2 screenResolution,
    Float2 invScreenResolution,

    TexObj illuminationPingTex,

    SurfObj materialBuffer,
    SurfObj normalRoughnessBuffer,
    SurfObj depthBuffer,
    SurfObj illuminationPingBuffer,

    SurfObj illuminationBuffer,

    Camera camera,

    int frameIndex)
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

    float centerViewZ = Load2DFloat1(depthBuffer, pixelPos);

    // Early out
    const float denoisingRange = 500000.0f;
    if (centerViewZ > denoisingRange)
        return;

    float centerMaterialID = Load2DFloat1(materialBuffer, pixelPos);
    Float4 centerNormalRoughness = Load2DFloat4(normalRoughnessBuffer, pixelPos);
    Float3 centerNormal = centerNormalRoughness.xyz;
    //float centerRoughness = centerNormalRoughness.w;
    Float3 centerWorldPos = GetCurrentWorldPosFromPixelPos(camera, pixelPos, centerViewZ);

    float angle1 = Weyl1D(0.5f, frameIndex) * radians(90.0f);
    Float4 rotator = GetRotator(angle1);

    Float2 pixelUv = Float2((float)pixelPos.x + 0.5f, (float)pixelPos.y + 0.5f) * invScreenResolution;

    //bool diffHasData = true;
    //Int2 diffPos = pixelPos;

    Float4 diffuseIllumination = Load2DFloat4(illuminationPingBuffer, pixelPos);

    float diffBlurRadius = 30.0f;

    float poissonSampleNum = 8;

    const Float3 g_Poisson8[8] =
        {
            Float3(-0.4706069f, -0.4427112f, +0.6461146f),
            Float3(-0.9057375f, +0.3003471f, +0.9542373f),
            Float3(-0.3487388f, +0.4037880f, +0.5335386f),
            Float3(+0.1023042f, +0.6439373f, +0.6520134f),
            Float3(+0.5699277f, +0.3513750f, +0.6695386f),
            Float3(+0.2939128f, -0.1131226f, +0.3149309f),
            Float3(+0.7836658f, -0.4208784f, +0.8895339f),
            Float3(+0.1564120f, -0.8198990f, +0.8346850f)};

    // Pre-blur for diffuse

    // Diffuse blur radius
    float pixelWorldSizeScaleToDepth = camera.getPixelWorldSizeScaleToDepth();
    float frustumSize = PixelRadiusToWorld(pixelWorldSizeScaleToDepth, min(screenResolution.x, screenResolution.y), centerViewZ);
    float hitDist = (diffuseIllumination.w == 0.0f ? 1.0f : diffuseIllumination.w);
    float hitDistFactor = GetHitDistFactor(hitDist, frustumSize); // NoD = 1
    float blurRadius = diffBlurRadius * hitDistFactor;

    if (diffuseIllumination.w == 0.0f)
        blurRadius = max(blurRadius, 1.0f);

    float lobeAngleFraction = 0.5f;

    float worldBlurRadius = PixelRadiusToWorld(pixelWorldSizeScaleToDepth, blurRadius, centerViewZ);
    float normalWeightParam = GetNormalWeightParam2(1.0f, 0.25f * lobeAngleFraction);
    Float2 hitDistanceWeightParams = GetHitDistanceWeightParams(diffuseIllumination.w, 1.0f / 9.0f);

    float weightSum = 1.0;
    float diffMinHitDistanceWeight = 0.2f;

    // Spatial blur
    for (unsigned int i = 0; i < poissonSampleNum; i++)
    {
        Float3 offset = g_Poisson8[i];

        // Sample coordinates
        Float2 uv = pixelUv * screenResolution + RotateVector(rotator, offset.xy) * blurRadius;
        uv = floor(uv) + 0.5f;
        Float2 samplePos = uv;
        uv = uv * invScreenResolution;
        Int2 samplePosInt = Int2(round(samplePos.x), round(samplePos.y));

        Float2 uvScaled = clamp2f(uv);

        // Fetch data
        float sampleMaterialID = Load2DFloat1(materialBuffer, samplePosInt);
        Float3 sampleNormal = Load2DFloat4(normalRoughnessBuffer, samplePosInt).xyz;
        float sampleViewZ = Load2DFloat1(depthBuffer, samplePosInt);
        Float3 sampleWorldPos = GetCurrentWorldPosFromPixelPos(camera, samplePosInt, sampleViewZ);

        // Sample weight
        float sampleWeight = IsInScreenNearest(uv);
        sampleWeight *= float(sampleViewZ < denoisingRange);
        sampleWeight *= (centerMaterialID == sampleMaterialID);

        float depthThreshold = 0.003f;

        sampleWeight *= GetPlaneDistanceWeight(
            centerWorldPos,
            centerNormal,
            centerViewZ,
            sampleWorldPos,
            depthThreshold);

        float angle = AcosApprox(dot(centerNormal, sampleNormal));
        sampleWeight *= ComputeNonExponentialWeight(angle, normalWeightParam, 0.0f);

        float4 t = tex2D<float4>(illuminationPingTex, uvScaled.x, uvScaled.y);
        Float4 sampleDiffuseIllumination = Float4(t.x, t.y, t.z, t.w);

        sampleDiffuseIllumination = Denanify(sampleWeight, sampleDiffuseIllumination);

        constexpr float expWeightScale = 3.0f;
        sampleWeight *= lerp(diffMinHitDistanceWeight, 1.0f, ComputeExponentialWeight(sampleDiffuseIllumination.w, hitDistanceWeightParams.x, hitDistanceWeightParams.y, expWeightScale));
        sampleWeight *= GetGaussianWeight(offset.z);

        // Accumulate
        weightSum += sampleWeight;

        diffuseIllumination += sampleDiffuseIllumination * sampleWeight;
    }

    diffuseIllumination /= weightSum;

    Store2DFloat4(diffuseIllumination, illuminationBuffer, pixelPos);
}