#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"

__device__ float getDiffuseNormalWeight(Float3 centerNormal, Float3 pointNormal)
{
    float historyFixEdgeStoppingNormalPower = 8.0f;
    return pow(max(0.01f, dot(centerNormal, pointNormal)), max(historyFixEdgeStoppingNormalPower, 0.01f));
}

__device__ float getRadius(float historyLength)
{
    constexpr float gHistoryFixFrameNum = 4.0f;
    // IMPORTANT: progression is "{8, 4, 2, 1} + 1". "+1" is important to better break blobs
    return exp2(gHistoryFixFrameNum - historyLength) + 1.0f;
}

// Main
__global__ void HistoryFix(
    Int2 screenResolution,
    Float2 invScreenResolution,

    SurfObj depthBuffer,
    SurfObj materialBuffer,
    SurfObj normalRoughnessBuffer,
    SurfObj historyLengthBuffer,
    SurfObj illuminationPingBuffer,

    SurfObj illuminationPongBuffer,

    Camera camera)
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

    if (pixelPos.x >= screenResolution.x || pixelPos.y >= screenResolution.y)
    {
        return;
    }

    // Early out if linearZ is beyond denoising range
    // Early out if no disocclusion detected
    float centerViewZ = Load2DFloat1(depthBuffer, pixelPos);
    float historyLength = Load2DFloat1(historyLengthBuffer, pixelPos);
    constexpr float historyFixFrameNum = 4.0f;
    const float denoisingRange = 500000.0f;
    if ((centerViewZ > denoisingRange) || (historyLength > historyFixFrameNum))
        return;

    // Loading center data
    float centerMaterialID = Load2DUshort1(materialBuffer, pixelPos);
    Float4 centerNormalRoughness = Load2DFloat4(normalRoughnessBuffer, pixelPos);
    Float3 centerNormal = centerNormalRoughness.xyz;
    //float centerRoughness = centerNormalRoughness.w;
    Float3 centerWorldPos = GetCurrentWorldPosFromPixelPos(camera, pixelPos, centerViewZ);
    constexpr float depthThresholdConstant = 0.003f;
    float depthThreshold = depthThresholdConstant * centerViewZ;

    Float4 diffuseIlluminationAnd2ndMomentSum = Load2DFloat4(illuminationPingBuffer, pixelPos);
    float diffuseWSum = 1.0f;

    // Running sparse cross-bilateral filter
    float r = getRadius(historyLength);
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
            Float3 sampleNormal = Load2DFloat4(normalRoughnessBuffer, samplePosInt).xyz;
            float sampleViewZ = Load2DFloat1(depthBuffer, samplePosInt);
            Float3 sampleWorldPos = GetCurrentWorldPosFromPixelPos(camera, samplePosInt, sampleViewZ);

            float geometryWeight = GetPlaneDistanceWeight_Atrous(
                centerWorldPos,
                centerNormal,
                sampleWorldPos,
                depthThreshold);

            // Summing up diffuse result
            float diffuseW = geometryWeight;
            diffuseW *= getDiffuseNormalWeight(centerNormal, sampleNormal);
            diffuseW = isInside ? diffuseW : 0;
            diffuseW *= (sampleMaterialID == centerMaterialID);

            if (diffuseW > 1e-4f)
            {
                Float4 sampleDiffuseIlluminationAnd2ndMoment = Load2DFloat4(illuminationPingBuffer, samplePosInt);

                diffuseIlluminationAnd2ndMomentSum += sampleDiffuseIlluminationAnd2ndMoment * diffuseW;

                diffuseWSum += diffuseW;
            }
        }
    }

    // Output buffers will hold the pixels with disocclusion processed by history fix.
    // The next shader will have to copy these areas to normal and responsive history buffers.

    Float4 outDiffuseIlluminationAnd2ndMoment = diffuseIlluminationAnd2ndMomentSum / diffuseWSum;
    Store2DFloat4(outDiffuseIlluminationAnd2ndMoment, illuminationPongBuffer, pixelPos);
}