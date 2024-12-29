#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"

namespace jazzfusion
{
    __device__ void Preload(
        Int2 sharedPos,
        Int2 globalPos,
        Int2 screenResolution,
        SurfObj illuminationBuffer,
        SurfObj illuminationPongBuffer,
        Float4 *s_DiffResponsiveYCoCg,
        Float4 *s_DiffNoisy_M2,
        int bufferSize)
    {
        globalPos = clamp2i(globalPos, Int2(0), Int2(screenResolution.x - 1, screenResolution.y - 1));

        Float4 diffuseResponsive = Load2DFloat4(illuminationPongBuffer, globalPos);
        Float4 diffuseResponsiveYcocg = RgbToYCoCg(diffuseResponsive);
        s_DiffResponsiveYCoCg[sharedPos.y * bufferSize + sharedPos.x] = diffuseResponsiveYcocg;

        Float4 diffuseNoisy = Load2DFloat4(illuminationBuffer, globalPos);
        float diffuseNoisyLuminance = luminance(diffuseNoisy.xyz);
        s_DiffNoisy_M2[sharedPos.y * bufferSize + sharedPos.x] = Float4(diffuseNoisy.xyz, diffuseNoisyLuminance * diffuseNoisyLuminance);
    }

    template <int groupSize, int border>
    __global__ void HistoryClamping(
        Int2 screenResolution,
        Float2 invScreenResolution,

        SurfObj depthBuffer,

        SurfObj illuminationBuffer,
        SurfObj illuminationPingBuffer,
        SurfObj illuminationPongBuffer,
        SurfObj historyLengthBuffer,

        SurfObj prevIlluminationBuffer,
        SurfObj prevFastIlluminationBuffer,
        SurfObj prevHistoryLengthBuffer)
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

        __shared__ Float4 s_DiffNoisy_M2[bufferSize * bufferSize];
        __shared__ Float4 s_DiffResponsiveYCoCg[bufferSize * bufferSize];

        Float2 pixelUv = Float2(Float2(pixelPos.x, pixelPos.y) + 0.5f) * invScreenResolution;

        Int2 groupBase = pixelPos - threadPos - border;
        unsigned int stageNum = (bufferSize * bufferSize + groupSize * groupSize - 1) / (groupSize * groupSize);
        for (unsigned int stage = 0; stage < stageNum; stage++)
        {
            unsigned int virtualIndex = threadIndex + stage * groupSize * groupSize;
            Int2 newId = Int2(virtualIndex % bufferSize, virtualIndex / bufferSize);
            if (stage == 0 || virtualIndex < bufferSize * bufferSize)
                Preload(newId, groupBase + newId, screenResolution, illuminationBuffer, illuminationPongBuffer, s_DiffResponsiveYCoCg, s_DiffNoisy_M2, bufferSize);
        }
        __syncthreads();

        if (pixelPos.x >= screenResolution.x || pixelPos.y >= screenResolution.y)
        {
            return;
        }

        float centerViewZ = Load2DFloat1(depthBuffer, pixelPos);

        // Early out
        const float denoisingRange = 500000.0f;
        if (centerViewZ > denoisingRange)
            return;

        // Reading history length
        float historyLength = Load2DFloat1(historyLengthBuffer, pixelPos);

        // Reading normal history
        Float3 diffuseResponsiveFirstMomentYCoCg = Float3(0.0f);
        Float3 diffuseResponsiveSecondMomentYCoCg = Float3(0.0f);
        Float3 diffuseNoisyFirstMoment = Float3(0.0f);
        float diffuseNoisySecondMoment = 0.0f;

        // Running history clamping
        Int2 sharedMemoryIndex = threadPos + Int2(border, border);
        for (int dx = -2; dx <= 2; dx++)
        {
            for (int dy = -2; dy <= 2; dy++)
            {
                Int2 sharedMemoryIndexP = sharedMemoryIndex + Int2(dx, dy);

                Float3 diffuseSampleYCoCg = s_DiffResponsiveYCoCg[sharedMemoryIndexP.y * bufferSize + sharedMemoryIndexP.x].xyz;
                diffuseResponsiveFirstMomentYCoCg += diffuseSampleYCoCg;
                diffuseResponsiveSecondMomentYCoCg += diffuseSampleYCoCg * diffuseSampleYCoCg;

                Float4 diffuseNoisySample = s_DiffNoisy_M2[sharedMemoryIndexP.y * bufferSize + sharedMemoryIndexP.x];
                diffuseNoisyFirstMoment += diffuseNoisySample.xyz;
                diffuseNoisySecondMoment += diffuseNoisySample.w;
            }
        }

        // Calculating color box
        float historyClampingColorBoxSigmaScale = 2.0f;
        diffuseResponsiveFirstMomentYCoCg /= 25.0f;
        diffuseResponsiveSecondMomentYCoCg /= 25.0f;
        diffuseNoisyFirstMoment /= 25.0f;
        diffuseNoisySecondMoment /= 25.0f;
        Float3 diffuseResponsiveSigmaYCoCg = sqrt3f(max3f(Float3(0.0f), diffuseResponsiveSecondMomentYCoCg - diffuseResponsiveFirstMomentYCoCg * diffuseResponsiveFirstMomentYCoCg));
        Float3 diffuseResponsiveColorMinYCoCg = diffuseResponsiveFirstMomentYCoCg - historyClampingColorBoxSigmaScale * diffuseResponsiveSigmaYCoCg;
        Float3 diffuseResponsiveColorMaxYCoCg = diffuseResponsiveFirstMomentYCoCg + historyClampingColorBoxSigmaScale * diffuseResponsiveSigmaYCoCg;

        // Expanding color box with color of the center pixel to minimize introduced bias
        Float4 diffuseResponsiveCenterYCoCg = s_DiffResponsiveYCoCg[sharedMemoryIndex.y * bufferSize + sharedMemoryIndex.x];
        diffuseResponsiveColorMinYCoCg = min(diffuseResponsiveColorMinYCoCg, diffuseResponsiveCenterYCoCg.xyz);
        diffuseResponsiveColorMaxYCoCg = max(diffuseResponsiveColorMaxYCoCg, diffuseResponsiveCenterYCoCg.xyz);

        float gDiffMaxFastAccumulatedFrameNum = 6;
        float gDiffMaxAccumulatedFrameNum = 30;

        // Clamping color with color box expansion
        Float4 diffuseIlluminationAnd2ndMoment = Load2DFloat4(illuminationPingBuffer, pixelPos);
        Float3 diffuseYCoCg = RgbToYCoCg(diffuseIlluminationAnd2ndMoment.xyz);
        Float3 clampedDiffuseYCoCg = diffuseYCoCg;

        clampedDiffuseYCoCg = clamp3f(diffuseYCoCg, diffuseResponsiveColorMinYCoCg, diffuseResponsiveColorMaxYCoCg);
        Float3 clampedDiffuse = YCoCgToRgb(clampedDiffuseYCoCg);

        constexpr float gHistoryFixFrameNum = 4.0f;

        // If history length is less than gHistoryFixFrameNum,
        // then it is the pixel with history fix applied in the previous (history fix) shader,
        // so data from responsive history needs to be copied to normal history,
        // and no history clamping is needed.
        Float4 outDiffuse = Float4(clampedDiffuse, diffuseIlluminationAnd2ndMoment.w);
        Float3 diffuseResponsiveCenter = YCoCgToRgb(diffuseResponsiveCenterYCoCg.xyz);
        Float4 outDiffuseResponsive = Float4(diffuseResponsiveCenter, 0.0f);
        if (historyLength <= gHistoryFixFrameNum)
            outDiffuse.xyz = outDiffuseResponsive.xyz;

        // Clamping factor: (clamped - slow) / (fast - slow)
        // The closer clamped is to fast, the closer clamping factor is to 1.
        float diffClampingFactor = (clampedDiffuseYCoCg.x - diffuseYCoCg.x) == 0.0f ? 0.0f : saturate((clampedDiffuseYCoCg.x - diffuseYCoCg.x) / (diffuseResponsiveCenterYCoCg.x - diffuseYCoCg.x));

        if (historyLength <= gHistoryFixFrameNum)
            diffClampingFactor = 1.0f;

        float gHistoryAccelerationAmount = 0.3f;
        float gHistoryResetSpatialSigmaScale = 4.5f;
        float gHistoryResetTemporalSigmaScale = 0.5f;
        float gHistoryResetAmount = 0.5f;

        // History acceleration based on (responsive - normal)
        float antiLagAccelerationAmountScale = 10.0f;
        float diffuseHistoryDifferenceL = antiLagAccelerationAmountScale * gHistoryAccelerationAmount * luminance(abs(diffuseResponsiveCenter - diffuseIlluminationAnd2ndMoment.xyz));

        // History acceleration amount should be proportional to history clamping amount
        diffuseHistoryDifferenceL *= diffClampingFactor;

        // No history acceleration if there is no difference between normal and responsive history
        if (historyLength <= gHistoryFixFrameNum)
            diffuseHistoryDifferenceL = 0.0f;

        // Using color space distance from responsive history to averaged noisy input to accelerate history
        Float3 diffuseColorDistanceToNoisyInput = diffuseNoisyFirstMoment - diffuseResponsiveCenter;

        float diffuseColorDistanceToNoisyInputL = luminance(abs(diffuseColorDistanceToNoisyInput));

        Float3 diffuseColorAcceleration = (diffuseColorDistanceToNoisyInputL == 0.0f) ? Float3(0.0f) : diffuseColorDistanceToNoisyInput * diffuseHistoryDifferenceL / diffuseColorDistanceToNoisyInputL;

        // Preventing overshooting and noise amplification by making sure luminance of accelerated responsive history
        // does not move beyond luminance of noisy input,
        // or does not move back from noisy input
        float diffuseColorAccelerationL = luminance(abs(diffuseColorAcceleration));

        float diffuseColorAccelerationRatio = (diffuseColorAccelerationL == 0.0f) ? 0.0f : diffuseColorDistanceToNoisyInputL / diffuseColorAccelerationL;

        if (diffuseColorAccelerationRatio < 1.0)
            diffuseColorAcceleration *= diffuseColorAccelerationRatio;

        if (diffuseColorAccelerationRatio <= 0.0f)
            diffuseColorAcceleration = Float3(0.0f);

        // Accelerating history
        outDiffuse.xyz += diffuseColorAcceleration;
        outDiffuseResponsive.xyz += diffuseColorAcceleration;

        // Calculating possibility for history reset
        float diffuseL = luminance(diffuseIlluminationAnd2ndMoment.xyz);
        float diffuseNoisyInputL = luminance(diffuseNoisyFirstMoment);
        float noisydiffuseTemporalSigma = gHistoryResetTemporalSigmaScale * sqrt(max(0.0f, diffuseNoisySecondMoment - diffuseNoisyInputL * diffuseNoisyInputL));
        float noisydiffuseSpatialSigma = gHistoryResetSpatialSigmaScale * diffuseResponsiveSigmaYCoCg.x;
        float diffuseHistoryResetAmount = gHistoryResetAmount * max(0.0f, abs(diffuseL - diffuseNoisyInputL) - noisydiffuseSpatialSigma - noisydiffuseTemporalSigma) / (1.0e-6f + max(diffuseL, diffuseNoisyInputL) + noisydiffuseSpatialSigma + noisydiffuseTemporalSigma);
        diffuseHistoryResetAmount = saturate(diffuseHistoryResetAmount);
        // Resetting history
        outDiffuse.xyz = lerp(outDiffuse.xyz, s_DiffNoisy_M2[sharedMemoryIndex.y * bufferSize + sharedMemoryIndex.x].xyz, diffuseHistoryResetAmount);
        outDiffuseResponsive.xyz = lerp(outDiffuseResponsive.xyz, s_DiffNoisy_M2[sharedMemoryIndex.y * bufferSize + sharedMemoryIndex.x].xyz, diffuseHistoryResetAmount);

        // 2nd moment correction
        float outDiffuseL = luminance(outDiffuse.xyz);
        float diffuseMomentCorrection = (outDiffuseL * outDiffuseL - diffuseL * diffuseL);

        outDiffuse.w += diffuseMomentCorrection;
        outDiffuse.w = max(0.0f, outDiffuse.w);

        // Writing outputs
        // Writing out history length for use in the next frame
        Store2DFloat4(outDiffuse, prevIlluminationBuffer, pixelPos);
        Store2DFloat4(outDiffuseResponsive, prevFastIlluminationBuffer, pixelPos);
        Store2DFloat1(historyLength, prevHistoryLengthBuffer, pixelPos);
    }
}