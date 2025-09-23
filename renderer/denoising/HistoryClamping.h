#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"

__device__ void Preload(
    Int2 sharedPos,
    Int2 globalPos,
    Int2 screenResolution,
    SurfObj diffuseIlluminationBuffer,
    SurfObj diffuseIlluminationPongBuffer,
    SurfObj specularIlluminationBuffer, 
    SurfObj specularIlluminationPongBuffer,
    Float4 *s_DiffResponsiveYCoCg,
    Float4 *s_DiffNoisy_M2,
    Float4 *s_SpecResponsiveYCoCg,
    Float4 *s_SpecNoisy_M2,
    int bufferSize)
{
    globalPos = clamp2i(globalPos, Int2(0), Int2(screenResolution.x - 1, screenResolution.y - 1));

    // Load diffuse data
    Float4 diffuseResponsive = Load2DFloat4(diffuseIlluminationPongBuffer, globalPos);
    Float4 diffuseResponsiveYcocg = RgbToYCoCg(diffuseResponsive);
    s_DiffResponsiveYCoCg[sharedPos.y * bufferSize + sharedPos.x] = diffuseResponsiveYcocg;

    Float4 diffuseNoisy = Load2DFloat4(diffuseIlluminationBuffer, globalPos);
    float diffuseNoisyLuminance = luminance(diffuseNoisy.xyz);
    s_DiffNoisy_M2[sharedPos.y * bufferSize + sharedPos.x] = Float4(diffuseNoisy.xyz, diffuseNoisyLuminance * diffuseNoisyLuminance);
    
    // Load specular data
    Float4 specularResponsive = Load2DFloat4(specularIlluminationPongBuffer, globalPos);
    Float4 specularResponsiveYcocg = RgbToYCoCg(specularResponsive);
    s_SpecResponsiveYCoCg[sharedPos.y * bufferSize + sharedPos.x] = specularResponsiveYcocg;

    Float4 specularNoisy = Load2DFloat4(specularIlluminationBuffer, globalPos);
    float specularNoisyLuminance = luminance(specularNoisy.xyz);
    s_SpecNoisy_M2[sharedPos.y * bufferSize + sharedPos.x] = Float4(specularNoisy.xyz, specularNoisyLuminance * specularNoisyLuminance);
}

template <int groupSize, int border>
__global__ void HistoryClamping(
    Int2 screenResolution,
    Float2 invScreenResolution,

    SurfObj depthBuffer,
    SurfObj historyLengthBuffer,

    // Separate diffuse buffers
    SurfObj diffuseIlluminationBuffer,
    SurfObj diffuseIlluminationPingBuffer,
    SurfObj diffuseIlluminationPongBuffer,
    SurfObj prevDiffuseIlluminationBuffer,
    SurfObj prevDiffuseFastIlluminationBuffer,
    
    // Separate specular buffers
    SurfObj specularIlluminationBuffer,
    SurfObj specularIlluminationPingBuffer,
    SurfObj specularIlluminationPongBuffer,
    SurfObj prevSpecularIlluminationBuffer,
    SurfObj prevSpecularFastIlluminationBuffer,
    SurfObj specularHitDistBuffer,
    SurfObj prevSpecularHitDistBuffer,
    
    SurfObj prevHistoryLengthBuffer,
    
    // RELAX clamping parameters
    float historyClampingColorBoxSigmaScale,
    float historyAccelerationAmount,
    float historyResetAmount)
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
    __shared__ Float4 s_SpecNoisy_M2[bufferSize * bufferSize];
    __shared__ Float4 s_SpecResponsiveYCoCg[bufferSize * bufferSize];

    Float2 pixelUv = Float2(Float2(pixelPos.x, pixelPos.y) + 0.5f) * invScreenResolution;

    Int2 groupBase = pixelPos - threadPos - border;
    unsigned int stageNum = (bufferSize * bufferSize + groupSize * groupSize - 1) / (groupSize * groupSize);
    for (unsigned int stage = 0; stage < stageNum; stage++)
    {
        unsigned int virtualIndex = threadIndex + stage * groupSize * groupSize;
        Int2 newId = Int2(virtualIndex % bufferSize, virtualIndex / bufferSize);
        if (stage == 0 || virtualIndex < bufferSize * bufferSize)
            Preload(newId, groupBase + newId, screenResolution, 
                   diffuseIlluminationBuffer, diffuseIlluminationPongBuffer,
                   specularIlluminationBuffer, specularIlluminationPongBuffer,
                   s_DiffResponsiveYCoCg, s_DiffNoisy_M2, s_SpecResponsiveYCoCg, s_SpecNoisy_M2, bufferSize);
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

    // Reading normal history for both diffuse and specular
    Float3 diffuseResponsiveFirstMomentYCoCg = Float3(0.0f);
    Float3 diffuseResponsiveSecondMomentYCoCg = Float3(0.0f);
    Float3 diffuseNoisyFirstMoment = Float3(0.0f);
    float diffuseNoisySecondMoment = 0.0f;
    
    Float3 specularResponsiveFirstMomentYCoCg = Float3(0.0f);
    Float3 specularResponsiveSecondMomentYCoCg = Float3(0.0f);
    Float3 specularNoisyFirstMoment = Float3(0.0f);
    float specularNoisySecondMoment = 0.0f;

    // Running history clamping
    Int2 sharedMemoryIndex = threadPos + Int2(border, border);
    for (int dx = -2; dx <= 2; dx++)
    {
        for (int dy = -2; dy <= 2; dy++)
        {
            Int2 sharedMemoryIndexP = sharedMemoryIndex + Int2(dx, dy);

            // Process diffuse samples
            Float3 diffuseSampleYCoCg = s_DiffResponsiveYCoCg[sharedMemoryIndexP.y * bufferSize + sharedMemoryIndexP.x].xyz;
            diffuseResponsiveFirstMomentYCoCg += diffuseSampleYCoCg;
            diffuseResponsiveSecondMomentYCoCg += diffuseSampleYCoCg * diffuseSampleYCoCg;

            Float4 diffuseNoisySample = s_DiffNoisy_M2[sharedMemoryIndexP.y * bufferSize + sharedMemoryIndexP.x];
            diffuseNoisyFirstMoment += diffuseNoisySample.xyz;
            diffuseNoisySecondMoment += diffuseNoisySample.w;
            
            // Process specular samples
            Float3 specularSampleYCoCg = s_SpecResponsiveYCoCg[sharedMemoryIndexP.y * bufferSize + sharedMemoryIndexP.x].xyz;
            specularResponsiveFirstMomentYCoCg += specularSampleYCoCg;
            specularResponsiveSecondMomentYCoCg += specularSampleYCoCg * specularSampleYCoCg;

            Float4 specularNoisySample = s_SpecNoisy_M2[sharedMemoryIndexP.y * bufferSize + sharedMemoryIndexP.x];
            specularNoisyFirstMoment += specularNoisySample.xyz;
            specularNoisySecondMoment += specularNoisySample.w;
        }
    }

    // Calculating color box for both diffuse and specular
    const float sampleCount = 25.0f; // 5x5 neighborhood
    
    // Diffuse color box
    diffuseResponsiveFirstMomentYCoCg /= sampleCount;
    diffuseResponsiveSecondMomentYCoCg /= sampleCount;
    diffuseNoisyFirstMoment /= sampleCount;
    diffuseNoisySecondMoment /= sampleCount;
    Float3 diffuseResponsiveSigmaYCoCg = sqrt3f(max3f(Float3(0.0f), diffuseResponsiveSecondMomentYCoCg - diffuseResponsiveFirstMomentYCoCg * diffuseResponsiveFirstMomentYCoCg));
    Float3 diffuseResponsiveColorMinYCoCg = diffuseResponsiveFirstMomentYCoCg - historyClampingColorBoxSigmaScale * diffuseResponsiveSigmaYCoCg;
    Float3 diffuseResponsiveColorMaxYCoCg = diffuseResponsiveFirstMomentYCoCg + historyClampingColorBoxSigmaScale * diffuseResponsiveSigmaYCoCg;
    
    // Specular color box
    specularResponsiveFirstMomentYCoCg /= sampleCount;
    specularResponsiveSecondMomentYCoCg /= sampleCount;
    specularNoisyFirstMoment /= sampleCount;
    specularNoisySecondMoment /= sampleCount;
    Float3 specularResponsiveSigmaYCoCg = sqrt3f(max3f(Float3(0.0f), specularResponsiveSecondMomentYCoCg - specularResponsiveFirstMomentYCoCg * specularResponsiveFirstMomentYCoCg));
    Float3 specularResponsiveColorMinYCoCg = specularResponsiveFirstMomentYCoCg - historyClampingColorBoxSigmaScale * specularResponsiveSigmaYCoCg;
    Float3 specularResponsiveColorMaxYCoCg = specularResponsiveFirstMomentYCoCg + historyClampingColorBoxSigmaScale * specularResponsiveSigmaYCoCg;

    // Expanding color box with color of the center pixel to minimize introduced bias
    Float4 diffuseResponsiveCenterYCoCg = s_DiffResponsiveYCoCg[sharedMemoryIndex.y * bufferSize + sharedMemoryIndex.x];
    diffuseResponsiveColorMinYCoCg = min(diffuseResponsiveColorMinYCoCg, diffuseResponsiveCenterYCoCg.xyz);
    diffuseResponsiveColorMaxYCoCg = max(diffuseResponsiveColorMaxYCoCg, diffuseResponsiveCenterYCoCg.xyz);
    
    Float4 specularResponsiveCenterYCoCg = s_SpecResponsiveYCoCg[sharedMemoryIndex.y * bufferSize + sharedMemoryIndex.x];
    specularResponsiveColorMinYCoCg = min(specularResponsiveColorMinYCoCg, specularResponsiveCenterYCoCg.xyz);
    specularResponsiveColorMaxYCoCg = max(specularResponsiveColorMaxYCoCg, specularResponsiveCenterYCoCg.xyz);

    // Clamping color with color box expansion for both diffuse and specular
    Float4 diffuseIlluminationAnd2ndMoment = Load2DFloat4(diffuseIlluminationPingBuffer, pixelPos);
    Float3 diffuseYCoCg = RgbToYCoCg(diffuseIlluminationAnd2ndMoment.xyz);
    Float3 clampedDiffuseYCoCg = clamp3f(diffuseYCoCg, diffuseResponsiveColorMinYCoCg, diffuseResponsiveColorMaxYCoCg);
    Float3 clampedDiffuse = YCoCgToRgb(clampedDiffuseYCoCg);
    
    Float4 specularIlluminationAnd2ndMoment = Load2DFloat4(specularIlluminationPingBuffer, pixelPos);
    Float3 specularYCoCg = RgbToYCoCg(specularIlluminationAnd2ndMoment.xyz);
    Float3 clampedSpecularYCoCg = clamp3f(specularYCoCg, specularResponsiveColorMinYCoCg, specularResponsiveColorMaxYCoCg);
    Float3 clampedSpecular = YCoCgToRgb(clampedSpecularYCoCg);

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
    
    Float4 outSpecular = Float4(clampedSpecular, specularIlluminationAnd2ndMoment.w);
    Float3 specularResponsiveCenter = YCoCgToRgb(specularResponsiveCenterYCoCg.xyz);
    Float4 outSpecularResponsive = Float4(specularResponsiveCenter, 0.0f);
    if (historyLength <= gHistoryFixFrameNum)
        outSpecular.xyz = outSpecularResponsive.xyz;

    // Clamping factor: (clamped - slow) / (fast - slow)
    // The closer clamped is to fast, the closer clamping factor is to 1.
    float diffClampingFactor = (clampedDiffuseYCoCg.x - diffuseYCoCg.x) == 0.0f ? 0.0f : saturate((clampedDiffuseYCoCg.x - diffuseYCoCg.x) / (diffuseResponsiveCenterYCoCg.x - diffuseYCoCg.x));
    float specClampingFactor = (clampedSpecularYCoCg.x - specularYCoCg.x) == 0.0f ? 0.0f : saturate((clampedSpecularYCoCg.x - specularYCoCg.x) / (specularResponsiveCenterYCoCg.x - specularYCoCg.x));

    if (historyLength <= gHistoryFixFrameNum)
    {
        diffClampingFactor = 1.0f;
        specClampingFactor = 1.0f;
    }

    float gHistoryResetSpatialSigmaScale = 4.5f;
    float gHistoryResetTemporalSigmaScale = 0.5f;

    // History acceleration based on (responsive - normal) for both diffuse and specular
    float antiLagAccelerationAmountScale = 10.0f;
    float diffuseHistoryDifferenceL = antiLagAccelerationAmountScale * historyAccelerationAmount * luminance(abs(diffuseResponsiveCenter - diffuseIlluminationAnd2ndMoment.xyz));
    float specularHistoryDifferenceL = antiLagAccelerationAmountScale * historyAccelerationAmount * luminance(abs(specularResponsiveCenter - specularIlluminationAnd2ndMoment.xyz));

    // History acceleration amount should be proportional to history clamping amount
    diffuseHistoryDifferenceL *= diffClampingFactor;
    specularHistoryDifferenceL *= specClampingFactor;

    // No history acceleration if there is no difference between normal and responsive history
    if (historyLength <= gHistoryFixFrameNum)
    {
        diffuseHistoryDifferenceL = 0.0f;
        specularHistoryDifferenceL = 0.0f;
    }

    // Using color space distance from responsive history to averaged noisy input to accelerate history
    Float3 diffuseColorDistanceToNoisyInput = diffuseNoisyFirstMoment - diffuseResponsiveCenter;
    Float3 specularColorDistanceToNoisyInput = specularNoisyFirstMoment - specularResponsiveCenter;

    float diffuseColorDistanceToNoisyInputL = luminance(abs(diffuseColorDistanceToNoisyInput));
    float specularColorDistanceToNoisyInputL = luminance(abs(specularColorDistanceToNoisyInput));

    Float3 diffuseColorAcceleration = (diffuseColorDistanceToNoisyInputL == 0.0f) ? Float3(0.0f) : diffuseColorDistanceToNoisyInput * diffuseHistoryDifferenceL / diffuseColorDistanceToNoisyInputL;
    Float3 specularColorAcceleration = (specularColorDistanceToNoisyInputL == 0.0f) ? Float3(0.0f) : specularColorDistanceToNoisyInput * specularHistoryDifferenceL / specularColorDistanceToNoisyInputL;

    // Preventing overshooting and noise amplification by making sure luminance of accelerated responsive history
    // does not move beyond luminance of noisy input, or does not move back from noisy input
    
    // Diffuse acceleration clamping
    float diffuseColorAccelerationL = luminance(abs(diffuseColorAcceleration));
    float diffuseColorAccelerationRatio = (diffuseColorAccelerationL == 0.0f) ? 0.0f : diffuseColorDistanceToNoisyInputL / diffuseColorAccelerationL;
    if (diffuseColorAccelerationRatio < 1.0f)
        diffuseColorAcceleration *= diffuseColorAccelerationRatio;
    if (diffuseColorAccelerationRatio <= 0.0f)
        diffuseColorAcceleration = Float3(0.0f);
    
    // Specular acceleration clamping
    float specularColorAccelerationL = luminance(abs(specularColorAcceleration));
    float specularColorAccelerationRatio = (specularColorAccelerationL == 0.0f) ? 0.0f : specularColorDistanceToNoisyInputL / specularColorAccelerationL;
    if (specularColorAccelerationRatio < 1.0f)
        specularColorAcceleration *= specularColorAccelerationRatio;
    if (specularColorAccelerationRatio <= 0.0f)
        specularColorAcceleration = Float3(0.0f);

    // Accelerating history for both diffuse and specular
    outDiffuse.xyz += diffuseColorAcceleration;
    outDiffuseResponsive.xyz += diffuseColorAcceleration;
    outSpecular.xyz += specularColorAcceleration;
    outSpecularResponsive.xyz += specularColorAcceleration;

    // Calculating possibility for history reset for both diffuse and specular
    float diffuseL = luminance(diffuseIlluminationAnd2ndMoment.xyz);
    float diffuseNoisyInputL = luminance(diffuseNoisyFirstMoment);
    float noisydiffuseTemporalSigma = gHistoryResetTemporalSigmaScale * sqrt(max(0.0f, diffuseNoisySecondMoment - diffuseNoisyInputL * diffuseNoisyInputL));
    float noisydiffuseSpatialSigma = gHistoryResetSpatialSigmaScale * diffuseResponsiveSigmaYCoCg.x;
    float diffuseHistoryResetAmount = historyResetAmount * max(0.0f, abs(diffuseL - diffuseNoisyInputL) - noisydiffuseSpatialSigma - noisydiffuseTemporalSigma) / (1.0e-6f + max(diffuseL, diffuseNoisyInputL) + noisydiffuseSpatialSigma + noisydiffuseTemporalSigma);
    diffuseHistoryResetAmount = saturate(diffuseHistoryResetAmount);
    
    float specularL = luminance(specularIlluminationAnd2ndMoment.xyz);
    float specularNoisyInputL = luminance(specularNoisyFirstMoment);
    float noisyspecularTemporalSigma = gHistoryResetTemporalSigmaScale * sqrt(max(0.0f, specularNoisySecondMoment - specularNoisyInputL * specularNoisyInputL));
    float noisyspecularSpatialSigma = gHistoryResetSpatialSigmaScale * specularResponsiveSigmaYCoCg.x;
    float specularHistoryResetAmount = historyResetAmount * max(0.0f, abs(specularL - specularNoisyInputL) - noisyspecularSpatialSigma - noisyspecularTemporalSigma) / (1.0e-6f + max(specularL, specularNoisyInputL) + noisyspecularSpatialSigma + noisyspecularTemporalSigma);
    specularHistoryResetAmount = saturate(specularHistoryResetAmount);
    
    // Resetting history for both diffuse and specular
    outDiffuse.xyz = lerp(outDiffuse.xyz, s_DiffNoisy_M2[sharedMemoryIndex.y * bufferSize + sharedMemoryIndex.x].xyz, diffuseHistoryResetAmount);
    outDiffuseResponsive.xyz = lerp(outDiffuseResponsive.xyz, s_DiffNoisy_M2[sharedMemoryIndex.y * bufferSize + sharedMemoryIndex.x].xyz, diffuseHistoryResetAmount);
    outSpecular.xyz = lerp(outSpecular.xyz, s_SpecNoisy_M2[sharedMemoryIndex.y * bufferSize + sharedMemoryIndex.x].xyz, specularHistoryResetAmount);
    outSpecularResponsive.xyz = lerp(outSpecularResponsive.xyz, s_SpecNoisy_M2[sharedMemoryIndex.y * bufferSize + sharedMemoryIndex.x].xyz, specularHistoryResetAmount);

    // 2nd moment correction for both diffuse and specular
    float outDiffuseL = luminance(outDiffuse.xyz);
    float diffuseMomentCorrection = (outDiffuseL * outDiffuseL - diffuseL * diffuseL);
    outDiffuse.w += diffuseMomentCorrection;
    outDiffuse.w = max(0.0f, outDiffuse.w);
    
    float outSpecularL = luminance(outSpecular.xyz);
    float specularMomentCorrection = (outSpecularL * outSpecularL - specularL * specularL);
    outSpecular.w += specularMomentCorrection;
    outSpecular.w = max(0.0f, outSpecular.w);

    // Writing outputs for both diffuse and specular
    // Writing out history length for use in the next frame
    Store2DFloat4(outDiffuse, prevDiffuseIlluminationBuffer, pixelPos);
    Store2DFloat4(outDiffuseResponsive, prevDiffuseFastIlluminationBuffer, pixelPos);
    Store2DFloat4(outSpecular, prevSpecularIlluminationBuffer, pixelPos);
    Store2DFloat4(outSpecularResponsive, prevSpecularFastIlluminationBuffer, pixelPos);
    
    // Also need to handle specular hit distance (preserve existing value for now)
    float currentSpecularHitDist = Load2DFloat1(specularHitDistBuffer, pixelPos);
    Store2DFloat1(currentSpecularHitDist, prevSpecularHitDistBuffer, pixelPos);
    
    Store2DFloat1(historyLength, prevHistoryLengthBuffer, pixelPos);
}