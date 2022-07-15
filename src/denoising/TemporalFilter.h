#include "shaders/LinearMath.h"
#include "shaders/Sampler.h"
#include "util/Gaussian.h"
#include "core/GlobalSettings.h"
#include "shaders/ShaderDebugUtils.h"

namespace jazzfusion
{

__global__ void CopyToHistoryColorBuffer(
    SurfObj   colorBuffer,
    SurfObj   historyColorBuffer,

    SurfObj   depthBuffer,
    SurfObj   depthHistoryBuffer,

    SurfObj   materialBuffer,
    SurfObj   materialHistoryBuffer,

    Int2    size)
{
    int x = threadIdx.x + blockIdx.x * 8;
    int y = threadIdx.y + blockIdx.y * 8;

    if (x >= size.x || y >= size.y) return;

    Int2 idx(x, y);

    surf2Dwrite(surf2Dread<ushort4>(colorBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp), historyColorBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);
    surf2Dwrite(surf2Dread<ushort1>(depthBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp), depthHistoryBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp);
    surf2Dwrite(surf2Dread<ushort1>(materialBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp), materialHistoryBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp);

    // surf2Dwrite(surf2Dread<ushort4>(normalBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp), normalHistoryBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);
}

__global__ void CopyToAccumulationColorBuffer(
    SurfObj colorBuffer,
    SurfObj accumulateBuffer,

    SurfObj albedoBuffer,
    SurfObj historyAlbedoBuffer,

    Int2    size)
{
    int x = threadIdx.x + blockIdx.x * 8;
    int y = threadIdx.y + blockIdx.y * 8;

    if (x >= size.x || y >= size.y) return;

    Int2 idx(x, y);

    surf2Dwrite(surf2Dread<ushort4>(colorBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp), accumulateBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);
    surf2Dwrite(surf2Dread<ushort4>(albedoBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp), historyAlbedoBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);
}

__global__ void ResetAccumulationCounter(SurfObj accuCounterBuffer, Int2 size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= size.x && y >= size.y)
    {
        return;
    }

    Store2DUshort1(1, accuCounterBuffer, Int2(x, y));
}

__global__ void TemporalFilter(
    int frameNum,
    int accuCounter,
    SurfObj   colorBuffer,
    SurfObj   accumulateBuffer,
    SurfObj   historyColorBuffer,
    SurfObj   normalBuffer,
    SurfObj   normalHistoryBuffer,
    SurfObj   depthBuffer,
    SurfObj   depthHistoryBuffer,
    SurfObj   materialBuffer,
    SurfObj   materialHistoryBuffer,
    SurfObj   albedoBuffer,
    SurfObj   historyAlbedoBuffer,
    SurfObj   motionVectorBuffer,
    DenoisingParams params,
    Int2      size,
    Int2      historySize)
{
    // Settings
    float depthDiffLimit = params.temporal_denoise_depth_diff_threshold;
    constexpr bool enableAntiFlickering = true;
    constexpr bool enableBlendUsingLumaHdrFactor = true;
    constexpr int blockdim = 8;
    constexpr int kernelRadius = 1;
    constexpr int threadCount = blockdim * blockdim;
    constexpr int kernelCoverDim = blockdim + kernelRadius * 2;
    constexpr int kernelCoverSize = kernelCoverDim * kernelCoverDim;
    constexpr int kernelDim = kernelRadius * 2 + 1;
    constexpr int kernelSize = kernelDim * kernelDim;

    __shared__ Float3 colorSharedBuffer[kernelCoverSize];
    __shared__ Float3 albedoSharedBuffer[kernelCoverSize];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = (threadIdx.x + threadIdx.y * blockdim);
    int x1 = blockIdx.x * blockdim - kernelRadius + id % kernelCoverDim;
    int y1 = blockIdx.y * blockdim - kernelRadius + id / kernelCoverDim;
    int x2 = blockIdx.x * blockdim - kernelRadius + (id + threadCount) % kernelCoverDim;
    int y2 = blockIdx.y * blockdim - kernelRadius + (id + threadCount) / kernelCoverDim;
    int centerIdx = threadIdx.x + kernelRadius + (threadIdx.y + kernelRadius) * kernelCoverDim;

    Float3 color1 = Load2DHalf4(colorBuffer, Int2(x1, y1)).xyz;
    Float3 color2 = Load2DHalf4(colorBuffer, Int2(x2, y2)).xyz;
    Float3 albedo1 = Load2DHalf4(albedoBuffer, Int2(x1, y1)).xyz;
    Float3 albedo2 = Load2DHalf4(albedoBuffer, Int2(x2, y2)).xyz;

    colorSharedBuffer[id] = color1;
    albedoSharedBuffer[id] = albedo1;
    if (id + threadCount < kernelCoverSize)
    {
        colorSharedBuffer[id + threadCount] = color2;
        albedoSharedBuffer[id + threadCount] = albedo2;
    }
    __syncthreads();

    // Return out of bound thread
    if (x >= size.x && y >= size.y)
    {
        return;
    }

    Float3 outColor;
    Float3 outAlbedo;

    // Return for straight-to-sky pixel
    uint maskValue = Load2DUshort1(materialBuffer, Int2(x, y));
    bool isSky = (maskValue == 0);
    if (isSky == true)
    {
        return;
    }

    // sample history color
    Float2 motionVec = Load2DHalf2(motionVectorBuffer, Int2(x, y)) - Float2(0.5f);
    Float2 uv = (Float2(x, y) + Float2(0.5f, 0.5f)) * (Float2(1.0f) / Float2(size.x, size.y));
    Float2 historyUv = uv + motionVec;
    if (historyUv.x < 0 || historyUv.y < 0 || historyUv.x > 1.0f || historyUv.y > 1.0f)
    {
        return;
    }

    // load center
    Float3 colorValue = colorSharedBuffer[centerIdx];
    Float3 albedoValue = albedoSharedBuffer[centerIdx];

    // Neighbor max/min
    Float3 neighbourMax = RgbToYcocg(colorValue);
    Float3 neighbourMin = RgbToYcocg(colorValue);
    Float3 neighbourMax2 = RgbToYcocg(colorValue);
    Float3 neighbourMin2 = RgbToYcocg(colorValue);

    Float3 neighbourAlbedoMax = RgbToYcocg(albedoValue);
    Float3 neighbourAlbedoMin = RgbToYcocg(albedoValue);
    Float3 neighbourAlbedoMax2 = RgbToYcocg(albedoValue);
    Float3 neighbourAlbedoMin2 = RgbToYcocg(albedoValue);

#pragma unroll
    for (int i = 0; i < kernelSize; ++i)
    {
        int xoffset = i % kernelDim;
        int yoffset = i / kernelDim;

        if (xoffset == kernelRadius && yoffset == kernelRadius)
        {
            continue;
        }

        // get data
        Float3 color = colorSharedBuffer[threadIdx.x + xoffset + (threadIdx.y + yoffset) * kernelCoverDim];
        Float3 albedo = albedoSharedBuffer[threadIdx.x + xoffset + (threadIdx.y + yoffset) * kernelCoverDim];

        // max min
        Float3 neighbourColor = RgbToYcocg(color);
        neighbourMax = max3f(neighbourMax, neighbourColor);
        neighbourMin = min3f(neighbourMin, neighbourColor);
        if ((abs(xoffset - kernelDim / 2) <= 1) && (abs(yoffset - kernelDim / 2) <= 1))
        {
            neighbourMax2 = max3f(neighbourMax2, neighbourColor);
            neighbourMin2 = min3f(neighbourMin2, neighbourColor);
        }

        Float3 neighbourAlbedo = RgbToYcocg(albedo);
        neighbourAlbedoMax = max3f(neighbourAlbedoMax, neighbourAlbedo);
        neighbourAlbedoMin = min3f(neighbourAlbedoMin, neighbourAlbedo);
        if ((abs(xoffset - kernelDim / 2) <= 1) && (abs(yoffset - kernelDim / 2) <= 1))
        {
            neighbourAlbedoMax2 = max3f(neighbourAlbedoMax2, neighbourAlbedo);
            neighbourAlbedoMin2 = min3f(neighbourAlbedoMin2, neighbourAlbedo);
        }
    }

    neighbourMax = (neighbourMax + neighbourMax2) / 2.0f;
    neighbourMin = (neighbourMin + neighbourMin2) / 2.0f;
    neighbourAlbedoMax = (neighbourAlbedoMax + neighbourAlbedoMax2) / 2.0f;
    neighbourAlbedoMin = (neighbourAlbedoMin + neighbourAlbedoMin2) / 2.0f;

    // sample history
    Float3 colorHistory = SampleBicubicSmoothStep<Load2DFuncHalf4<Float3>>(accumulateBuffer, historyUv, historySize);
    Float3 albedoHistory = SampleBicubicSmoothStep<Load2DFuncHalf4<Float3>>(historyAlbedoBuffer, historyUv, historySize);

    // Clamp history
    Float3 colorHistoryYcocg = RgbToYcocg(colorHistory);
    Float3 clampedColorHistory = YcocgToRgb(clamp3f(colorHistoryYcocg, neighbourMin, neighbourMax));
    Float3 albedoHistoryYcocg = RgbToYcocg(albedoHistory);
    Float3 clampedAlbedoHistory = YcocgToRgb(clamp3f(albedoHistoryYcocg, neighbourAlbedoMin, neighbourAlbedoMax));

    // Discard history factor. 0 = no discard, good history sample. 1 = full discard, total bad history sample.
    float discardHistoryFactor = 0.0f;
    //     if ((abs(motionVec.x) + abs(motionVec.y)) > 1e-4f)
    //     {
    //         Int2 historyIdx = Int2(floorf(historyUv.x * historySize.x), floorf(historyUv.y * historySize.y));
    //         float depthValue = Load2DHalf1(depthBuffer, Int2(x, y));
    // #pragma unroll
    //         for (int i = 0; i < 4; ++i)
    //         {
    //             Int2 offset(i % 2, i / 2);

    //             Int2 historyNeighborIdx = historyIdx + offset;

    //             float depthHistory = Load2DHalf1(depthHistoryBuffer, historyNeighborIdx);
    //             ushort maskHistory = Load2DUshort1(materialHistoryBuffer, historyNeighborIdx);

    //             float depthDiff = logf(depthHistory) - logf(depthValue);

    //             discardHistoryFactor += (float)((maskValue != maskHistory) || (abs(depthDiff) > depthDiffLimit));
    //         }
    //         discardHistoryFactor /= 4.0f;
    //     }

    Float3 blendedColorHistory = lerp3f(clampedColorHistory, colorValue, discardHistoryFactor);
    Float3 blendedAlbedoHistory = lerp3f(clampedAlbedoHistory, albedoValue, discardHistoryFactor);

    float lumaHistory;
    float lumaMin;
    float lumaMax;
    float lumaCurrent;

    if constexpr (enableAntiFlickering || enableBlendUsingLumaHdrFactor)
    {
        lumaMin = neighbourMin.x * neighbourAlbedoMin.x;
        lumaMax = neighbourMax.x * neighbourAlbedoMax.x;
        lumaCurrent = RgbToYcocg(colorValue).x * RgbToYcocg(albedoValue).x;
        lumaHistory = RgbToYcocg(blendedColorHistory).x * RgbToYcocg(blendedAlbedoHistory).x;
    }

    float blendFactor = 1.0f / 16.0f;

    if constexpr (enableAntiFlickering)
    {
        // anti flickering
        float antiFlickeringFactor = clampf(0.5f * min(abs(lumaHistory - lumaMin), abs(lumaHistory - lumaMax)) / max3(lumaHistory, lumaCurrent, 1e-4f));
        blendFactor *= (1.0f - 0.8f) + 0.8f * antiFlickeringFactor;
    }

    if constexpr (enableBlendUsingLumaHdrFactor)
    {
        // weight with luma hdr factor
        float weightA = blendFactor * max(0.0001f, 1.0f / (lumaCurrent + 4.0f));
        float weightB = (1.0f - blendFactor) * max(0.0001f, 1.0f / (lumaHistory + 4.0f));
        float weightSum = SafeDivide(1.0f, weightA + weightB);
        weightA *= weightSum;
        weightB *= weightSum;

        outColor = colorValue * weightA + blendedColorHistory * weightB;
        outAlbedo = albedoValue * weightA + blendedAlbedoHistory * weightB;
    }
    else
    {
        outColor = lerp3f(blendedColorHistory, colorValue, blendFactor);
        outAlbedo = lerp3f(blendedAlbedoHistory, albedoValue, blendFactor);
    }

    Store2DHalf4(Float4(outColor, 0), colorBuffer, Int2(x, y));
    Store2DHalf4(Float4(outAlbedo, 0), albedoBuffer, Int2(x, y));
}


__global__ void TemporalFilter2(
    int frameNum,
    int accuCounter,
    SurfObj   colorBuffer,
    SurfObj   accumulateBuffer,
    SurfObj   historyColorBuffer,
    SurfObj   normalBuffer,
    SurfObj   normalHistoryBuffer,
    SurfObj   depthBuffer,
    SurfObj   depthHistoryBuffer,
    SurfObj   materialBuffer,
    SurfObj   materialHistoryBuffer,
    SurfObj   albedoBuffer,
    SurfObj   historyAlbedoBuffer,
    SurfObj   motionVectorBuffer,
    DenoisingParams params,
    Int2      size,
    Int2      historySize)
{
    // Settings
    float depthDiffLimit = params.temporal_denoise_depth_diff_threshold;
    constexpr bool enableAntiFlickering = true;
    constexpr bool enableBlendUsingLumaHdrFactor = true;
    constexpr int blockdim = 8;
    constexpr int kernelRadius = 1;
    constexpr int threadCount = blockdim * blockdim;
    constexpr int kernelCoverDim = blockdim + kernelRadius * 2;
    constexpr int kernelCoverSize = kernelCoverDim * kernelCoverDim;
    constexpr int kernelDim = kernelRadius * 2 + 1;
    constexpr int kernelSize = kernelDim * kernelDim;

    __shared__ Float3 colorSharedBuffer[kernelCoverSize];
    __shared__ Float3 albedoSharedBuffer[kernelCoverSize];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = (threadIdx.x + threadIdx.y * blockdim);
    int x1 = blockIdx.x * blockdim - kernelRadius + id % kernelCoverDim;
    int y1 = blockIdx.y * blockdim - kernelRadius + id / kernelCoverDim;
    int x2 = blockIdx.x * blockdim - kernelRadius + (id + threadCount) % kernelCoverDim;
    int y2 = blockIdx.y * blockdim - kernelRadius + (id + threadCount) / kernelCoverDim;
    int centerIdx = threadIdx.x + kernelRadius + (threadIdx.y + kernelRadius) * kernelCoverDim;

    Float3 color1 = Load2DHalf4(colorBuffer, Int2(x1, y1)).xyz;
    Float3 color2 = Load2DHalf4(colorBuffer, Int2(x2, y2)).xyz;

    colorSharedBuffer[id] = color1;
    if (id + threadCount < kernelCoverSize)
    {
        colorSharedBuffer[id + threadCount] = color2;
    }
    __syncthreads();

    // Return out of bound thread
    if (x >= size.x && y >= size.y)
    {
        return;
    }

    Float3 outColor;

    // Return for straight-to-sky pixel
    uint maskValue = Load2DUshort1(materialBuffer, Int2(x, y));
    bool isSky = (maskValue == 0);
    if (isSky == true)
    {
        return;
    }

    // sample history color
    Float2 motionVec = Load2DHalf2(motionVectorBuffer, Int2(x, y)) - Float2(0.5f);
    Float2 uv = (Float2(x, y) + Float2(0.5f, 0.5f)) * (Float2(1.0f) / Float2(size.x, size.y));
    Float2 historyUv = uv + motionVec;
    if (historyUv.x < 0 || historyUv.y < 0 || historyUv.x > 1.0f || historyUv.y > 1.0f)
    {
        return;
    }

    // load center
    Float3 colorValue = colorSharedBuffer[centerIdx];

    // Neighbor max/min
    Float3 neighbourMax = RgbToYcocg(colorValue);
    Float3 neighbourMin = RgbToYcocg(colorValue);
    Float3 neighbourMax2 = RgbToYcocg(colorValue);
    Float3 neighbourMin2 = RgbToYcocg(colorValue);

#pragma unroll
    for (int i = 0; i < kernelSize; ++i)
    {
        int xoffset = i % kernelDim;
        int yoffset = i / kernelDim;

        if (xoffset == kernelRadius && yoffset == kernelRadius)
        {
            continue;
        }

        // get data
        Float3 color = colorSharedBuffer[threadIdx.x + xoffset + (threadIdx.y + yoffset) * kernelCoverDim];

        // max min
        Float3 neighbourColor = RgbToYcocg(color);
        neighbourMax = max3f(neighbourMax, neighbourColor);
        neighbourMin = min3f(neighbourMin, neighbourColor);
        if ((abs(xoffset - kernelDim / 2) <= 1) && (abs(yoffset - kernelDim / 2) <= 1))
        {
            neighbourMax2 = max3f(neighbourMax2, neighbourColor);
            neighbourMin2 = min3f(neighbourMin2, neighbourColor);
        }
    }

    neighbourMax = (neighbourMax + neighbourMax2) / 2.0f;
    neighbourMin = (neighbourMin + neighbourMin2) / 2.0f;

    // sample history
    Float3 colorHistory = SampleBicubicSmoothStep<Load2DFuncHalf4<Float3>>(historyColorBuffer, historyUv, historySize);

    // Clamp history
    Float3 colorHistoryYcocg = RgbToYcocg(colorHistory);
    Float3 clampedColorHistory = YcocgToRgb(clamp3f(colorHistoryYcocg, neighbourMin, neighbourMax));

    // Discard history factor. 0 = no discard, good history sample. 1 = full discard, total bad history sample.
    float discardHistoryFactor = 0.0f;
    //     if ((abs(motionVec.x) + abs(motionVec.y)) > 1e-4f)
    //     {
    //         Int2 historyIdx = Int2(floorf(historyUv.x * historySize.x), floorf(historyUv.y * historySize.y));
    //         float depthValue = Load2DHalf1(depthBuffer, Int2(x, y));
    // #pragma unroll
    //         for (int i = 0; i < 4; ++i)
    //         {
    //             Int2 offset(i % 2, i / 2);

    //             Int2 historyNeighborIdx = historyIdx + offset;

    //             float depthHistory = Load2DHalf1(depthHistoryBuffer, historyNeighborIdx);
    //             ushort maskHistory = Load2DUshort1(materialHistoryBuffer, historyNeighborIdx);

    //             float depthDiff = logf(depthHistory) - logf(depthValue);

    //             discardHistoryFactor += (float)((maskValue != maskHistory) || (abs(depthDiff) > depthDiffLimit));
    //         }
    //         discardHistoryFactor /= 4.0f;
    //     }

    Float3 blendedColorHistory = lerp3f(clampedColorHistory, colorValue, discardHistoryFactor);

    float lumaHistory;
    float lumaMin;
    float lumaMax;
    float lumaCurrent;

    if constexpr (enableAntiFlickering || enableBlendUsingLumaHdrFactor)
    {
        lumaMin = neighbourMin.x;
        lumaMax = neighbourMax.x;
        lumaCurrent = RgbToYcocg(colorValue).x;
        lumaHistory = RgbToYcocg(blendedColorHistory).x;
    }

    float blendFactor = 1.0f / 16.0f;

    if constexpr (enableAntiFlickering)
    {
        // anti flickering
        float antiFlickeringFactor = clampf(0.5f * min(abs(lumaHistory - lumaMin), abs(lumaHistory - lumaMax)) / max3(lumaHistory, lumaCurrent, 1e-4f));
        blendFactor *= (1.0f - 0.8f) + 0.8f * antiFlickeringFactor;
    }

    if constexpr (enableBlendUsingLumaHdrFactor)
    {
        // weight with luma hdr factor
        float weightA = blendFactor * max(0.0001f, 1.0f / (lumaCurrent + 4.0f));
        float weightB = (1.0f - blendFactor) * max(0.0001f, 1.0f / (lumaHistory + 4.0f));
        float weightSum = SafeDivide(1.0f, weightA + weightB);
        weightA *= weightSum;
        weightB *= weightSum;

        outColor = colorValue * weightA + blendedColorHistory * weightB;
    }
    else
    {
        outColor = lerp3f(blendedColorHistory, colorValue, blendFactor);
    }

    Store2DHalf4(Float4(outColor, 0), colorBuffer, Int2(x, y));
}

}