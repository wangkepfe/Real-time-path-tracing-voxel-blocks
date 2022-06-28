#include "shaders/LinearMath.h"
#include "shaders/Sampler.h"
#include "util/Gaussian.h"
#include "core/GlobalSettings.h"
#include "shaders/ShaderDebugUtils.h"

namespace jazzfusion
{

__global__ void CopyToHistoryColorDepthBuffer(
    int accuCounter,
    SurfObj colorBuffer,
    SurfObj depthBuffer,
    SurfObj historyColorBuffer,
    SurfObj depthHistoryBuffer,
    SurfObj   materialBuffer,
    SurfObj   materialHistoryBuffer,
    SurfObj   normalBuffer,
    SurfObj   normalHistoryBuffer,
    Int2    size)
{
    int x = threadIdx.x + blockIdx.x * 8;
    int y = threadIdx.y + blockIdx.y * 8;

    if (x >= size.x || y >= size.y) return;

    Int2 idx(x, y);

    surf2Dwrite(surf2Dread<ushort4>(colorBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp), historyColorBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);

    if (accuCounter == 1)
    {
        surf2Dwrite(surf2Dread<ushort1>(depthBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp), depthHistoryBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp);
        surf2Dwrite(surf2Dread<ushort1>(materialBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp), materialHistoryBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp);
        surf2Dwrite(surf2Dread<ushort4>(normalBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp), normalHistoryBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);
    }
    else
    {
        float factor = 1.0f / (float)accuCounter;

        Float3 normal = Load2DHalf4(normalBuffer, idx).xyz;
        Float3 normalHistory = Load2DHalf4(normalHistoryBuffer, idx).xyz;

        float depth = Load2DHalf1(depthBuffer, idx);
        float depthHistory = Load2DHalf1(depthHistoryBuffer, idx);

        ushort mat = Load2DUshort1(materialBuffer, idx);
        ushort matHistory = Load2DUshort1(materialHistoryBuffer, idx);

        Float3 outNormal = slerp3f(normalHistory, normal, factor);
        float outdepth = lerpf(depthHistory, depth, factor);
        ushort outMat = min(mat, matHistory);

        Store2DHalf4(Float4(outNormal, 0.0f), normalHistoryBuffer, idx);
        Store2DHalf1(outdepth, depthHistoryBuffer, idx);
        Store2DUshort1(outMat, materialHistoryBuffer, idx);
    }
}

__global__ void CopyToHistoryColorBuffer(
    SurfObj colorBuffer,
    SurfObj accumulateBuffer,
    Int2    size)
{
    int x = threadIdx.x + blockIdx.x * 8;
    int y = threadIdx.y + blockIdx.y * 8;

    if (x >= size.x || y >= size.y) return;

    Int2 idx(x, y);

    surf2Dwrite(surf2Dread<ushort4>(colorBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp), accumulateBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);
}

template<bool EnableAtrousFilter>
struct AtrousLDS
{
};

template<>
struct AtrousLDS<false>
{
    Half3 color;
    ushort mask;
};

template<>
struct AtrousLDS<true>
{
    Half3 color;
    ushort mask;
    Half3 normal;
    half depth;
};

template<bool EnableAtrousFilter>
__global__ void TemporalFilter(
    int frameNum,
    int accuCounter,
    SurfObj   colorBuffer,
    SurfObj   accumulateBuffer,
    SurfObj   normalBuffer,
    SurfObj   depthBuffer,
    SurfObj   depthHistoryBuffer,
    SurfObj   materialBuffer,
    SurfObj   materialHistoryBuffer,
    SurfObj   motionVectorBuffer,
    SurfObj   historyNormalBuffer,
    DenoisingParams params,
    Int2      size,
    Int2      historySize)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Settings
    constexpr int stride = 1;
    bool useSoftMax = params.temporal_denoise_use_softmax;
    float baseBlendingFactor = params.temporal_denoise_baseBlendingFactor;
    float antiFlickeringWeight = params.temporal_denoise_antiFlickeringWeight;
    float depthDiffLimit = params.temporal_denoise_depth_diff_threshold;

    constexpr bool enableAntiFlickering = true;
    constexpr bool enableBlendUsingLumaHdrFactor = true;

    constexpr int blockdim = 8;
    constexpr int kernelRadius = 1;


    // Calculations of kernel dim
    constexpr int threadCount = blockdim * blockdim;

    constexpr int kernelCoverDim = blockdim + kernelRadius * 2;
    constexpr int kernelCoverSize = kernelCoverDim * kernelCoverDim;

    constexpr int kernelDim = kernelRadius * 2 + 1;
    constexpr int kernelSize = kernelDim * kernelDim;

    int centerIdx = threadIdx.x + kernelRadius + (threadIdx.y + kernelRadius) * kernelCoverDim;

    __shared__ AtrousLDS<EnableAtrousFilter> sharedBuffer[kernelCoverSize];

    // calculate address
    int id = (threadIdx.x + threadIdx.y * blockdim);

    int x1 = blockIdx.x * blockdim - kernelRadius + id % kernelCoverDim;
    int y1 = blockIdx.y * blockdim - kernelRadius + id / kernelCoverDim;
    int x2 = blockIdx.x * blockdim - kernelRadius + (id + threadCount) % kernelCoverDim;
    int y2 = blockIdx.y * blockdim - kernelRadius + (id + threadCount) / kernelCoverDim;

    Half3 halfColor1 = Load2DHalf4<Half3>(colorBuffer, Int2(x1, y1));
    Half3 halfColor2 = Load2DHalf4<Half3>(colorBuffer, Int2(x2, y2));

    ushort mask1 = Load2DUshort1(materialBuffer, Int2(x1, y1));
    ushort mask2 = Load2DUshort1(materialBuffer, Int2(x2, y2));

    // Early Nan color check
    Float3 color1 = half3ToFloat3(halfColor1);
    Float3 color2 = half3ToFloat3(halfColor2);

    if (isnan(color1.x) || isnan(color1.y) || isnan(color1.z))
    {
        color1 = Float3(0.5f);
        halfColor1 = float3ToHalf3(color1);
    }
    if (isnan(color2.x) || isnan(color2.y) || isnan(color2.z))
    {
        color2 = Float3(0.5f);
        halfColor2 = float3ToHalf3(color2);
    }

    // global load 1
    if constexpr (EnableAtrousFilter)
    {
        sharedBuffer[id] =
        {
            halfColor1,
            mask1,
            Load2DHalf4<Half3>(normalBuffer, Int2(x1, y1)),
            Load2DHalf1<half>(depthBuffer, Int2(x1, y1))
        };
    }
    else
    {
        sharedBuffer[id] =
        {
            halfColor1,
            mask1,
        };
    }


    // global load 2
    if (id + threadCount < kernelCoverSize)
    {
        if constexpr (EnableAtrousFilter)
        {
            sharedBuffer[id + threadCount] =
            {
                halfColor2,
                mask2,
                Load2DHalf4<Half3>(normalBuffer, Int2(x2, y2)),
                Load2DHalf1<half>(depthBuffer, Int2(x2, y2))
            };
        }
        else
        {
            sharedBuffer[id + threadCount] =
            {
                halfColor2,
                mask2,
            };
        }
    }

    __syncthreads();

    // Return out of bound thread
    if (x >= size.x && y >= size.y)
    {
        return;
    }

    // load center
    AtrousLDS center = sharedBuffer[centerIdx];
    Float3 colorValue = half3ToFloat3(center.color);
    ushort maskValue = center.mask;
    float depthValue;
    Float3 normalValue;
    if constexpr (EnableAtrousFilter)
    {
        depthValue = __half2float(center.depth);
        normalValue = half3ToFloat3(center.normal);
    }

    // Return for sky pixel
    if (maskValue == SkyMaterialID)
    {
        return;
    }

    // Neighbor max/min
    Float3 neighbourMax = RgbToYcocg(colorValue);
    Float3 neighbourMin = RgbToYcocg(colorValue);

    // For softmax/min
    Float3 neighbourMax2 = RgbToYcocg(colorValue);
    Float3 neighbourMin2 = RgbToYcocg(colorValue);

    Float3 filteredColor = Float3(0);
    float weightSum = 0;

    int j;
    if (stride > 1)
    {
        j = frameNum % stride;
    }
    else
    {
        j = 0;
    }

#pragma unroll
    for (int i = 0; i < kernelSize / stride; ++i)
    {
        int xoffset = j % kernelDim;
        int yoffset = j / kernelDim;
        j += stride;

        AtrousLDS bufferReadTmp = sharedBuffer[threadIdx.x + xoffset + (threadIdx.y + yoffset) * kernelCoverDim];

        // get data
        Float3 color = half3ToFloat3(bufferReadTmp.color);

        if constexpr (EnableAtrousFilter)
        {
            ushort mask = bufferReadTmp.mask;
            float depth = __half2float(bufferReadTmp.depth);
            Float3 normal = half3ToFloat3(bufferReadTmp.normal);

            float weight = 1.0f;

            // normal diff factor
            weight *= powf(max(dot(normalValue, normal), 0.0f), params.temporal_denoise_sigma_normal);

            // depth diff fatcor
            float deltaDepth = (depthValue - depth) / params.temporal_denoise_sigma_depth;
            weight *= expf(-0.5f * deltaDepth * deltaDepth);

            // material mask diff factor
            weight *= (maskValue != mask) ? 1.0f / params.temporal_denoise_sigma_material : 1.0f;

            // gaussian filter weight
            if (kernelDim == 3)
            {
                weight *= GetGaussian3x3(xoffset + yoffset * kernelDim);
            }
            else if (kernelDim == 5)
            {
                weight *= GetGaussian5x5(xoffset + yoffset * kernelDim);
            }
            else if (kernelDim == 7)
            {
                weight *= GetGaussian7x7(xoffset + yoffset * kernelDim);
            }

            if (isnan(color.x) || isnan(color.y) || isnan(color.z))
            {
                color = Float3(0);
                weight = 0;
            }

            // accumulate
            filteredColor += color * weight;
            weightSum += weight;
        }

        if (isnan(color.x) || isnan(color.y) || isnan(color.z))
        {
            color = Float3(0.5f);
        }

        // max min
        Float3 neighbourColor = RgbToYcocg(color);

        neighbourMax = max3f(neighbourMax, neighbourColor);
        neighbourMin = min3f(neighbourMin, neighbourColor);

        if (useSoftMax && (abs(xoffset - kernelDim / 2) <= 1) && (abs(yoffset - kernelDim / 2) <= 1))
        {
            neighbourMax2 = max3f(neighbourMax2, neighbourColor);
            neighbourMin2 = min3f(neighbourMin2, neighbourColor);
        }
    }

    if (useSoftMax)
    {
        neighbourMax = (neighbourMax + neighbourMax2) / 2.0f;
        neighbourMin = (neighbourMin + neighbourMin2) / 2.0f;
    }

    if (weightSum > 0)
    {
        filteredColor /= weightSum;
    }
    else
    {
        filteredColor = Float3(0.5f);;
    }

    if (isnan(filteredColor.x) || isnan(filteredColor.y) || isnan(filteredColor.z))
    {
        filteredColor = Float3(0.5f);
    }

    // sample history color
    Float2 motionVec = Load2DHalf2(motionVectorBuffer, Int2(x, y)) - Float2(0.5f);
    Float2 uv = (Float2(x, y) + 0.5f) * (1.0f / Float2(size.x, size.y));
    Float2 historyUv = uv + motionVec;

    // history uv out of screen
    if (historyUv.x < 0 || historyUv.y < 0 || historyUv.x > 1.0 || historyUv.y > 1.0)
    {
        if (EnableAtrousFilter)
        {
            Store2DHalf4(Float4(filteredColor, 0), colorBuffer, Int2(x, y));
        }
    }
    else
    {
        Float3 outColor;

        if (!EnableAtrousFilter)
        {
            filteredColor = colorValue;
        }

        // sample history
        Float3 colorHistory = SampleBicubicSmoothStep<Load2DFuncHalf4<Float3>>(accumulateBuffer, historyUv, historySize);

        float accumulationBlendFactor = 1 / (float)accuCounter;

        if (accuCounter)

            if (accumulationBlendFactor < baseBlendingFactor)
            {
                outColor = colorValue * accumulationBlendFactor + colorHistory * (1.0f - accumulationBlendFactor);
            }
            else
            {
                // clamp history
                Float3 colorHistoryYcocg = RgbToYcocg(colorHistory);
                colorHistoryYcocg = clamp3f(colorHistoryYcocg, neighbourMin, neighbourMax);
                colorHistory = YcocgToRgb(colorHistoryYcocg);

                float lumaHistory;
                float lumaMin;
                float lumaMax;
                float lumaCurrent;

                if (enableAntiFlickering)
                {
                    lumaMin = neighbourMin.x;
                    lumaMax = neighbourMax.x;
                    lumaCurrent = RgbToYcocg(colorValue).x;
                }

                // load history material mask and depth for discard history
                float discardHistory = 0;
                Int2 historyIdx = Int2(floorf(historyUv.x * historySize.x), floorf(historyUv.y * historySize.y));

#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    Int2 offset(i % 2, i / 2);

                    Int2 historyNeighborIdx = historyIdx + offset;

                    float depthHistory = Load2DHalf1(depthHistoryBuffer, historyNeighborIdx);
                    ushort maskHistory = surf2Dread<ushort1>(materialHistoryBuffer, historyNeighborIdx.x * sizeof(ushort1), historyNeighborIdx.y, cudaBoundaryModeClamp).x;

                    float depthDiff = logf(depthHistory) - logf(depthValue);

                    discardHistory += (float)((maskValue != maskHistory) || (abs(depthDiff) > depthDiffLimit));
                }
                discardHistory /= 4.0f;

                colorHistory = colorHistory * (1.0f - discardHistory) + filteredColor * discardHistory;

                if (enableAntiFlickering)
                {
                    lumaHistory = RgbToYcocg(colorHistory).x;
                }

                // base blend factor
                float blendFactor = baseBlendingFactor;

                if (enableAntiFlickering)
                {
                    // anti flickering
                    float antiFlickeringFactor = clampf(0.5f * min(abs(lumaHistory - lumaMin), abs(lumaHistory - lumaMax)) / max3(lumaHistory, lumaCurrent, 1e-4f));
                    blendFactor *= (1.0f - antiFlickeringWeight) + antiFlickeringWeight * antiFlickeringFactor;
                }

                if (enableBlendUsingLumaHdrFactor)
                {
                    // weight with luma hdr factor
                    float weightA = blendFactor * max(0.0001f, 1.0f / (lumaCurrent + 4.0f));
                    float weightB = (1.0f - blendFactor) * max(0.0001f, 1.0f / (lumaHistory + 4.0f));
                    float weightSum = SafeDivide(1.0f, weightA + weightB);
                    weightA *= weightSum;
                    weightB *= weightSum;

                    outColor = colorValue * weightA + colorHistory * weightB;
                }
                else
                {
                    outColor = colorValue * blendFactor + colorHistory * (1.0f - blendFactor);
                }
            }

        // store to current
        // Store2DHalf4(Float4((float)maskValue / 10.0f, 0, 0, 0), colorBuffer, Int2(x, y));
        Store2DHalf4(Float4(outColor, 0), colorBuffer, Int2(x, y));
    }
}

}