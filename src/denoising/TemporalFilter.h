#include "shaders/LinearMath.h"
#include "shaders/Sampler.h"
#include "util/Gaussian.h"
#include "core/GlobalSettings.h"
#include "shaders/ShaderDebugUtils.h"

namespace jazzfusion
{

// __global__ void TemporalDenoiseDepthNormalMat(
//     int accuCounter,
//     SurfObj   depthBuffer,
//     SurfObj   depthHistoryBuffer,
//     SurfObj   materialBuffer,
//     SurfObj   materialHistoryBuffer,
//     SurfObj   normalBuffer,
//     SurfObj   normalHistoryBuffer,
//     Int2    size)
// {
//     int x = threadIdx.x + blockIdx.x * 8;
//     int y = threadIdx.y + blockIdx.y * 8;

//     if (x >= size.x || y >= size.y) return;

//     Int2 idx(x, y);

//     // Get first hit material
//     ushort mat = Load2DUshort1(materialBuffer, idx);
//     uint firstMat = (uint)mat;
//     while (firstMat > NUM_MATERIALS * 2)
//     {
//         firstMat /= NUM_MATERIALS;
//     }
//     firstMat -= NUM_MATERIALS;

//     // Get final hit material
//     uint finalMat = (uint)maskValue % NUM_MATERIALS;
//     if (firstMat != INDEX_BSDF_SPECULAR_REFLECTION_TRANSMISSION)
//     {
//         surf2Dwrite(surf2Dread<ushort1>(depthBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp), depthHistoryBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp);
//         surf2Dwrite(surf2Dread<ushort1>(materialBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp), materialHistoryBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp);
//         surf2Dwrite(surf2Dread<ushort4>(normalBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp), normalHistoryBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);
//         return;
//     }

//     if (accuCounter == 1)
//     {
//         surf2Dwrite(surf2Dread<ushort1>(depthBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp), depthHistoryBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp);
//         surf2Dwrite(surf2Dread<ushort1>(materialBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp), materialHistoryBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp);
//         surf2Dwrite(surf2Dread<ushort4>(normalBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp), normalHistoryBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);
//     }
//     else
//     {
//         float factor = 1.0f / (float)accuCounter;

//         Float3 normal = Load2DHalf4(normalBuffer, idx).xyz;
//         Float3 normalHistory = Load2DHalf4(normalHistoryBuffer, idx).xyz;

//         float depth = Load2DHalf1(depthBuffer, idx);
//         float depthHistory = Load2DHalf1(depthHistoryBuffer, idx);

//         ushort mat = Load2DUshort1(materialBuffer, idx);
//         ushort matHistory = Load2DUshort1(materialHistoryBuffer, idx);

//         Float3 outNormal = slerp3f(normalHistory, normal, factor);
//         float outdepth = lerpf(depthHistory, depth, factor);
//         ushort outMat = factor == 1.0f ? mat : min(mat, matHistory);

//         // Store2DHalf4(Float4(outNormal, 0.0f), normalBuffer, idx);
//         Store2DHalf1(outdepth, depthBuffer, idx);
//         Store2DUshort1(outMat, materialBuffer, idx);

//         // Store2DHalf4(Float4(outNormal, 0.0f), normalHistoryBuffer, idx);
//         Store2DHalf1(outdepth, depthHistoryBuffer, idx);
//         Store2DUshort1(outMat, materialHistoryBuffer, idx);
//     }
// }

// INL_DEVICE void CopyToHistoryColorBufferImpl(
//     int accuCounter,
//     Int2 idx,

//     Float3 outHistoryColor,
//     SurfObj historyColorBuffer

//     float   depth,
//     SurfObj   depthHistoryBuffer,

//     ushort   material,
//     SurfObj   materialHistoryBuffer,

//     Float3   normal,
//     SurfObj   normalHistoryBuffer
// )
// {
//     // // surf2Dwrite(surf2Dread<ushort4>(colorBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp), historyColorBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);

//     // // surf2Dwrite(surf2Dread<ushort1>(depthFrontBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp), historyDepthFrontBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp);
//     // // surf2Dwrite(surf2Dread<ushort4>(normalFrontBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp), historyNormalFrontBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);
//     // //surf2Dwrite(surf2Dread<ushort1>(materialFrontBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp), historyMaterialFrontBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp);

//     // // surf2Dwrite(surf2Dread<ushort1>(depthBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp), depthHistoryBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp);
//     // // surf2Dwrite(surf2Dread<ushort1>(materialBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp), materialHistoryBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp);
//     // // surf2Dwrite(surf2Dread<ushort4>(normalBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp), normalHistoryBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);
//     // // surf2Dwrite(surf2Dread<ushort4>(albedoBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp), historyAlbedoBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);

//     // float factor = 1.0f / (float)accuCounter;

//     // // Float3 albedo = Load2DHalf4(albedoBuffer, idx).xyz;
//     // // Float3 albedoHistory = Load2DHalf4(historyAlbedoBuffer, idx).xyz;

//     // Float3 normal = Load2DHalf4(normalBuffer, idx).xyz;
//     // Float3 normalHistory = Load2DHalf4(normalHistoryBuffer, idx).xyz;

//     // float depth = Load2DHalf1(depthBuffer, idx);
//     // float depthHistory = Load2DHalf1(depthHistoryBuffer, idx);

//     // ushort mat = Load2DUshort1(materialBuffer, idx);
//     // ushort matHistory = Load2DUshort1(materialHistoryBuffer, idx);

//     // // Float3 outAlbedo = lerp3f(albedoHistory, albedo, factor);
//     // Float3 outNormal = slerp3f(normalHistory, normal, factor);
//     // float outdepth = lerpf(depthHistory, depth, factor);
//     // ushort outMat = factor == 1.0f ? mat : min(mat, matHistory);

//     // // Store2DHalf4(Float4(outAlbedo, 0.0f), historyAlbedoBuffer, idx);
//     // Store2DHalf4(Float4(outNormal, 0.0f), normalHistoryBuffer, idx);
//     // Store2DHalf1(outdepth, depthHistoryBuffer, idx);
//     // Store2DUshort1(outMat, materialHistoryBuffer, idx);

//     Store2DHalf4(Float4(outHistoryColor, 1.0f), historyColorBuffer, idx);


//     float factor = 1.0f / (float)accuCounter;

//     Float3 normalHistory = Load2DHalf4(normalHistoryBuffer, idx).xyz;

//     // float depth = Load2DHalf1(depthBuffer, idx);
//     float depthHistory = Load2DHalf1(depthHistoryBuffer, idx);

//     // ushort mat = Load2DUshort1(materialBuffer, idx);
//     ushort matHistory = Load2DUshort1(materialHistoryBuffer, idx);

//     Float3 outNormal = normalize(lerp3f(normalHistory, normal, factor));
//     float outdepth = lerpf(depthHistory, depth, factor);
//     ushort outMat = factor == 1.0f ? mat : min(mat, matHistory);

//     // Store2DHalf4(Float4(outNormal, 0.0f), normalBuffer, idx);
//     Store2DHalf1(outdepth, depthBuffer, idx);
//     Store2DUshort1(outMat, materialBuffer, idx);

//     // Store2DHalf4(Float4(outNormal, 0.0f), normalHistoryBuffer, idx);
//     Store2DHalf1(outdepth, depthHistoryBuffer, idx);
//     Store2DUshort1(outMat, materialHistoryBuffer, idx);
// }

__global__ void CopyToHistoryColorBuffer(
    SurfObj   colorBuffer,
    SurfObj   historyColorBuffer,

    SurfObj   depthBuffer,
    SurfObj   depthHistoryBuffer,

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
    surf2Dwrite(surf2Dread<ushort1>(depthBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp), depthHistoryBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp);
    surf2Dwrite(surf2Dread<ushort1>(materialBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp), materialHistoryBuffer, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp);
    surf2Dwrite(surf2Dread<ushort4>(normalBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp), normalHistoryBuffer, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);
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

template<bool IsFirstPass>
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
    Half3 albedo;
    half depth;
};

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

template<bool IsFirstPass, bool IsFirstAccumulation>
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
    constexpr int stride = 1;
    bool useSoftMax = params.temporal_denoise_use_softmax;
    // float baseBlendingFactor = params.temporal_denoise_baseBlendingFactor;
    float antiFlickeringWeight = 0.8f;
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

    __shared__ AtrousLDS<IsFirstPass> sharedBuffer[kernelCoverSize];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    Float3 outColor;
    Float3 outAlbedo;
    Float3 outNormal;
    float outDepth;
    ushort outMaterial;

    ushort maskValue;

    if constexpr (IsFirstAccumulation)
    {
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
        if constexpr (IsFirstPass)
        {
            sharedBuffer[id] =
            {
                halfColor1,
                mask1,
                Load2DHalf4<Half3>(albedoBuffer, Int2(x1, y1)),
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
            if constexpr (IsFirstPass)
            {
                sharedBuffer[id + threadCount] =
                {
                    halfColor2,
                    mask2,
                    Load2DHalf4<Half3>(albedoBuffer, Int2(x2, y2)),
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
        maskValue = center.mask;
        float depthValue;
        Float3 albedoValue;
        if constexpr (IsFirstPass)
        {
            depthValue = __half2float(center.depth);
            albedoValue = half3ToFloat3(center.albedo);
        }

        // outColor = colorValue;
        // outAlbedo = albedoValue;
        // outMaterial = maskValue;
        // if constexpr (!IsFirstPass)
        // {
        //     outNormal = Load2DHalf4(normalBuffer, Int2(x, y));
        //     outdepth = Load2DHalf1(depthBuffer, Int2(x, y));
        // }

        // Return for sky pixel
        if (maskValue == SkyMaterialID)
        {
            return;
        }

        // Neighbor max/min
        Float3 neighbourMax = RgbToYcocg(colorValue);
        Float3 neighbourMin = RgbToYcocg(colorValue);
        Float3 neighbourMax2 = RgbToYcocg(colorValue);
        Float3 neighbourMin2 = RgbToYcocg(colorValue);

        Float3 neighbourAlbedoMax = RgbToYcocg(albedoValue);
        Float3 neighbourAlbedoMin = RgbToYcocg(albedoValue);
        Float3 neighbourAlbedoMax2 = RgbToYcocg(albedoValue);
        Float3 neighbourAlbedoMin2 = RgbToYcocg(albedoValue);

        Float3 filteredColor = Float3(0);
        // Float3 filteredAlbedo = Float3(0);
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
            Float3 albedo;

            if constexpr (IsFirstPass)
            {
                albedo = half3ToFloat3(bufferReadTmp.albedo);

                ushort mask = bufferReadTmp.mask;
                float depth = __half2float(bufferReadTmp.depth);
                // Float3 normal = half3ToFloat3(bufferReadTmp.normal);

                float weight = 1.0f;

                // normal diff factor
                // weight *= powf(max(dot(normalValue, normal), 0.0f), params.temporal_denoise_sigma_normal);

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
                // filteredAlbedo += albedo * weight;
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

            if constexpr (IsFirstPass)
            {
                Float3 neighbourAlbedo = RgbToYcocg(albedo);
                neighbourAlbedoMax = max3f(neighbourAlbedoMax, neighbourAlbedo);
                neighbourAlbedoMin = min3f(neighbourAlbedoMin, neighbourAlbedo);
                if (useSoftMax && (abs(xoffset - kernelDim / 2) <= 1) && (abs(yoffset - kernelDim / 2) <= 1))
                {
                    neighbourAlbedoMax2 = max3f(neighbourAlbedoMax2, neighbourAlbedo);
                    neighbourAlbedoMin2 = min3f(neighbourAlbedoMin2, neighbourAlbedo);
                }
            }
        }

        if (useSoftMax)
        {
            neighbourMax = (neighbourMax + neighbourMax2) / 2.0f;
            neighbourMin = (neighbourMin + neighbourMin2) / 2.0f;

            if constexpr (IsFirstPass)
            {
                neighbourAlbedoMax = (neighbourAlbedoMax + neighbourAlbedoMax2) / 2.0f;
                neighbourAlbedoMin = (neighbourAlbedoMin + neighbourAlbedoMin2) / 2.0f;
            }
        }

        if (weightSum > 0)
        {
            filteredColor /= weightSum;
            // filteredAlbedo /= weightSum;
        }
        else
        {
            filteredColor = Float3(0.5f);
            // filteredAlbedo = Float3(0.5f);
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
        if (historyUv.x < 0 || historyUv.y < 0 || historyUv.x > 1.0f || historyUv.y > 1.0f)
        {
            if constexpr (IsFirstPass)
            {
                outColor = filteredColor;
                outAlbedo = albedoValue;
            }
            else
            {
                outColor = colorValue;
                outDepth = depthValue;
                outNormal = Load2DHalf4(normalBuffer, Int2(x, y)).xyz;
                outDepth = Load2DHalf1(depthBuffer, Int2(x, y));
                outMaterial = Load2DUshort1(materialBuffer, Int2(x, y));
            }
        }
        else
        {
            if constexpr (!IsFirstPass)
            {
                filteredColor = colorValue;
            }

            // sample history
            Float3 colorHistory;
            Float3 albedoHistory;
            if constexpr (IsFirstPass)
            {
                colorHistory = SampleBicubicSmoothStep<Load2DFuncHalf4<Float3>>(accumulateBuffer, historyUv, historySize);
                albedoHistory = SampleBicubicSmoothStep<Load2DFuncHalf4<Float3>>(historyAlbedoBuffer, historyUv, historySize);
            }
            else
            {
                colorHistory = SampleBicubicSmoothStep<Load2DFuncHalf4<Float3>>(historyColorBuffer, historyUv, historySize);
            }

            // Clamp history
            Float3 colorHistoryYcocg = RgbToYcocg(colorHistory);
            colorHistoryYcocg = clamp3f(colorHistoryYcocg, neighbourMin, neighbourMax);
            Float3 clampedColorHistory = YcocgToRgb(colorHistoryYcocg);

            Float3 albedoHistoryYcocg;
            Float3 clampedAlbedoHistory;
            if constexpr (IsFirstPass)
            {
                albedoHistoryYcocg = RgbToYcocg(albedoHistory);
                albedoHistoryYcocg = clamp3f(albedoHistoryYcocg, neighbourAlbedoMin, neighbourAlbedoMax);
                clampedAlbedoHistory = YcocgToRgb(albedoHistoryYcocg);
            }

            // Discard history factor. 0 = no discard, good history sample. 1 = full discard, total bad history sample.
            float discardHistoryFactor = 0.0f;

            Int2 historyIdx = Int2(floorf(historyUv.x * historySize.x), floorf(historyUv.y * historySize.y));
            // float historyDepth = 0;
            // ushort historyMaterial = 65535;

#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                Int2 offset(i % 2, i / 2);

                Int2 historyNeighborIdx = historyIdx + offset;

                float depthHistory = Load2DHalf1(depthHistoryBuffer, historyNeighborIdx);
                ushort maskHistory = surf2Dread<ushort1>(materialHistoryBuffer, historyNeighborIdx.x * sizeof(ushort1), historyNeighborIdx.y, cudaBoundaryModeClamp).x;

                float depthDiff = logf(depthHistory) - logf(depthValue);

                discardHistoryFactor += (float)((maskValue != maskHistory) || (abs(depthDiff) > depthDiffLimit));

                // historyDepth += depthHistory;
                // historyMaterial = min(historyMaterial, maskHistory);

            }
            discardHistoryFactor /= 4.0f;
            // historyDepth /= 4.0f;

            // Float3 historyNormal;
            // if constexpr (!IsFirstPass)
            // {
            //     historyNormal = SampleBicubicSmoothStep<Load2DFuncHalf4<Float3>>(normalHistoryBuffer, historyUv, historySize);
            // }

            float blendFactor = 1.0f / 2.0f;
            Float3 blendedColorHistory = clampedColorHistory * (1.0f - discardHistoryFactor) + filteredColor * discardHistoryFactor;

            Float3 blendedAlbedoHistory;
            if constexpr (IsFirstPass)
            {
                // blendedAlbedoHistory = clampedAlbedoHistory * (1.0f - discardHistoryFactor) + filteredAlbedo * discardHistoryFactor;
                blendedAlbedoHistory = clampedAlbedoHistory * (1.0f - discardHistoryFactor) + albedoValue * discardHistoryFactor;
            }

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

            if constexpr (enableAntiFlickering)
            {
                // anti flickering
                float antiFlickeringFactor = clampf(0.5f * min(abs(lumaHistory - lumaMin), abs(lumaHistory - lumaMax)) / max3(lumaHistory, lumaCurrent, 1e-4f));
                blendFactor *= (1.0f - antiFlickeringWeight) + antiFlickeringWeight * antiFlickeringFactor;
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
                outColor = colorValue * blendFactor + blendedColorHistory * (1.0f - blendFactor);
            }

            if constexpr (IsFirstPass)
            {
                outAlbedo = lerp3f(blendedAlbedoHistory, albedoValue, 0.5f);
            }

            if constexpr (!IsFirstPass)
            {
                // outNormal = Load2DHalf4(normalBuffer, Int2(x, y)).xyz;
                outDepth = Load2DHalf1(depthBuffer, Int2(x, y));
                outMaterial = maskValue;
            }
        }
    }
    else // Not first accumulation, progressive rendering
    {
        if (x >= size.x && y >= size.y)
        {
            return;
        }


        // else
        // {
        //     maskValue = Load2DUshort1(materialBuffer, Int2(x, y));
        //     ushort historyMaterial = Load2DUshort1(materialHistoryBuffer, Int2(x, y));
        //     outMaterial = (maskValue == SkyMaterialID) ? historyMaterial : maskValue;
        // }

        maskValue = Load2DUshort1(materialBuffer, Int2(x, y));
        ushort historyMaterial = Load2DUshort1(materialHistoryBuffer, Int2(x, y));
        outMaterial = (maskValue == SkyMaterialID) ? historyMaterial : maskValue;

        if (accuCounter == 2)
        {
            outMaterial = maskValue;
        }

        // if (outMaterial == SkyMaterialID)
        // {
        //     return;
        // }

        float blendFactor = min(0.5f, 1.0f / (float)accuCounter);

        outColor = Load2DHalf4(colorBuffer, Int2(x, y)).xyz;

        Float3 colorHistory;
        Float3 albedoHistory;
        if constexpr (IsFirstPass)
        {
            colorHistory = Load2DHalf4(accumulateBuffer, Int2(x, y)).xyz;
            outAlbedo = Load2DHalf4(albedoBuffer, Int2(x, y)).xyz;
            albedoHistory = Load2DHalf4(historyAlbedoBuffer, Int2(x, y)).xyz;
        }
        else
        {
            colorHistory = Load2DHalf4(historyColorBuffer, Int2(x, y)).xyz;

            // outNormal = Load2DHalf4(normalBuffer, Int2(x, y)).xyz;
            outDepth = Load2DHalf1(depthBuffer, Int2(x, y));
        }


        outColor = outColor * blendFactor + colorHistory * (1.0f - blendFactor);

        if constexpr (IsFirstPass)
        {
            outAlbedo = outAlbedo * blendFactor + albedoHistory * (1.0f - blendFactor);
        }
        else
        {
            // Float3 historyNormal = Load2DHalf4(normalHistoryBuffer, Int2(x, y)).xyz;
            float historyDepth = Load2DHalf1(depthHistoryBuffer, Int2(x, y));

            // outNormal = outNormal * blendFactor + historyNormal * (1.0f - blendFactor);
            outDepth = outDepth * blendFactor + historyDepth * (1.0f - blendFactor);
        }
    }

    // Write to current buffer
    if (outMaterial != SkyMaterialID)
    {
        Store2DHalf4(Float4(outColor, 0), colorBuffer, Int2(x, y));
    }

    if constexpr (IsFirstPass)
    {
        Store2DHalf4(Float4(outAlbedo, 0), albedoBuffer, Int2(x, y));
    }

    // Write to history buffer
    if constexpr (IsFirstPass)
    {
        // Store2DHalf4(Float4(outColor, 0.0f), accumulateBuffer, Int2(x, y));
        // Store2DHalf4(Float4(outAlbedo, 0.0f), historyAlbedoBuffer, Int2(x, y));
    }
    else
    {
        Store2DHalf4(Float4(outColor, 0.0f), historyColorBuffer, Int2(x, y));
        // Store2DHalf4(Float4(outNormal, 0.0f), normalHistoryBuffer, Int2(x, y));
        Store2DHalf1(outDepth, depthHistoryBuffer, Int2(x, y));
        Store2DUshort1(outMaterial, materialHistoryBuffer, Int2(x, y));
    }
}

}