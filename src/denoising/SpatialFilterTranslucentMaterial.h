#include "shaders/LinearMath.h"
#include "shaders/Sampler.h"
#include "util/Gaussian.h"
#include "core/GlobalSettings.h"
#include "shaders/ShaderDebugUtils.h"

namespace jazzfusion
{

__global__ void SpatialFilter2(
    int frameNum,
    int accuCounter,
    SurfObj colorBuffer,
    SurfObj materialBuffer,
    SurfObj normalBuffer,
    SurfObj depthBuffer,
    SurfObj noiseLevelBuffer16,
    DenoisingParams params,
    Int2    size)
{
    int x = threadIdx.x + blockIdx.x * 16;
    int y = threadIdx.y + blockIdx.y * 16;

    if (ENABLE_DENOISING_NOISE_CALCULATION)
    {
        float noiseLevel = Load2DHalf1(noiseLevelBuffer16, Int2(blockIdx.x, blockIdx.y));
        if (noiseLevel < params.noise_threshold_local)
        {
            return;
        }
    }

    struct AtrousLDS
    {
        Half3 color;
        ushort mask;
        Half3 normal;
        half depth;
    };

    constexpr int blockdim = 16;
    constexpr int kernelRadius = 2;

    constexpr int threadCount = blockdim * blockdim;

    constexpr int kernelCoverDim = blockdim + kernelRadius * 2;
    constexpr int kernelCoverSize = kernelCoverDim * kernelCoverDim;

    constexpr int kernelDim = kernelRadius * 2 + 1;
    constexpr int kernelSize = kernelDim * kernelDim;

    int centerIdx = threadIdx.x + kernelRadius + (threadIdx.y + kernelRadius) * kernelCoverDim;

    __shared__ AtrousLDS sharedBuffer[kernelCoverSize];

    // calculate address
    int id = (threadIdx.x + threadIdx.y * blockdim);
    int x1 = blockIdx.x * blockdim - kernelRadius + id % kernelCoverDim;
    int y1 = blockIdx.y * blockdim - kernelRadius + id / kernelCoverDim;
    int x2 = blockIdx.x * blockdim - kernelRadius + (id + threadCount) % kernelCoverDim;
    int y2 = blockIdx.y * blockdim - kernelRadius + (id + threadCount) / kernelCoverDim;

    // global load 1
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

    sharedBuffer[id] =
    {
        halfColor1,
        mask1,
        Load2DHalf4<Half3>(normalBuffer, Int2(x1, y1)),
        Load2DHalf1<half>(depthBuffer, Int2(x1, y1))
    };

    // global load 2
    if (id + threadCount < kernelCoverSize)
    {
        sharedBuffer[id + threadCount] =
        {
            halfColor2,
            mask2,
            Load2DHalf4<Half3>(normalBuffer, Int2(x2, y2)),
            Load2DHalf1<half>(depthBuffer, Int2(x2, y2))
        };
    }

    __syncthreads();

    if (x >= size.x && y >= size.y) return;

    // load center
    AtrousLDS center = sharedBuffer[centerIdx];
    Float3 colorValue = half3ToFloat3(center.color);
    float depthValue = __half2float(center.depth);
    Float3 normalValue = half3ToFloat3(center.normal);
    ushort maskValue = center.mask;

    // Get first hit material
    uint firstMat = (uint)maskValue;
    while (firstMat > NUM_MATERIALS * 2)
    {
        firstMat /= NUM_MATERIALS;
    }
    firstMat -= NUM_MATERIALS;

    // Get final hit material
    uint finalMat = (uint)maskValue % NUM_MATERIALS;

    if (firstMat != INDEX_BSDF_SPECULAR_REFLECTION_TRANSMISSION)
    {
        return;
    }

    // -------------------------------- atrous filter --------------------------------
    Float3 sumOfColor = Float3(0.0f);
    float sumOfWeight = 0;

#if 0
#pragma unroll
    for (int i = 0; i < SAMPLE_KERNEL_SAMPLE_PER_FRAME; i += 1)
    {
        int kernelSelect = frameNum % SAMPLE_KERNEL_FRAME_COUNT;
        int kernelSelectIdx = kernelSelect * SAMPLE_KERNEL_SAMPLE_PER_FRAME + i;

        int xoffset = SampleKernel3d[kernelSelectIdx][0];
        int yoffset = SampleKernel3d[kernelSelectIdx][1];
#elif 1
#pragma unroll
    for (int i = 0; i < kernelSize; i += 1)
    {
        int xoffset = i % kernelDim;
        int yoffset = i / kernelDim;
#elif 0
    constexpr int stride = 2;
    int j = frameNum % stride;
#pragma unroll
    for (int i = 0; i < kernelSize / stride; ++i)
    {
        int xoffset = j % kernelDim;
        int yoffset = j / kernelDim;
        j += stride;
#endif
        AtrousLDS bufferReadTmp = sharedBuffer[threadIdx.x + xoffset + (threadIdx.y + yoffset) * kernelCoverDim];

        // get data
        Float3 color = half3ToFloat3(bufferReadTmp.color);
        float depth = __half2float(bufferReadTmp.depth);
        Float3 normal = half3ToFloat3(bufferReadTmp.normal);
        ushort mask = bufferReadTmp.mask;

        float weight = 1.0f;

        // normal diff factor
        weight *= powf(max(dot(normalValue, normal), 0.0001f), params.local_denoise_sigma_normal);

        // depth diff fatcor
        float deltaDepth = (depthValue - depth) / params.local_denoise_sigma_depth;
        weight *= expf(-0.5f * deltaDepth * deltaDepth);

        // material mask diff factor
        // weight *= (maskValue != mask) ? 1.0f / params.local_denoise_sigma_material : 1.0f;

        // gaussian filter weight
        weight *= GetGaussian5x5(xoffset + yoffset * kernelDim);

        if (isnan(color.x) || isnan(color.y) || isnan(color.z))
        {
            color = Float3(0);
            weight = 0;
        }

        // accumulate
        sumOfColor += color * weight;
        sumOfWeight += weight;
    }

    Float3 finalColor;

    // final color
    if (sumOfWeight == 0)
    {
        finalColor = Float3(0.5f);
    }
    else
    {
        finalColor = sumOfColor / sumOfWeight;
    }

    if (isnan(finalColor.x) || isnan(finalColor.y) || isnan(finalColor.z))
    {
        finalColor = Float3(0.5f);
    }

    // store to current
    Store2DHalf4(Float4(finalColor, 0), colorBuffer, Int2(x, y));
}

template<int KernelStride>
__global__ void SpatialWideFilter2(
    int frameNum,
    int accuCounter,
    SurfObj colorBuffer,
    SurfObj materialBuffer,
    SurfObj normalBuffer,
    SurfObj depthBuffer,
    SurfObj noiseLevelBuffer,
    DenoisingParams params,
    Int2    size)
{
    if (ENABLE_DENOISING_NOISE_CALCULATION)
    {
        float noiseLevel = Load2DHalf1(noiseLevelBuffer, Int2(blockIdx.x, blockIdx.y));
        if (noiseLevel < params.noise_threshold_large)
        {
            return;
        }
    }

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= size.x && y >= size.y) return;

    Float3 normalValue = Load2DHalf4(normalBuffer, Int2(x, y)).xyz;
    Float3 colorValue = Load2DHalf4(colorBuffer, Int2(x, y)).xyz;
    ushort maskValue = Load2DUshort1(materialBuffer, Int2(x, y));
    float depthValue = Load2DHalf1(depthBuffer, Int2(x, y));

    if (isnan(colorValue.x) || isnan(colorValue.y) || isnan(colorValue.z))
    {
        colorValue = Float3(0.5f);
    }

    // Get first hit material
    uint firstMat = (uint)maskValue;
    while (firstMat > NUM_MATERIALS * 2)
    {
        firstMat /= NUM_MATERIALS;
    }
    firstMat -= NUM_MATERIALS;

    // Get final hit material
    uint finalMat = (uint)maskValue % NUM_MATERIALS;

    bool isGlass = (firstMat == INDEX_BSDF_SPECULAR_REFLECTION_TRANSMISSION);

    if (firstMat != INDEX_BSDF_SPECULAR_REFLECTION_TRANSMISSION)
    {
        return;
    }

    Float3 sumOfColor{ 0 };
    float sumOfWeight = 0;

    constexpr int stride = 1;
    int k;
    if (stride > 1)
    {
        k = frameNum % stride;
    }
    else
    {
        k = 0;
    }

#pragma unroll
    for (int loopIdx = 0; loopIdx < 25 / stride; ++loopIdx)
    {
        int i = k % 5;
        int j = k / 5;

        k += stride;

        Int2 loadIdx(x + (i - 2) * KernelStride, y + (j - 2) * KernelStride);

        Float3 normal = Load2DHalf4(normalBuffer, loadIdx).xyz;
        Float3 color = Load2DHalf4(colorBuffer, loadIdx).xyz;
        ushort mask = surf2Dread<ushort1>(materialBuffer, loadIdx.x * sizeof(ushort), loadIdx.y, cudaBoundaryModeClamp).x;
        float depth = Load2DHalf1(depthBuffer, loadIdx);

        float weight = 1.0f;

        // normal diff factor
        weight *= powf(max(dot(normalValue, normal), 0.0f), params.large_denoise_sigma_normal);

        // depth diff fatcor
        float deltaDepth = (depthValue - depth) / params.large_denoise_sigma_depth;
        weight *= expf(-0.5f * deltaDepth * deltaDepth);

        // material mask diff factor
        // if (!isGlass)
        {
            // weight *= (maskValue != mask) ? 1.0f / params.large_denoise_sigma_material : 1.0f;
        }

        // gaussian filter weight
        weight *= GetGaussian5x5(i + j * 5);

        if (isnan(color.x) || isnan(color.y) || isnan(color.z))
        {
            color = Float3(0);
            weight = 0;
        }

        // accumulate
        sumOfColor += color * weight;
        sumOfWeight += weight;
    }


    // final color
    Float3 finalColor;
    if (sumOfWeight == 0)
    {
        finalColor = Float3(0.5f);
    }
    else
    {
        finalColor = sumOfColor / sumOfWeight;
    }

    if (isnan(finalColor.x) || isnan(finalColor.y) || isnan(finalColor.z))
    {
        finalColor = Float3(0.5f);
    }


    // store to current
    Store2DHalf4(Float4(finalColor, 0), colorBuffer, Int2(x, y));
}

__global__ void BilateralFilter2(
    int             frameNum,
    int             accuCounter,
    SurfObj         colorBuffer,
    SurfObj         materialBuffer,
    SurfObj         depthBuffer,
    DenoisingParams params,
    Int2            size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Settings
    constexpr int blockdim = 16;
    constexpr int kernelRadius = 2;

    // Calculations of kernel dim
    constexpr int threadCount = blockdim * blockdim;

    constexpr int kernelCoverDim = blockdim + kernelRadius * 2;
    constexpr int kernelCoverSize = kernelCoverDim * kernelCoverDim;

    constexpr int kernelDim = kernelRadius * 2 + 1;
    constexpr int kernelSize = kernelDim * kernelDim;

    int centerIdx = threadIdx.x + kernelRadius + (threadIdx.y + kernelRadius) * kernelCoverDim;

    __shared__ Float4 sharedBuffer[kernelCoverSize];

    // calculate address
    int id = (threadIdx.x + threadIdx.y * blockdim);

    int x1 = blockIdx.x * blockdim - kernelRadius + id % kernelCoverDim;
    int y1 = blockIdx.y * blockdim - kernelRadius + id / kernelCoverDim;
    int x2 = blockIdx.x * blockdim - kernelRadius + (id + threadCount) % kernelCoverDim;
    int y2 = blockIdx.y * blockdim - kernelRadius + (id + threadCount) / kernelCoverDim;

    Float3 color1 = Load2DHalf4(colorBuffer, Int2(x1, y1)).xyz;
    Float3 color2 = Load2DHalf4(colorBuffer, Int2(x2, y2)).xyz;

    // float depth1 = Load2DHalf1(depthBuffer, Int2(x1, y1));
    // float depth2 = Load2DHalf1(depthBuffer, Int2(x2, y2));
    float depth1 = 0;
    float depth2 = 0;

    // global load 1
    sharedBuffer[id] = Float4(color1, depth1);

    // global load 2
    if (id + threadCount < kernelCoverSize)
    {
        sharedBuffer[id + threadCount] = Float4(color2, depth2);
    }

    __syncthreads();

    // Return out of bound thread
    if (x >= size.x && y >= size.y)
    {
        return;
    }

    // load center
    Float3 centerColor = sharedBuffer[centerIdx].xyz;
    // float centerDepth = sharedBuffer[centerIdx].w;

    // Return for sky pixel
    ushort maskValue = Load2DUshort1(materialBuffer, Int2(x, y));
    uint firstMat = (uint)maskValue;
    while (firstMat > NUM_MATERIALS * 2)
    {
        firstMat /= NUM_MATERIALS;
    }
    firstMat -= NUM_MATERIALS;

    // Get final hit material
    uint finalMat = (uint)maskValue % NUM_MATERIALS;
    bool isGlass = (firstMat == INDEX_BSDF_SPECULAR_REFLECTION_TRANSMISSION);

    if (firstMat != INDEX_BSDF_SPECULAR_REFLECTION_TRANSMISSION)
    {
        return;
    }

    // Run the filter
    Float3 filteredColor = Float3(0);
    float weightSum = 0;

#pragma unroll
    for (int i = 0; i < kernelSize; ++i)
    {
        int xoffset = i % kernelDim;
        int yoffset = i / kernelDim;

        Float4 colorDepth = sharedBuffer[threadIdx.x + xoffset + (threadIdx.y + yoffset) * kernelCoverDim];
        Float3 color = colorDepth.xyz;
        // float depth = colorDepth.w;

        float weight = 1.0f;

        // depth diff fatcor
        // float deltaDepth = (centerDepth - depth) / params.local_denoise_sigma_depth;
        // weight *= expf(-0.5f * deltaDepth * deltaDepth);

        // gaussian filter weight
        if constexpr (kernelDim == 3)
        {
            weight *= GetGaussian3x3(xoffset + yoffset * kernelDim);
        }
        else if constexpr (kernelDim == 5)
        {
            weight *= GetGaussian5x5(xoffset + yoffset * kernelDim);
        }
        else if constexpr (kernelDim == 7)
        {
            weight *= GetGaussian7x7(xoffset + yoffset * kernelDim);
        }

        float closeness = 1.0f - min(distance(color, centerColor) / 1.732f, 1.0f);
        weight *= closeness;

        // accumulate
        filteredColor += color * weight;
        weightSum += weight;
    }

    if (weightSum > 0)
    {
        filteredColor /= weightSum;
    }
    else
    {
        filteredColor = Float3(0.5f);
    }

    Store2DHalf4(Float4(filteredColor, 1.0f), colorBuffer, Int2(x, y));
}


template<int KernelStride>
__global__ void BilateralFilterWide2(
    int             frameNum,
    int             accuCounter,
    SurfObj         colorBuffer,
    SurfObj         materialBuffer,
    SurfObj         depthBuffer,
    DenoisingParams params,
    Int2            size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // Return out of bound thread
    if (x >= size.x && y >= size.y)
    {
        return;
    }

    Float3 centerColor = Load2DHalf4(colorBuffer, Int2(x, y)).xyz;
    // float centerDepth = Load2DHalf1(depthBuffer, Int2(x, y));

    // Return for sky pixel
    ushort maskValue = Load2DUshort1(materialBuffer, Int2(x, y));
    uint firstMat = (uint)maskValue;
    while (firstMat > NUM_MATERIALS * 2)
    {
        firstMat /= NUM_MATERIALS;
    }
    firstMat -= NUM_MATERIALS;

    // Get final hit material
    uint finalMat = (uint)maskValue % NUM_MATERIALS;
    bool isGlass = (firstMat == INDEX_BSDF_SPECULAR_REFLECTION_TRANSMISSION);

    if (firstMat != INDEX_BSDF_SPECULAR_REFLECTION_TRANSMISSION)
    {
        return;
    }

    // Run the filter
    Float3 filteredColor = Float3(0);
    float weightSum = 0;

#pragma unroll
    for (int k = 0; k < 25; ++k)
    {
        int i = k % 5;
        int j = k / 5;

        Int2 loadIdx(x + (i - 2) * KernelStride, y + (j - 2) * KernelStride);

        Float3 color = Load2DHalf4(colorBuffer, loadIdx).xyz;
        // float depth = Load2DHalf1(depthBuffer, loadIdx);

        float weight = 1.0f;

        // depth diff fatcor
        // float deltaDepth = (centerDepth - depth) / params.large_denoise_sigma_depth;
        // weight *= expf(-0.5f * deltaDepth * deltaDepth);

        // gaussian filter weight
        weight *= GetGaussian5x5(i + j * 5);

        float closeness = 1.0f - min(distance(color, centerColor) / 1.732f, 1.0f);
        weight *= closeness;

        // accumulate
        filteredColor += color * weight;
        weightSum += weight;
    }

    if (weightSum > 0)
    {
        filteredColor /= weightSum;
    }
    else
    {
        filteredColor = Float3(0.5f);
    }

    Store2DHalf4(Float4(filteredColor, 1.0f), colorBuffer, Int2(x, y));
}


}
