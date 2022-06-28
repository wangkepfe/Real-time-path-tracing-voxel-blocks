#include "shaders/LinearMath.h"
#include "shaders/Sampler.h"
#include "util/Gaussian.h"
#include "core/GlobalSettings.h"

namespace jazzfusion
{

template<int KernelStride>
__global__ void SpatialWideFilter5x5(
    int frameNum,
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

    uint finalMat = (uint)maskValue % NUM_MATERIALS;
    if (finalMat == SKY_MATERIAL_ID)
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
        weight *= (maskValue != mask) ? 1.0f / params.large_denoise_sigma_material : 1.0f;

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

}