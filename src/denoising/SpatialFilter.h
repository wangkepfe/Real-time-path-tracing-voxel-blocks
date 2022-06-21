#include "shaders/LinearMath.h"
#include "shaders/Sampler.h"
#include "util/Gaussian.h"
#include "core/GlobalSettings.h"

#define SPATIAL_FILTER_7X7_USE_SAMPLE_KERNEL_3D_PATTERN 0
#define SPATIAL_FILTER_7X7_USE_DEFAULT_PATTERN 0
#define SPATIAL_FILTER_7X7_USE_STRIDE_PATTERN 1

namespace jazzfusion
{

// o: 16x16 thread block, 256 thread
// x: 22x22 data block, 484 data, 256 x 2 = 512, two passes
//
//       x x x x x x x x x x x x x x x x x x x x x x      1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
//       x x x x x x x x x x x x x x x x x x x x x x      1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
//       x x x x x x x x x x x x x x x x x x x x x x      1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
//       x x x o o o o o o o o o o o o o o o o x x x      1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
//       x x x o o o o o o o o o o o o o o o o x x x      1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
//       x x x o o o o o o o o o o o o o o o o x x x      1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
//       x x x o o o o o o o o o o o o o o o o x x x      1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
//       x x x o o o o o o o o o o o o o o o o x x x      1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
//       x x x o o o o o o o o o o o o o o o o x x x      1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
//       x x x o o o o o o o o o o o o o o o o x x x      1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
//       x x x o o o o o o o o o o o o o o o o x x x      1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
//       x x x o o o o o o o o o o o o o o o o x x x      1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2
//       x x x o o o o o o o o o o o o o o o o x x x      2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
//       x x x o o o o o o o o o o o o o o o o x x x      2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
//       x x x o o o o o o o o o o o o o o o o x x x      2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
//       x x x o o o o o o o o o o o o o o o o x x x      2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
//       x x x o o o o o o o o o o o o o o o o x x x      2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
//       x x x o o o o o o o o o o o o o o o o x x x      2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
//       x x x o o o o o o o o o o o o o o o o x x x      2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
//       x x x x x x x x x x x x x x x x x x x x x x      2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
//       x x x x x x x x x x x x x x x x x x x x x x      2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
//       x x x x x x x x x x x x x x x x x x x x x x      2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2

__global__ void SpatialFilter7x7(
    int frameNum,
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

    if (0)
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
    constexpr int kernelRadius = 3;

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

    if (depthValue >= RayMaxLowerBound) return;

    // -------------------------------- atrous filter --------------------------------
    Float3 sumOfColor = Float3(0.0f);
    float sumOfWeight = 0;

#if SPATIAL_FILTER_7X7_USE_SAMPLE_KERNEL_3D_PATTERN
#pragma unroll
    for (int i = 0; i < SAMPLE_KERNEL_SAMPLE_PER_FRAME; i += 1)
    {
        int kernelSelect = frameNum % SAMPLE_KERNEL_FRAME_COUNT;
        int kernelSelectIdx = kernelSelect * SAMPLE_KERNEL_SAMPLE_PER_FRAME + i;

        int xoffset = SampleKernel3d[kernelSelectIdx][0];
        int yoffset = SampleKernel3d[kernelSelectIdx][1];
#elif SPATIAL_FILTER_7X7_USE_DEFAULT_PATTERN
#pragma unroll
    for (int i = 0; i < kernelSize; i += 1)
    {
        int xoffset = i % kernelDim;
        int yoffset = i / kernelDim;
#elif SPATIAL_FILTER_7X7_USE_STRIDE_PATTERN
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
        weight *= (maskValue != mask) ? 1.0f / params.local_denoise_sigma_material : 1.0f;

        // gaussian filter weight
        weight *= GetGaussian7x7(xoffset + yoffset * kernelDim);

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

}
