#include "shaders/LinearMath.h"
#include "shaders/Sampler.h"
#include "util/Gaussian.h"
#include "core/GlobalSettings.h"
#include "shaders/ShaderDebugUtils.h"

namespace jazzfusion
{

__global__ void BilateralFilter(
    int             frameNum,
    int             accuCounter,
    SurfObj         colorBuffer,
    SurfObj         materialBuffer,
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

    __shared__ Float3 sharedBuffer[kernelCoverSize];

    // calculate address
    int id = (threadIdx.x + threadIdx.y * blockdim);

    int x1 = blockIdx.x * blockdim - kernelRadius + id % kernelCoverDim;
    int y1 = blockIdx.y * blockdim - kernelRadius + id / kernelCoverDim;
    int x2 = blockIdx.x * blockdim - kernelRadius + (id + threadCount) % kernelCoverDim;
    int y2 = blockIdx.y * blockdim - kernelRadius + (id + threadCount) / kernelCoverDim;

    Float3 color1 = Load2DHalf4(colorBuffer, Int2(x1, y1)).xyz;
    Float3 color2 = Load2DHalf4(colorBuffer, Int2(x2, y2)).xyz;

    // global load 1
    sharedBuffer[id] = color1;

    // global load 2
    if (id + threadCount < kernelCoverSize)
    {
        sharedBuffer[id + threadCount] = color2;
    }

    __syncthreads();

    // Return out of bound thread
    if (x >= size.x && y >= size.y)
    {
        return;
    }

    // load center
    Float3 centerColor = sharedBuffer[centerIdx];

    // Return for sky pixel


    // Run the filter
    Float3 filteredColor = Float3(0);
    float weightSum = 0;

#pragma unroll
    for (int i = 0; i < kernelSize; ++i)
    {
        int xoffset = i % kernelDim;
        int yoffset = i / kernelDim;

        Float3 color = sharedBuffer[threadIdx.x + xoffset + (threadIdx.y + yoffset) * kernelCoverDim];

        float weight = 1.0f;

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
        filteredColor = Float3(0.5f);;
    }

    Store2DHalf4(Float4(filteredColor, 1.0f), colorBuffer, Int2(x, y));
}

template<int KernelStride>
__global__ void BilateralFilterWide(
    int             frameNum,
    int             accuCounter,
    SurfObj         colorBuffer,
    SurfObj         materialBuffer,
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

    // Return for sky pixel


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

        float weight = 1.0f;

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