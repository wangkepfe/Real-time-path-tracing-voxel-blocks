#include "shaders/LinearMath.h"
#include "shaders/Sampler.h"
#include "core/GlobalSettings.h"

namespace jazzfusion
{

__global__ void CalculateTileNoiseLevel(
    SurfObj colorBuffer,
    SurfObj depthBuffer,
    SurfObj noiseLevelBuffer,
    Int2    size)
{
    int x = threadIdx.x + blockIdx.x * 8;
    int y = threadIdx.y + blockIdx.y * 4;

    int x1 = x;
    int y1 = y * 2;

    int x2 = x;
    int y2 = y * 2 + 1;

    float depthValue1 = Load2DHalf1(depthBuffer, Int2(x1, y1));
    Float3 colorValue1 = Load2DHalf4(colorBuffer, Int2(x1, y1)).xyz;

    float depthValue2 = Load2DHalf1(depthBuffer, Int2(x2, y2));
    Float3 colorValue2 = Load2DHalf4(colorBuffer, Int2(x2, y2)).xyz;

    uint background1 = depthValue1 >= RayMax;
    uint background2 = depthValue2 >= RayMax;

    float lum1 = colorValue1.getmax();
    float lum12 = lum1 * lum1;

    float lum2 = colorValue2.getmax();
    float lum22 = lum2 * lum2;

    WarpReduceSum(background1);
    WarpReduceSum(background2);
    WarpReduceSum(lum1);
    WarpReduceSum(lum12);
    WarpReduceSum(lum2);
    WarpReduceSum(lum22);

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        float notSkyRatio = 1.0f - (background1 + background2) / 64.0f;

        float lumAve = (lum1 + lum2) / 64.0f;
        float lumAveSq = lumAve * lumAve;
        float lumSqAve = (lum12 + lum22) / 64.0f;

        float blockVariance = max(1e-20f, lumSqAve - lumAveSq);

        float noiseLevel = blockVariance / max(lumAveSq, 1e-20f);
        noiseLevel *= notSkyRatio;

        Int2 gridLocation = Int2(blockIdx.x, blockIdx.y);
        Store2DHalf1(noiseLevel, noiseLevelBuffer, gridLocation);
    }
}

__global__ void TileNoiseLevel8x8to16x16(SurfObj noiseLevelBuffer, SurfObj noiseLevelBuffer16)
{
    int x = threadIdx.x + blockIdx.x * 8;
    int y = threadIdx.y + blockIdx.y * 8;
    float v1 = Load2DHalf1(noiseLevelBuffer, Int2(x * 2, y * 2));
    float v2 = Load2DHalf1(noiseLevelBuffer, Int2(x * 2 + 1, y * 2));
    float v3 = Load2DHalf1(noiseLevelBuffer, Int2(x * 2, y * 2 + 1));
    float v4 = Load2DHalf1(noiseLevelBuffer, Int2(x * 2 + 1, y * 2 + 1));
    Store2DHalf1((v1 + v2 + v3 + v4) / 4.0f, noiseLevelBuffer16, Int2(x, y));
}

template<int Level>
__global__ void TileNoiseLevelVisualize(SurfObj colorBuffer, SurfObj normalBuffer, SurfObj depthBuffer, SurfObj noiseLevelBuffer, Int2 size, DenoisingParams params)
{
    float noiseLevel = Load2DHalf1(noiseLevelBuffer, Int2(blockIdx.x, blockIdx.y));

    Int2 idx(threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y);

    if (Level == 1)
    {
        if (noiseLevel > params.noise_threshold_local)
        {
            if ((threadIdx.x == 0) || (threadIdx.x == (blockDim.x - 1)) || (threadIdx.y == 0) || (threadIdx.y == (blockDim.y - 1)))
            {
                Store2DHalf4(Float4(1, 0.5f, 0, 0), colorBuffer, idx);
                Store2DHalf4(Float4(0), normalBuffer, idx);
                Store2DHalf1(RayMax, depthBuffer, idx);
            }
        }
    }
    else if (Level == 2)
    {
        if (noiseLevel > params.noise_threshold_large)
        {
            if ((threadIdx.x == 0) || (threadIdx.x == (blockDim.x - 1)) || (threadIdx.y == 0) || (threadIdx.y == (blockDim.y - 1)))
            {
                Store2DHalf4(Float4(1, 0, 0, 0), colorBuffer, idx);
                Store2DHalf4(Float4(0), normalBuffer, idx);
                Store2DHalf1(RayMax, depthBuffer, idx);
            }
        }
    }
}

}