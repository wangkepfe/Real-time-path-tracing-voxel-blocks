#include "shaders/LinearMath.h"
#include "shaders/Sampler.h"

namespace jazzfusion
{

__global__ void BicubicFilter(
    SurfObj outColorBuffer,
    SurfObj colorBuffer,
    Int2 texSize,
    Int2 outSize)
{
    Int2 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx.x >= outSize.x || idx.y >= outSize.y) return;

    Float2 uv = (ToFloat2(idx) + 0.5f) / ToFloat2(outSize) * ToFloat2(texSize);
    Float3 sampledColor = SampleBicubicCatmullRom<Load2DFuncHalf4<Float3>>(colorBuffer, uv, texSize);
    Store2DHalf4(Float4(sampledColor, 0), outColorBuffer, idx);
}

}