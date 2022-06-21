#include "shaders/LinearMath.h"
#include "shaders/Sampler.h"

namespace jazzfusion
{

__global__ void BicubicFilter(
    Float4* out,
    SurfObj colorBuffer,
    Int2 texSize,
    Int2 outSize)
{
    Int2 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx.x >= outSize.x || idx.y >= outSize.y) return;

    int linearId = idx.y * outSize.x + idx.x;
    Float2 uv = ToFloat2(idx) / ToFloat2(outSize);

    Float3 sampledColor = SampleBicubicCatmullRom<Load2DFuncHalf4<Float3>>(colorBuffer, uv, texSize);
    // Float3 sampledColor = SampleNearest<Load2DFuncHalf4<Float3>>(colorBuffer, uv, texSize);
    // Float3 sampledColor = Load2DHalf4(colorBuffer, idx).xyz;

    out[linearId] = Float4(sampledColor, 0);
}

}