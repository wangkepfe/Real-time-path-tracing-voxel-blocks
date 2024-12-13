#include "shaders/LinearMath.h"
#include "shaders/Sampler.h"
#include "core/GlobalSettings.h"

namespace jazzfusion
{

__global__ void ToneMappingReinhardExtended(
    SurfObj   colorBuffer,
    Int2      size,
    PostProcessParams params)
{
    Int2 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx.x >= size.x || idx.y >= size.y) return;

    Float3 color = Load2DHalf4(colorBuffer, idx).xyz;

    float lum = dot(color, Float3(0.2126f, 0.7152f, 0.0722f));
    color = color * params.gain * (1.0f + (lum / (params.maxWhite * params.maxWhite))) / (1.0f + lum);

    color = pow3f(color, Float3(1.0f / params.gamma));

    Store2DHalf4(Float4(color, 1.0), colorBuffer, idx);
}

}