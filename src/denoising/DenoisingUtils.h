#include "shaders/LinearMath.h"
#include "shaders/Sampler.h"

namespace jazzfusion
{

__global__ void RecoupleAlbedo(SurfObj colorBuffer, SurfObj albedoBuffer, Int2 texSize)
{
    Int2 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx.x >= texSize.x || idx.y >= texSize.y) return;

    Float3 color = Load2DHalf4(colorBuffer, idx).xyz;
    Float3 albedo = Load2DHalf4(albedoBuffer, idx).xyz;

    Store2DHalf4(Float4(color * albedo, 0.0f), colorBuffer, idx);
}

}