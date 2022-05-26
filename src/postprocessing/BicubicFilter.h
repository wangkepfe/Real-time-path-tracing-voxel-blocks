#include "util/LinearMath.h"

namespace jazzfusion
{

__global__ void BicubicFilter(
    float4* out,
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

    Float2 UV = uv * texSize;
    Float2 invTexSize = 1.0f / texSize;
    Float2 tc = floor(UV - 0.5f) + 0.5f;
    Float2 f = UV - tc;

    Float2 f2 = f * f;
    Float2 f3 = f2 * f;

    Float2 w0 = f2 - 0.5f * (f3 + f);
    Float2 w1 = 1.5f * f3 - 2.5f * f2 + 1.0f;
    Float2 w3 = 0.5f * (f3 - f2);
    Float2 w2 = 1.0f - w0 - w1 - w3;

    Int2 tc1 = floori(UV - 0.5f);
    Int2 tc0 = tc1 - 1;
    Int2 tc2 = tc1 + 1;
    Int2 tc3 = tc1 + 2;

    Int2 sampleUV[16] = {
        { tc0.x, tc0.y }, { tc1.x, tc0.y }, { tc2.x, tc0.y }, { tc3.x, tc0.y },
        { tc0.x, tc1.y }, { tc1.x, tc1.y }, { tc2.x, tc1.y }, { tc3.x, tc1.y },
        { tc0.x, tc2.y }, { tc1.x, tc2.y }, { tc2.x, tc2.y }, { tc3.x, tc2.y },
        { tc0.x, tc3.y }, { tc1.x, tc3.y }, { tc2.x, tc3.y }, { tc3.x, tc3.y },
    };

    float weights[16] = {
        w0.x * w0.y,  w1.x * w0.y,  w2.x * w0.y,  w3.x * w0.y,
        w0.x * w1.y,  w1.x * w1.y,  w2.x * w1.y,  w3.x * w1.y,
        w0.x * w2.y,  w1.x * w2.y,  w2.x * w2.y,  w3.x * w2.y,
        w0.x * w3.y,  w1.x * w3.y,  w2.x * w3.y,  w3.x * w3.y,
    };

    Float3 outColor = Float3(0);
    float sumWeight = 0;

#pragma unroll
    for (int i = 0; i < 16; i++)
    {
        sumWeight += weights[i];
        outColor += Float3(surf2Dread<float4>(colorBuffer, sampleUV[i].x * sizeof(float4), sampleUV[i].y, cudaBoundaryModeClamp)) * weights[i];
    }

    outColor /= sumWeight;

    out[linearId] = (float4)outColor;
}

}