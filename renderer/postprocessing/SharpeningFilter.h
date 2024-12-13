#include "shaders/LinearMath.h"
#include "shaders/HalfPrecision.h"
#include "shaders/Sampler.h"

namespace jazzfusion
{

INL_DEVICE Float3 min3f3(const Float3& v1, const Float3& v2, const Float3& v3) { return min3f(min3f(v1, v2), v3); }
INL_DEVICE Float3 max3f3(const Float3& v1, const Float3& v2, const Float3& v3) { return max3f(max3f(v1, v2), v3); }

__global__ void SharpeningFilter(SurfObj colorBuffer, Int2 texSize)
{
    // Reference: https://github.com/GPUOpen-Effects/FidelityFX-CAS

    const float sharpness = 1.0f;

    Int2 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx.x >= texSize.x || idx.y >= texSize.y) return;

    Float3 color[3][3];

#pragma unroll
    for (int i = 0; i <= 2; ++i)
    {
#pragma unroll
        for (int j = 0; j <= 2; ++j)
        {
            color[i][j] = Load2DHalf4(colorBuffer, idx + Int2(i - 1, j - 1)).xyz;
        }
    }

    // ----------------soft max------------------
    //        a b c             b
    // (max ( d e f ) + max ( d e f )) * 0.5
    //        g h i             h
    int x = 1, y = 1;
    Float3 tmp1 = max3f3(color[x][y], color[x - 1][y], color[x + 1][y]);
    Float3 tmp2 = max3f3(tmp1, color[x][y - 1], color[x][y + 1]);
    Float3 tmp3 = max3f3(tmp2, color[x - 1][y - 1], color[x - 1][y + 1]);
    Float3 tmp4 = max3f3(tmp3, color[x + 1][y - 1], color[x + 1][y + 1]);
    Float3 softmax = tmp2 + tmp4;

    // soft min
    tmp1 = min3f3(color[x][y], color[x - 1][y], color[x + 1][y]);
    tmp2 = min3f3(tmp1, color[x][y - 1], color[x][y + 1]);
    tmp3 = min3f3(tmp2, color[x - 1][y - 1], color[x - 1][y + 1]);
    tmp4 = min3f3(tmp3, color[x + 1][y - 1], color[x + 1][y + 1]);
    Float3 softmin = tmp2 + tmp4;

    // amp factor
    Float3 amp = clamp3f(min3f(softmin, 2.0f - softmax) / softmax);
    amp = rsqrt3f(amp);

    // weight
    float peak = 8.0f - 3.0f * sharpness;
    Float3 w = -Float3(1.0f) / (amp * peak);

    // 0 w 0
    // w 1 w
    // 0 w 0
    Float3 outColor = (color[0][1] + color[2][1] + color[1][0] + color[1][2]) * w + color[1][1];
    outColor /= (Float3(1.0f) + Float3(4.0f) * w);

    Store2DHalf4(Float4(outColor, 1.0f), colorBuffer, idx);
}

}
