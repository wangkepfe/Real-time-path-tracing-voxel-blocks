#include "core/GlobalSettings.h"
#include "shaders/LinearMath.h"
#include "shaders/Sampler.h"

__global__ void ToneMappingReinhardExtended(SurfObj colorBuffer, Int2 size, PostProcessParams postProcessParams)
{
    Int2 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx.x >= size.x || idx.y >= size.y)
        return;

    Float3 color = Load2DFloat4(colorBuffer, idx).xyz;

    float gain = postProcessParams.gain;
    float postGain = postProcessParams.postGain;
    float maxWhite = postProcessParams.maxWhite;

    color *= gain;
    maxWhite *= gain;
    color = color * ((color / Float3(maxWhite * maxWhite)) + 1.0f) / (color + 1.0f);
    color *= postGain;

    float gamma = 2.2f;
    color = pow3f(color, Float3(1.0f / gamma));

    Store2DFloat4(Float4(color, 1.0), colorBuffer, idx);
}

__device__ __forceinline__ Float3 ChangeLuminance(Float3 c_in, float l_out)
{
    float l_in = luminance(c_in);
    return c_in * (l_out / l_in);
}

__device__ __forceinline__ Float3 RRTAndODTFit(Float3 v)
{
    Float3 a = v * (v + 0.0245786f) - 0.000090537f;
    Float3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}

__device__ __forceinline__ Float3 RRTAndODTFitLuminance(Float3 v)
{
    float lum = luminance(v);
    float a = lum * (lum + 0.0245786f) - 0.000090537f;
    float b = lum * (0.983729f * lum + 0.4329510f) + 0.238081f;
    return ChangeLuminance(v, a / b);
}

__device__ __forceinline__ Float3 ACESFitted(Float3 color)
{
    // sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
    Mat3 ACESInputMat(
        0.59719, 0.35458, 0.04823,
        0.07600, 0.90834, 0.01566,
        0.02840, 0.13383, 0.83777);

    // ODT_SAT => XYZ => D60_2_D65 => sRGB
    Mat3 ACESOutputMat(
        1.60475, -0.53108, -0.07367,
        -0.10208, 1.10813, -0.00605,
        -0.00327, -0.07276, 1.07602);

    color = ACESInputMat * color;

    // Apply RRT and ODT
    // color = RRTAndODTFit(color);
    color = RRTAndODTFitLuminance(color);

    color = ACESOutputMat * color;

    // Clamp to [0, 1]
    color = clamp3f(color);

    return color;
}

__global__ void ToneMappingACES(
    SurfObj colorBuffer,
    Int2 size
    // float *exposure,
    // PostProcessParams params
)
{
    Int2 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx.x >= size.x || idx.y >= size.y)
        return;

    Float3 color = Load2DFloat4(colorBuffer, idx).xyz;
    color *= 10.0f; // exposure[0];

    // Tone mapping
    color = ACESFitted(color);

    // Gamma correction
    color = clamp3f(pow3f(color, 1.0f / 2.2f), Float3(0), Float3(1));

    Store2DFloat4(Float4(color, 1.0), colorBuffer, idx);
}
