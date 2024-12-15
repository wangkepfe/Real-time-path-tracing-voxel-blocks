#pragma once

#include "shaders/LinearMath.h"

namespace jazzfusion
{

    __device__ float saturate(float x)
    {
        return fminf(fmaxf(x, 0.0f), 1.0f);
    }

    __device__ Float2 saturate(const Float2 &v)
    {
        return Float2(saturate(v.x), saturate(v.y));
    }

    __device__ float LinearStep(float a, float b, float x)
    {
        return saturate((x - a) / (b - a));
    }

    __device__ float GetSpecularLobeHalfAngle(float linearRoughness, float percentOfVolume = 0.75)
    {
        float m = linearRoughness * linearRoughness;
        return atan(m * percentOfVolume / (1.0 - percentOfVolume));
    }

    __device__ float AcosApprox(float x)
    {
        return sqrt(2.0f) * sqrt(saturate(1.0f - x));
    }

    __device__ float AtanApprox(float x)
    {
        return 0.25f * M_PI * x - (abs(x) * x - x) * (0.2447f + 0.0663f * abs(x));
    }

    __device__ float IsInScreen(Float2 uv)
    {
        return float(saturate(uv) == uv);
    }

    __device__ float GetGaussianWeight(float r)
    {
        return exp(-0.66f * r * r); // assuming r is normalized to 1
    }

    __device__ float GetBilateralWeight(float z, float zc)
    {
        float t = abs(z - zc) * rcp(max(abs(z), abs(zc)) + 1e-6f);

        constexpr float bilateralWeightCutoff = 0.03f;

        return LinearStep(bilateralWeightCutoff, 0.0, t);
    }

    __device__ float ExpApprox(float x)
    {
        return rcp((x) * (x) - (x) + 1.0);
    }

    __device__ float ComputeExponentialWeight(float x, float px, float py, float scale)
    {
        return ExpApprox(-scale * abs((x) * (px) + (py)));
    }

}