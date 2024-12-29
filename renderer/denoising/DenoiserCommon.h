#pragma once

#include "shaders/LinearMath.h"
#include "shaders/Camera.h"

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

    __device__ float SmoothStep(float a, float b, float x)
    {
        float t = LinearStep(a, b, x);
        return t * t * (3.0f - 2.0f * t);
    }

    __device__ float step(float edge, float v)
    {
        return (v < edge) ? 0.0f : 1.0f;
    }

    __device__ Float3 step(Float3 edge, float v)
    {
        return Float3(step(edge.x, v), step(edge.y, v), step(edge.z, v));
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

    __device__ float Denanify(float w, float x)
    {
        return (w) == 0.0f ? 0.0f : (x);
    }

    __device__ Float4 Denanify(float w, Float4 x)
    {
        return (w) == 0.0f ? Float4(0.0f) : (x);
    }

    __device__ Float4 IsInScreenBilinear(Int2 footprintOrigin, Int2 rectSize)
    {
        Int4 p = Int4(footprintOrigin.x, footprintOrigin.y, footprintOrigin.x + 1, footprintOrigin.y + 1);

        Float4 r = Float4(
            (p.x >= 0) ? 1.0f : 0.0f,
            (p.y >= 0) ? 1.0f : 0.0f,
            (p.z >= 0) ? 1.0f : 0.0f,
            (p.w >= 0) ? 1.0f : 0.0f);

        Float4 comp = Float4(
            (p.x < rectSize.x) ? 1.0f : 0.0f,
            (p.y < rectSize.y) ? 1.0f : 0.0f,
            (p.z < rectSize.x) ? 1.0f : 0.0f,
            (p.w < rectSize.y) ? 1.0f : 0.0f);

        r *= comp;

        Float4 r_xzxz = Float4(r.x, r.z, r.x, r.z);
        Float4 r_yyww = Float4(r.y, r.y, r.w, r.w);

        return Float4(r_xzxz * r_yyww);
    }

    struct BilinearFiltering
    {
        __device__ BilinearFiltering() {}

        Float2 origin;
        Float2 weights;
    };

    __device__ Float4 GetBilinearCustomWeights(BilinearFiltering f, Float4 customWeights)
    {
        Float2 oneMinusWeights = Float2(1.0f) - f.weights;

        Float4 weights = customWeights;
        weights.x *= oneMinusWeights.x * oneMinusWeights.y;
        weights.y *= f.weights.x * oneMinusWeights.y;
        weights.z *= oneMinusWeights.x * f.weights.y;
        weights.w *= f.weights.x * f.weights.y;

        return weights;
    }

    __device__ inline Float4 SampleTexture(TexObj tex, float u, float v)
    {
        return Float4(tex2D<float4>(tex, u, v));
    }

    // The main function translated to CUDA
    __device__ Float4 BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights(
        Float2 samplePos,
        Float2 invResourceSize,
        Float4 bilinearCustomWeights,
        bool useBicubic,
        TexObj tex0)
    {
        // Catmul-Rom with 12 taps (excluding corners)

        // centerPos = floor(samplePos - 0.5) + 0.5
        Float2 centerPos(floorf(samplePos.x - 0.5f) + 0.5f, floorf(samplePos.y - 0.5f) + 0.5f);

        Float2 f(samplePos.x - centerPos.x, samplePos.y - centerPos.y);
        f.x = saturate(f.x);
        f.y = saturate(f.y);

        // Compute Catmull-Rom weights
        constexpr float catromSharpness = 0.5f;
        Float2 w0 = f * (f * (-catromSharpness * f + 2.0f * catromSharpness) - catromSharpness);
        Float2 w1 = f * (f * ((2.0f - catromSharpness) * f - (3.0f - catromSharpness))) + 1.0f;
        Float2 w2 = f * (f * (-(2.0f - catromSharpness) * f + (3.0f - 2.0f * catromSharpness)) + catromSharpness);
        Float2 w3 = f * (f * (catromSharpness * f - catromSharpness));

        Float2 w12(w1.x + w2.x, w1.y + w2.y);
        Float2 tc(w2.x / w12.x, w2.y / w12.y);

        Float4 w(0.0f);
        w.x = w12.x * w0.y;
        w.y = w0.x * w12.y;
        w.z = w12.x * w12.y;
        w.w = w3.x * w12.y;

        float w4 = w12.x * w3.y;

        // Fallback to custom bilinear if not using bicubic
        w = useBicubic ? w : bilinearCustomWeights;
        w4 = useBicubic ? w4 : 0.0f;

        // sum = dot(w, 1.0) + w4
        float sum = dot(w, Float4(1.0f)) + w4;

        // Texture coordinates
        Float4 uv01, uv23;
        Float2 uv4Coord;

        if (useBicubic)
        {
            // uv01 = centerPos.xyxy + Float4( tc.x, -1.0, -1.0, tc.y )
            uv01 = Float4(centerPos.x, centerPos.y, centerPos.x, centerPos.y) +
                   Float4(tc.x, -1.0f, -1.0f, tc.y);

            // uv23 = centerPos.xyxy + Float4( tc.x, tc.y, 2.0, tc.y )
            uv23 = Float4(centerPos.x, centerPos.y, centerPos.x, centerPos.y) +
                   Float4(tc.x, tc.y, 2.0f, tc.y);

            // uv4 = centerPos + Float2( tc.x, 2.0 )
            uv4Coord = Float2(centerPos.x + tc.x, centerPos.y + 2.0f);
        }
        else
        {
            // uv01 = centerPos.xyxy + Float4( 0, 0, 1, 0 )
            uv01 = Float4(centerPos.x, centerPos.y, centerPos.x, centerPos.y) +
                   Float4(0.0f, 0.0f, 1.0f, 0.0f);

            // uv23 = centerPos.xyxy + Float4( 0, 1, 1, 1 )
            uv23 = Float4(centerPos.x, centerPos.y, centerPos.x, centerPos.y) +
                   Float4(0.0f, 1.0f, 1.0f, 1.0f);

            // uv4 = centerPos + f
            uv4Coord = Float2(centerPos.x + f.x, centerPos.y + f.y);
        }

        // Scale by inverse resource size
        uv01.x *= invResourceSize.x;
        uv01.y *= invResourceSize.y;
        uv01.z *= invResourceSize.x;
        uv01.w *= invResourceSize.y;

        uv23.x *= invResourceSize.x;
        uv23.y *= invResourceSize.y;
        uv23.z *= invResourceSize.x;
        uv23.w *= invResourceSize.y;

        uv4Coord.x *= invResourceSize.x;
        uv4Coord.y *= invResourceSize.y;

        // Sample texture
        Float4 color(0.0f);
        Float4 c;

        // color = tex.SampleLevel(gLinearClamp, uv01.xy, 0) * w.x;
        c = SampleTexture(tex0, uv01.x, uv01.y);
        color.x += c.x * w.x;
        color.y += c.y * w.x;
        color.z += c.z * w.x;
        color.w += c.w * w.x;

        // color += tex.SampleLevel(gLinearClamp, uv01.zw, 0) * w.y;
        c = SampleTexture(tex0, uv01.z, uv01.w);
        color.x += c.x * w.y;
        color.y += c.y * w.y;
        color.z += c.z * w.y;
        color.w += c.w * w.y;

        // color += tex.SampleLevel(gLinearClamp, uv23.xy, 0) * w.z;
        c = SampleTexture(tex0, uv23.x, uv23.y);
        color.x += c.x * w.z;
        color.y += c.y * w.z;
        color.z += c.z * w.z;
        color.w += c.w * w.z;

        // color += tex.SampleLevel(gLinearClamp, uv23.zw, 0) * w.w;
        c = SampleTexture(tex0, uv23.z, uv23.w);
        color.x += c.x * w.w;
        color.y += c.y * w.w;
        color.z += c.z * w.w;
        color.w += c.w * w.w;

        // color += tex.SampleLevel(gLinearClamp, uv4, 0) * w4;
        c = SampleTexture(tex0, uv4Coord.x, uv4Coord.y);
        color.x += c.x * w4;
        color.y += c.y * w4;
        color.z += c.z * w4;
        color.w += c.w * w4;

        // Normalize similarly to "Filtering::ApplyBilinearCustomWeights()"
        if (sum < 0.0001f)
        {
            color = Float4(0.0f);
        }
        else
        {
            float invSum = 1.0f / sum;
            color *= invSum;
        }

        return color;
    }

    __device__ Float3 GetCurrentWorldPosFromPixelPos(
        Camera camera,
        Int2 idx,
        float depth)
    {
        Float2 uv = (Float2(idx.x, idx.y) + 0.5f) * camera.inversedResolution;
        return camera.pos + camera.uvToWorldDirection(uv) * depth;
    }

    __device__ float BilinearWithCustomWeightsImmediateFloat(float s00, float s10, float s01, float s11, Float4 bilinearCustomWeights)
    {
        float output = s00 * bilinearCustomWeights.x;
        output += s10 * bilinearCustomWeights.y;
        output += s01 * bilinearCustomWeights.z;
        output += s11 * bilinearCustomWeights.w;

        float sumWeights = dot(bilinearCustomWeights, Float4(1.0f));
        output = sumWeights < 0.0001f ? 0 : output * rcp(sumWeights);
        return output;
    }

    __device__ float GetPlaneDistanceWeight_Atrous(Float3 centerWorldPos, Float3 centerNormal, Float3 sampleWorldPos, float threshold)
    {
        float distanceToCenterPointPlane = abs(dot(sampleWorldPos - centerWorldPos, centerNormal));

        return distanceToCenterPointPlane < threshold ? 1.0f : 0.0f;
    }

    __device__ float GetPlaneDistanceWeight(Float3 centerWorldPos, Float3 centerNormal, float centerViewZ, Float3 sampleWorldPos, float threshold)
    {
        float distanceToCenterPointPlane = abs(dot(sampleWorldPos - centerWorldPos, centerNormal));

        return distanceToCenterPointPlane / centerViewZ > threshold ? 0.0f : 1.0f;
    }

    // Float3/YCoCg conversions
    __device__ Float3 RgbToYCoCg(const Float3 &c)
    {
        return Float3(0.25f * (c.x + 2.0f * c.y + c.z), c.x - c.z, c.y - 0.5f * (c.x + c.z));
    }

    __device__ Float3 YCoCgToRgb(const Float3 &c)
    {
        return Float3(c.x + 0.5f * (c.y - c.z), c.x + 0.5f * c.z, c.x - 0.5f * (c.y + c.z));
    }

    __device__ Float4 RgbToYCoCg(const Float4 &c)
    {
        return Float4(RgbToYCoCg(Float3(c.xyz)), c.w);
    }

    __device__ Float4 YCoCgToRgb(const Float4 &c)
    {
        return Float4(YCoCgToRgb(Float3(c.xyz)), c.w);
    }

    __device__ Float4 GetRotator(float angle)
    {
        float ca = cos(angle);
        float sa = sin(angle);

        return Float4(ca, sa, -sa, ca);
    }

    __device__ float Weyl1D(float p, int n)
    {
        return fract(p + float(n * 10368889) / exp2(24.0f));
    }

    __device__ Float2 RotateVector(Float4 rotator, Float2 v)
    {
        return v.x * rotator.xz() + v.y * rotator.yw();
    }

    __device__ float IsInScreenNearest(Float2 uv)
    {
        return (uv.x >= 0.0f && uv.x < 1.0f && uv.y >= 0.0f && uv.y < 1.0f) ? 1.0f : 0.0f;
    }

    __device__ float ComputeNonExponentialWeight(float x, float px, float py)
    {
        return SmoothStep(1.0f, 0.0f, abs(x * px + py));
    }

    __device__ float PixelRadiusToWorld(float unproject, float pixelRadius, float viewZ)
    {
        return pixelRadius * unproject * viewZ;
    }

    __device__ float GetHitDistFactor(float hitDist, float frustumSize)
    {
        return saturate(hitDist / frustumSize);
    }

    __device__ float GetSpecLobeTanHalfAngle(float roughness, float percentOfVolume = 0.75f)
    {
        // TODO: ideally should migrate to fixed "ImportanceSampling::GetSpecularLobeTanHalfAngle", but since
        // denoisers behavior have been tuned for the old version, let's continue to use it in critical places
        roughness = saturate(roughness);
        percentOfVolume = saturate(percentOfVolume);

        return roughness * roughness * percentOfVolume / (1.0f - percentOfVolume + 1e-6f);
    }

    __device__ float GetNormalWeightParam2(float roughness, float angleFraction)
    {
        float angle = atan(GetSpecLobeTanHalfAngle(roughness, angleFraction));
        angle = 1.0f / max(angle, 1e-6f);

        return angle;
    }

    __device__ float Pow01(float x, float y)
    {
        return pow(saturate(x), y);
    }

    __device__ float GetSpecMagicCurve(float roughness, float power = 0.25f)
    {
        float f = 1.0f - exp2(-200.0f * roughness * roughness);
        f *= Pow01(roughness, power);

        return f;
    }

    __device__ Float2 GetHitDistanceWeightParams(float hitDist, float nonLinearAccumSpeed, float roughness = 1.0f)
    {
        float smc = GetSpecMagicCurve(roughness);
        float norm = lerp(0.0005f, 1.0f, min(nonLinearAccumSpeed, smc));
        float a = 1.0f / norm;
        float b = hitDist * a;

        return Float2(a, -b);
    }

    __device__ unsigned int SequenceHash(unsigned int x)
    {
// This little gem is from https://nullprogram.com/blog/2018/07/31/, "Prospecting for Hash Functions" by Chris Wellons
#if 1 // faster, higher bias
        x ^= x >> 16;
        x *= 0x7FEB352D;
        x ^= x >> 15;
        x *= 0x846CA68B;
        x ^= x >> 16;
#else // slower, lower bias
        x ^= x >> 17;
        x *= 0xED5AD4BB;
        x ^= x >> 11;
        x *= 0xAC4C1B51;
        x ^= x >> 15;
        x *= 0x31848BAB;
        x ^= x >> 14;
#endif

        return x;
    }

    __device__ float UintToFloat01(unsigned int value)
    {
        return value / 4294967295.0f; // 4294967295 is the maximum value of a 32-bit unsigned integer
    }

    __device__ Float2 UintToFloat01(UInt2 value)
    {
        return Float2(UintToFloat01(value.x), UintToFloat01(value.y));
    }

    __device__ Float4 UintToFloat01(UInt4 value)
    {
        return Float4(UintToFloat01(value.x), UintToFloat01(value.y), UintToFloat01(value.z), UintToFloat01(value.w));
    }

    __device__ unsigned int SequenceHashCombine(unsigned int seed, unsigned int value)
    {
        return seed ^ (SequenceHash(value) + 0x9E3779B9 + (seed << 6) + (seed >> 2));
    }

    __device__ unsigned int SequenceIntegerExplode(unsigned int x)
    {
        x = (x | (x << 8)) & 0x00FF00FF;
        x = (x | (x << 4)) & 0x0F0F0F0F;
        x = (x | (x << 2)) & 0x33333333;
        x = (x | (x << 1)) & 0x55555555;

        return x;
    }

    __device__ unsigned int SequenceZorder(UInt2 xy)
    {
        return SequenceIntegerExplode(xy.x) | (SequenceIntegerExplode(xy.y) << 1);
    }

    __device__ void RngHashInitialize(unsigned int &rngHashState, unsigned int linearIndex, unsigned int frameIndex)
    {
        rngHashState = SequenceHashCombine(SequenceHash(frameIndex + 0x035F9F29), linearIndex);
    }

    __device__ void RngHashInitialize(unsigned int &rngHashState, UInt2 samplePos, unsigned int frameIndex)
    {
        RngHashInitialize(rngHashState, SequenceZorder(samplePos), frameIndex);
    }

    __device__ unsigned int RngNext(unsigned int seed)
    {
        seed = SequenceHash(seed);
        return seed;
    }

    __device__ unsigned int RngHashGetUint(unsigned int &rngHashState)
    {
        rngHashState = RngNext(rngHashState);

        return rngHashState;
    }

    __device__ UInt2 RngHashGetUint2(unsigned int &rngHashState)
    {
        return UInt2(RngHashGetUint(rngHashState), RngHashGetUint(rngHashState));
    }

    __device__ UInt4 RngHashGetUint4(unsigned int &rngHashState)
    {
        return UInt4(RngHashGetUint2(rngHashState), RngHashGetUint2(rngHashState));
    }

    __device__ float RngHashGetFloat(unsigned int &rngHashState)
    {
        unsigned int x = RngHashGetUint(rngHashState);
        return UintToFloat01(x);
    }

    __device__ Float2 RngHashGetFloat2(unsigned int &rngHashState)
    {
        UInt2 x = RngHashGetUint2(rngHashState);
        return UintToFloat01(x);
    }

    __device__ Float4 RngHashGetFloat4(unsigned int &rngHashState)
    {
        UInt4 x = RngHashGetUint4(rngHashState);
        return UintToFloat01(x);
    }
}