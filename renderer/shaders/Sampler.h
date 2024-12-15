#pragma once

#include "HalfPrecision.h"
#include "ShaderDebugUtils.h"

namespace jazzfusion
{

    template <typename ReturnType = Float4>
    INL_DEVICE ReturnType Load2DHalf4(
        SurfObj tex,
        Int2 idx)
    {
        ushort4 ret = surf2Dread<ushort4>(tex, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);

        ushort4ToHalf4Converter conv(ret);
        Half4 hf4 = conv.hf4;

        return half4ToFloat4(hf4);
    }

    template <>
    INL_DEVICE Half4 Load2DHalf4<Half4>(
        SurfObj tex,
        Int2 idx)
    {
        ushort4 ret = surf2Dread<ushort4>(tex, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);

        ushort4ToHalf4Converter conv(ret);
        Half4 hf4 = conv.hf4;

        return hf4;
    }

    template <>
    INL_DEVICE Half3 Load2DHalf4<Half3>(
        SurfObj tex,
        Int2 idx)
    {
        ushort4 ret = surf2Dread<ushort4>(tex, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);

        ushort4ToHalf3Converter conv(ret);
        Half3 hf3 = conv.hf3;

        return hf3;
    }

    INL_DEVICE void Store2DHalf4(
        Float4 fl4,
        SurfObj tex,
        Int2 idx)
    {
        Half4 hf4 = float4ToHalf4(fl4);

        ushort4ToHalf4Converter conv(hf4);
        ushort4 us4 = conv.us4;

        surf2Dwrite(us4, tex, idx.x * 4 * sizeof(unsigned short), idx.y, cudaBoundaryModeClamp);
    }

    INL_DEVICE void Store2DHalf4(
        Half4 hf4,
        SurfObj tex,
        Int2 idx)
    {
        ushort4ToHalf4Converter conv(hf4);
        ushort4 us4 = conv.us4;

        surf2Dwrite(us4, tex, idx.x * 4 * sizeof(unsigned short), idx.y, cudaBoundaryModeClamp);
    }

    INL_DEVICE Float2 Load2DHalf2(SurfObj tex, Int2 idx)
    {
        ushort2ToHalf2Converter conv(surf2Dread<ushort2>(tex, idx.x * 2 * sizeof(short), idx.y, cudaBoundaryModeClamp));
        float2 ret = __half22float2(conv.hf2);
        return Float2(ret.x, ret.y);
    }

    INL_DEVICE void Store2DHalf2(Float2 val, SurfObj tex, Int2 idx)
    {
        ushort2ToHalf2Converter conv(__float22half2_rn(make_float2(val.x, val.y)));
        surf2Dwrite(conv.us2, tex, idx.x * 2 * sizeof(short), idx.y, cudaBoundaryModeClamp);
    }

    template <typename ReturnType = float>
    INL_DEVICE ReturnType Load2DHalf1(SurfObj tex, Int2 idx)
    {
        ushort1ToHalf1Converter conv(surf2Dread<ushort1>(tex, idx.x * 1 * sizeof(short), idx.y, cudaBoundaryModeClamp));
        return __half2float(conv.hf1);
    }

    template <>
    INL_DEVICE half Load2DHalf1<half>(SurfObj tex, Int2 idx)
    {
        ushort1ToHalf1Converter conv(surf2Dread<ushort1>(tex, idx.x * 1 * sizeof(short), idx.y, cudaBoundaryModeClamp));
        return conv.hf1;
    }

    INL_DEVICE void Store2DHalf1(float val, SurfObj tex, Int2 idx)
    {
        ushort1ToHalf1Converter conv(__float2half(val));
        surf2Dwrite(conv.us1, tex, idx.x * 1 * sizeof(short), idx.y, cudaBoundaryModeClamp);
    }

    INL_DEVICE unsigned short Load2DUshort1(
        SurfObj tex,
        Int2 idx)
    {
        return surf2Dread<ushort1>(tex, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp).x;
    }

    INL_DEVICE void Store2DUshort1(
        unsigned short value,
        SurfObj tex,
        Int2 idx)
    {
        surf2Dwrite(make_ushort1(value), tex, idx.x * sizeof(ushort1), idx.y, cudaBoundaryModeClamp);
    }

    INL_DEVICE Float4 Load2DUshort4(
        SurfObj tex,
        Int2 idx)
    {
        ushort4 val = surf2Dread<ushort4>(tex, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);
        return Float4((float)val.x, (float)val.y, (float)val.z, (float)val.w) / 65535.0f;
    }

    INL_DEVICE void Store2DUshort4(
        Float4 val,
        SurfObj tex,
        Int2 idx)
    {
        Float4 value = min4f(val * 65535.0f, Float4(65535.0f));
        surf2Dwrite(make_ushort4((unsigned short)value.x, (unsigned short)value.y, (unsigned short)value.z, (unsigned short)value.w), tex, idx.x * sizeof(ushort4), idx.y, cudaBoundaryModeClamp);
    }

    INL_DEVICE void Store2DFloat4(
        Float4 fl4,
        SurfObj tex,
        Int2 idx)
    {
        surf2Dwrite(make_float4(fl4.x, fl4.y, fl4.z, fl4.w), tex, idx.x * sizeof(float4), idx.y, cudaBoundaryModeClamp);
    }

    INL_DEVICE Float4 Load2DFloat4(
        SurfObj tex,
        Int2 idx)
    {
        float4 val = surf2Dread<float4>(tex, idx.x * sizeof(float4), idx.y, cudaBoundaryModeClamp);
        return Float4(val.x, val.y, val.z, val.w);
    }

    INL_DEVICE void Store2DFloat2(
        Float2 fl2,
        SurfObj tex,
        Int2 idx)
    {
        surf2Dwrite(make_float2(fl2.x, fl2.y), tex, idx.x * sizeof(float2), idx.y, cudaBoundaryModeClamp);
    }

    INL_DEVICE Float2 Load2DFloat2(
        SurfObj tex,
        Int2 idx)
    {
        float2 val = surf2Dread<float2>(tex, idx.x * sizeof(float2), idx.y, cudaBoundaryModeClamp);
        return Float2(val.x, val.y);
    }

    INL_DEVICE void Store2DFloat1(
        float fl1,
        SurfObj tex,
        Int2 idx)
    {
        surf2Dwrite(fl1, tex, idx.x * sizeof(float), idx.y, cudaBoundaryModeClamp);
    }

    INL_DEVICE float Load2DFloat1(
        SurfObj tex,
        Int2 idx)
    {
        float val = surf2Dread<float>(tex, idx.x * sizeof(float), idx.y, cudaBoundaryModeClamp);
        return val;
    }

    //----------------------------------------------------------------------------------------------
    //
    //                                   Boundary Functors
    //
    //----------------------------------------------------------------------------------------------

    struct BoundaryFuncDefault
    {
        INL_DEVICE Int2 operator()(Int2 uv, Int2 size) { return uv; }
    };

    struct BoundaryFuncClamp
    {
        INL_DEVICE Int2 operator()(Int2 uvIn, Int2 size)
        {
            Int2 uv = uvIn;

            if (uv.x >= size.x)
            {
                uv.x = size.x - 1;
            }
            if (uv.y >= size.y)
            {
                uv.y = size.y - 1;
            }
            if (uv.x < 0)
            {
                uv.x = 0;
            }
            if (uv.y < 0)
            {
                uv.y = 0;
            }

            return uv;
        }
    };

    struct BoundaryFuncRepeat
    {
        INL_DEVICE Int2 operator()(Int2 uvIn, Int2 size)
        {
            Int2 uv = uvIn;

            if (uv.x >= size.x)
            {
                uv.x %= size.x;
            }
            if (uv.y >= size.y)
            {
                uv.y %= size.y;
            }
            if (uv.x < 0)
            {
                uv.x = size.x - (-uv.x) % size.x;
            }
            if (uv.y < 0)
            {
                uv.y = size.y - (-uv.y) % size.y;
            }

            return uv;
        }
    };

    struct BoundaryFuncRepeatXClampY
    {
        INL_DEVICE Int2 operator()(Int2 uvIn, Int2 size)
        {
            Int2 uv = uvIn;

            if (uv.x >= size.x)
            {
                uv.x %= size.x;
            }
            if (uv.x < 0)
            {
                uv.x = size.x - (-uv.x) % size.x;
            }
            if (uv.y >= size.y)
            {
                uv.y = size.y - 1;
            }
            if (uv.y < 0)
            {
                uv.y = 0;
            }

            return uv;
        }
    };

    //----------------------------------------------------------------------------------------------
    //
    //                                   Load Functors
    //
    //----------------------------------------------------------------------------------------------

    template <typename VectorType = Float3>
    struct Load2DFuncHalf4
    {
        INL_DEVICE VectorType operator()(SurfObj tex, Int2 uv) { return Load2DHalf4(tex, uv).xyz; }
    };

    template <typename VectorType = Float3>
    struct Load2DFuncFloat4
    {
        INL_DEVICE VectorType operator()(SurfObj tex, Int2 uv) { return Load2DFloat4(tex, uv).xyz; }
    };

    //----------------------------------------------------------------------------------------------
    //
    //                                   Sample Functions
    //
    //----------------------------------------------------------------------------------------------

    template <typename LoadFunc, typename VectorType = Float3, typename BoundaryFunc = BoundaryFuncDefault>
    INL_DEVICE VectorType SampleNearest(
        SurfObj tex,
        const Float2 &UV,
        const Int2 &texSize)
    {
        Int2 tc = floori(UV - 0.5f);
        return LoadFunc()(tex, BoundaryFunc()(tc, texSize));
    }

    template <typename LoadFunc, typename VectorType = Float3, typename BoundaryFunc = BoundaryFuncDefault>
    INL_DEVICE VectorType SampleBilinear(
        SurfObj tex,
        const Float2 &uv,
        const Int2 &texSize)
    {
        Float2 UV = uv * texSize;
        Float2 invTexSize = 1.0f / texSize;
        Float2 tc = floor(UV - 0.5f) + 0.5f;
        Float2 f = UV - tc;

        Float2 w1 = f;
        Float2 w0 = 1.0f - f;

        Int2 tc0 = floori(UV - 0.5f);
        Int2 tc1 = tc0 + 1;

        Int2 sampleUV[4] = {
            {tc0.x, tc0.y},
            {tc1.x, tc0.y},
            {tc0.x, tc1.y},
            {tc1.x, tc1.y},
        };

        float weights[4] = {
            w0.x * w0.y,
            w1.x * w0.y,
            w0.x * w1.y,
            w1.x * w1.y,
        };

        VectorType OutColor;
        float sumWeight = 0;

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            sumWeight += weights[i];
            OutColor += LoadFunc()(tex, BoundaryFunc()(sampleUV[i], texSize)) * weights[i];
        }

        OutColor /= sumWeight;

        return OutColor;
    }

    template <typename LoadFunc, typename VectorType = Float3, typename BoundaryFunc = BoundaryFuncDefault>
    INL_DEVICE VectorType SampleBicubicCatmullRom(
        SurfObj tex,
        const Float2 &UV,
        const Int2 &texSize)
    {
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
            {tc0.x, tc0.y},
            {tc1.x, tc0.y},
            {tc2.x, tc0.y},
            {tc3.x, tc0.y},
            {tc0.x, tc1.y},
            {tc1.x, tc1.y},
            {tc2.x, tc1.y},
            {tc3.x, tc1.y},
            {tc0.x, tc2.y},
            {tc1.x, tc2.y},
            {tc2.x, tc2.y},
            {tc3.x, tc2.y},
            {tc0.x, tc3.y},
            {tc1.x, tc3.y},
            {tc2.x, tc3.y},
            {tc3.x, tc3.y},
        };

        float weights[16] = {
            w0.x * w0.y,
            w1.x * w0.y,
            w2.x * w0.y,
            w3.x * w0.y,
            w0.x * w1.y,
            w1.x * w1.y,
            w2.x * w1.y,
            w3.x * w1.y,
            w0.x * w2.y,
            w1.x * w2.y,
            w2.x * w2.y,
            w3.x * w2.y,
            w0.x * w3.y,
            w1.x * w3.y,
            w2.x * w3.y,
            w3.x * w3.y,
        };

        VectorType OutColor;
        float sumWeight = 0;

#pragma unroll
        for (int i = 0; i < 16; i++)
        {
            sumWeight += weights[i];
            OutColor += LoadFunc()(tex, BoundaryFunc()(sampleUV[i], texSize)) * weights[i];
        }

        OutColor /= sumWeight;

        return OutColor;
    }

    template <typename LoadFunc, typename VectorType = Float3, typename BoundaryFunc = BoundaryFuncDefault>
    INL_DEVICE VectorType SampleBicubicSmoothStep(
        SurfObj tex,
        const Float2 &UV,
        const Int2 &texSize)
    {
        Float2 tc = floor(UV - 0.5f) + 0.5f;
        Float2 f = UV - tc;

        Float2 f2 = f * f;
        Float2 f3 = f2 * f;

        Float2 w1 = -2.0f * f3 + 3.0f * f2;
        Float2 w0 = 1.0f - w1;

        Int2 tc0 = floori(UV - 0.5f);
        Int2 tc1 = tc0 + 1;

        Int2 sampleUV[4] = {
            {tc0.x, tc0.y},
            {tc1.x, tc0.y},
            {tc0.x, tc1.y},
            {tc1.x, tc1.y},
        };

        float weights[4] = {
            w0.x * w0.y,
            w1.x * w0.y,
            w0.x * w1.y,
            w1.x * w1.y,
        };

        VectorType OutColor;
        float sumWeight = 0;

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            sumWeight += weights[i];
            OutColor += LoadFunc()(tex, BoundaryFunc()(sampleUV[i], texSize)) * weights[i];
        }

        OutColor /= sumWeight;

        return OutColor;
    }

}