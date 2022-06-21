#pragma once

#include "Common.h"
#include "cuda_fp16.h"

namespace jazzfusion
{

struct Half3
{
    half2 a;
    half b;
};

INL_DEVICE Float3 half3ToFloat3(Half3 v)
{
    float2 a = __half22float2(v.a);
    float b = __half2float(v.b);

    Float3 out;
    out.x = a.x;
    out.y = a.y;
    out.z = b;

    return out;
}

INL_DEVICE Half3 float3ToHalf3(Float3 v)
{
    float2 a;
    a.x = v.x;
    a.y = v.y;
    float b = v.z;

    Half3 out;
    out.a = __float22half2_rn(a);
    out.b = __float2half_rn(b);

    return out;
}

struct Half4
{
    half2 a;
    half2 b;
};

INL_DEVICE Float4 half4ToFloat4(Half4 v)
{
    float2 a = __half22float2(v.a);
    float2 b = __half22float2(v.b);

    Float4 out;
    out.x = a.x;
    out.y = a.y;
    out.z = b.x;
    out.w = b.y;

    return out;
}

INL_DEVICE Half4 float4ToHalf4(Float4 v)
{
    float2 a;
    a.x = v.x;
    a.y = v.y;

    float2 b;
    b.x = v.z;
    b.y = v.w;

    Half4 out;
    out.a = __float22half2_rn(a);
    out.b = __float22half2_rn(b);

    return out;
}

union ushortToHalf
{
    INL_DEVICE ushortToHalf(const ushort& v) : us{ v } {}
    INL_DEVICE ushortToHalf(const half& v) : hf{ v } {}

    INL_DEVICE explicit operator ushort() { return us; }
    INL_DEVICE explicit operator half() { return hf; }

    ushort us;
    half hf;
};

union ushort4ToHalf4Converter
{
    INL_DEVICE ushort4ToHalf4Converter(const ushort4& v) : us4{ v } {}
    INL_DEVICE ushort4ToHalf4Converter(const Half4& v) : hf4{ v } {}

    ushort4 us4;
    Half4 hf4;
};

union ushort4ToHalf3Converter
{
    INL_DEVICE ushort4ToHalf3Converter(const ushort4& v) : us4{ v } {}
    INL_DEVICE ushort4ToHalf3Converter(const Half3& v, const half v2) : hf3{ v }, hf1{ v2 } {}

    ushort4 us4;
    struct
    {
        Half3 hf3;
        half hf1;
    };
};

union ushort2ToHalf2Converter
{
    __device__ ushort2ToHalf2Converter(const ushort2& v) : us2{ v } {}
    __device__ ushort2ToHalf2Converter(const half2& v) : hf2{ v } {}

    ushort2 us2;
    half2 hf2;
};

union ushort1ToHalf1Converter
{
    __device__ ushort1ToHalf1Converter(const ushort1& v) : us1{ v } {}
    __device__ ushort1ToHalf1Converter(const half& v) : hf1{ v } {}

    ushort1 us1;
    half hf1;
};

}