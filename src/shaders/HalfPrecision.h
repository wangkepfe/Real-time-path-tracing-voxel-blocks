#pragma once

#include "Common.h"
#include "cuda_fp16.h"

namespace jazzfusion
{

struct __align__(4) Half2
{
    INL_DEVICE Half2() : a(make_half2(0, 0)) {}
    INL_DEVICE Half2(const Half2 & rhs) : a(rhs.a) {}
    INL_DEVICE Half2 operator=(const Half2 & rhs) { a = rhs.a; return *this; }

    INL_DEVICE explicit Half2(half2 v) : a(v) {}
    INL_DEVICE explicit Half2(half v) : a(make_half2(v, v)) {}
    INL_DEVICE explicit Half2(half v1, half v2) : a(make_half2(v1, v2)) {}
    INL_DEVICE explicit Half2(Float2 v)
    {
        a = __float22half2_rn(float2(v));
    }
    INL_DEVICE explicit Half2(float v)
    {
        half val = __float2half_rn(v);
        a = make_half2(val, val);
    }

    INL_DEVICE Half2  operator+(const Half2 & v) const { return Half2(a + v.a); }
    INL_DEVICE Half2  operator-(const Half2 & v) const { return Half2(a - v.a); }
    INL_DEVICE Half2  operator*(const Half2 & v) const { return Half2(a * v.a); }
    INL_DEVICE Half2  operator/(const Half2 & v) const { return Half2(a / v.a); }

    INL_DEVICE Half2  operator+(half v) const { return Half2(a + make_half2(v, v)); }
    INL_DEVICE Half2  operator-(half v) const { return Half2(a - make_half2(v, v)); }
    INL_DEVICE Half2  operator*(half v) const { return Half2(a * make_half2(v, v)); }
    INL_DEVICE Half2  operator/(half v) const { return Half2(a / make_half2(v, v)); }

    INL_DEVICE Half2& operator+=(const Half2 & v) { a = a + v.a; return *this; }
    INL_DEVICE Half2& operator-=(const Half2 & v) { a = a - v.a; return *this; }
    INL_DEVICE Half2& operator*=(const Half2 & v) { a = a * v.a; return *this; }
    INL_DEVICE Half2& operator/=(const Half2 & v) { a = a / v.a; return *this; }

    INL_DEVICE Half2& operator+=(half v) { a = a + make_half2(v, v); return *this; }
    INL_DEVICE Half2& operator-=(half v) { a = a - make_half2(v, v); return *this; }
    INL_DEVICE Half2& operator*=(half v) { a = a * make_half2(v, v); return *this; }
    INL_DEVICE Half2& operator/=(half v) { a = a / make_half2(v, v); return *this; }

    INL_DEVICE Half2 operator-() const { return Half2(-a); }

    INL_DEVICE explicit operator half2() const { return a; }

    INL_DEVICE Half2 xx() const { return Half2(x, x); }
    INL_DEVICE Half2 yy() const { return Half2(y, y); }

    INL_DEVICE Float2 toFloat2() const { return Float2(__half22float2(a)); }

    union
    {
        half2 a;

        struct
        {
            half x;
            half y;
        };
    };

};

// TODO: 6 bytes weird alignment. Should be replaced by half4.
struct Half3
{
    INL_DEVICE Half3() : a(make_half2(0, 0)), b(0) {}
    INL_DEVICE Half3(Float3 v)
    {
        float2 af;
        af.x = v.x;
        af.y = v.y;
        float bf = v.z;

        a = __float22half2_rn(af);
        b = __float2half_rn(bf);
    }

    half2 a;
    half b;
};

struct __align__(4) Half4
{
    INL_DEVICE Half4() : a(make_half2(0, 0)), b(make_half2(0, 0)) {}
    INL_DEVICE Half4(const Half4 & rhs) : a(rhs.a), b(rhs.b) {}
    INL_DEVICE Half4 operator=(const Half4 & rhs) { a = rhs.a; b = rhs.b; return *this; }

    INL_DEVICE explicit Half4(half v1, half v2, half v3, half v4) : a(make_half2(v1, v2)), b(make_half2(v3, v4)) {}
    INL_DEVICE explicit Half4(half2 v1, half2 v2) : a(v1), b(v2) {}
    INL_DEVICE explicit Half4(half v) : a(make_half2(v, v)), b(make_half2(v, v)) {}
    INL_DEVICE explicit Half4(Float4 v)
    {
        a = __float22half2_rn(make_float2(v.x, v.y));
        b = __float22half2_rn(make_float2(v.z, v.w));
    }
    INL_DEVICE explicit Half4(float v)
    {
        half val = __float2half_rn(v);
        a = make_half2(val, val);
        b = make_half2(val, val);
    }

    INL_DEVICE Half4  operator+(const Half4 & v) const { return Half4(a + v.a, b + v.b); }
    INL_DEVICE Half4  operator-(const Half4 & v) const { return Half4(a - v.a, b - v.b); }
    INL_DEVICE Half4  operator*(const Half4 & v) const { return Half4(a * v.a, b * v.b); }
    INL_DEVICE Half4  operator/(const Half4 & v) const { return Half4(a / v.a, b / v.b); }

    INL_DEVICE Half4  operator+(half v) const { return Half4(a + make_half2(v, v), b + make_half2(v, v)); }
    INL_DEVICE Half4  operator-(half v) const { return Half4(a - make_half2(v, v), b - make_half2(v, v)); }
    INL_DEVICE Half4  operator*(half v) const { return Half4(a * make_half2(v, v), b * make_half2(v, v)); }
    INL_DEVICE Half4  operator/(half v) const { return Half4(a / make_half2(v, v), b / make_half2(v, v)); }

    INL_DEVICE Half4& operator+=(const Half4 & v) { a = a + v.a; b = b + v.b; return *this; }
    INL_DEVICE Half4& operator-=(const Half4 & v) { a = a - v.a; b = b - v.b; return *this; }
    INL_DEVICE Half4& operator*=(const Half4 & v) { a = a * v.a; b = b * v.b; return *this; }
    INL_DEVICE Half4& operator/=(const Half4 & v) { a = a / v.a; b = b / v.b; return *this; }

    INL_DEVICE Half4& operator+=(half v) { a = a + make_half2(v, v); b = b + make_half2(v, v); return *this; }
    INL_DEVICE Half4& operator-=(half v) { a = a - make_half2(v, v); b = b - make_half2(v, v); return *this; }
    INL_DEVICE Half4& operator*=(half v) { a = a * make_half2(v, v); b = b * make_half2(v, v); return *this; }
    INL_DEVICE Half4& operator/=(half v) { a = a / make_half2(v, v); b = b / make_half2(v, v); return *this; }

    INL_DEVICE Half4 operator-() const { return Half4(-a, -b); }

    INL_DEVICE Float4 toFloat4() const { float2 a1 = __half22float2(a); float2 b1 = __half22float2(b); return Float4(a1.x, a1.y, b1.x, b1.y); }

    union
    {
        struct
        {
            half2 a;
            half2 b;
        };

        struct
        {
            Half2 xy;
            Half2 zw;
        };

        struct
        {
            half x;
            half y;
            half z;
            half w;
        };
    };
};

INL_DEVICE half abs(half v) { return __habs(v); }
INL_DEVICE half rcp(half v) { return half(1.0) / v; }

INL_DEVICE Half2 abs(const Half2& v) { return Half2(__habs2(v.a)); }
INL_DEVICE Half2 rcp(const Half2& v) { return Half2(half(1.0)) / v; }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
INL_DEVICE half max1h(half v1, half v2) { return __hmax(v1, v2); }
INL_DEVICE half min1h(half v1, half v2) { return __hmin(v1, v2); }
INL_DEVICE Half2 max2h(const Half2& v1, const Half2& v2) { return Half2(__hmax2(v1.a, v2.a)); }
INL_DEVICE Half2 min2h(const Half2& v1, const Half2& v2) { return Half2(__hmin2(v1.a, v2.a)); }
INL_DEVICE Half4 max4h(const Half4& v1, const Half4& v2) { return Half4(__hmax2(v1.a, v2.a), __hmax2(v1.b, v2.b)); }
INL_DEVICE Half4 min4h(const Half4& v1, const Half4& v2) { return Half4(__hmin2(v1.a, v2.a), __hmin2(v1.b, v2.b)); }
#else
INL_DEVICE half max1h(half v1, half v2) { return max(v1, v2); }
INL_DEVICE half min1h(half v1, half v2) { return min(v1, v2); }
INL_DEVICE Half2 max2h(const Half2& v1, const Half2& v2) { return Half2(max(v1.a.x, v2.a.x), max(v1.a.y, v2.a.y)); }
INL_DEVICE Half2 min2h(const Half2& v1, const Half2& v2) { return Half2(min(v1.a.x, v2.a.x), min(v1.a.y, v2.a.y)); }
INL_DEVICE Half4 max4h(const Half4& v1, const Half4& v2) { return Half4(max(v1.a.x, v2.a.x), max(v1.a.y, v2.a.y), max(v1.b.x, v2.b.x), max(v1.b.y, v2.b.y)); }
INL_DEVICE Half4 min4h(const Half4& v1, const Half4& v2) { return Half4(min(v1.a.x, v2.a.x), min(v1.a.y, v2.a.y), min(v1.b.x, v2.b.x), min(v1.b.y, v2.b.y)); }
#endif

INL_DEVICE Half2 clamp2h(const Half2& v, Half2 lo = Half2(half(0.0)), Half2 hi = Half2(half(1.0))) { return min2h(max2h(v, lo), hi); }

// float ToFloat(half v) { return __half2float(v); }

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

// Minimize squared error across {smallest normal to 16384.0}. 1 op
INL_DEVICE half PrxLoRcp(half a) { return (half)ushortToHalf(ushort(0x7784) - (ushort)ushortToHalf(a)); }
// Minimize squared error across {smallest normal to 16384.0}, 2 ops.
INL_DEVICE half PrxLoRsq(half a) { return (half)ushortToHalf(ushort(0x59a3) - ((ushort)ushortToHalf(a) >> ushort(1))); }

}