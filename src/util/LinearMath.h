#pragma once

#include <math.h>
#include <cuda_runtime.h>
#include "shaders/Common.h"

#define uint unsigned int
#define ushort unsigned short
#define uint64 unsigned long long int
#define ushort unsigned short

#define PI_OVER_4               0.7853981633974483096156608458198757210492f
#define PI_OVER_2               1.5707963267948966192313216916397514420985f
#define SQRT_OF_ONE_THIRD       0.5773502691896257645091487805019574556476f
#ifndef M_PI
#define M_PI                    3.1415926535897932384626422832795028841971f
#endif
#define TWO_PI                  6.2831853071795864769252867665590057683943f
#define Pi_over_180             0.01745329251f
#define INV_PI                  0.31830988618f
#define INV_TWO_PI              0.15915494309f

namespace jazzfusion
{

struct Float3;
struct Int2;

template<typename T>
INL_HOST_DEVICE T max(const T& a, const T& b) { return a > b ? a : b; }

template<typename T>
INL_HOST_DEVICE T min(const T& a, const T& b) { return a < b ? a : b; }

template<typename T>
INL_HOST_DEVICE T max(const T& a, const T& b, const T& c) { return max(max(a, b), c); }

template<typename T>
INL_HOST_DEVICE T min(const T& a, const T& b, const T& c) { return min(min(a, b), c); }

INL_HOST_DEVICE float FMA(float a, float b, float c)
{
    return fma(a, b, c);
}

// https://pharr.org/matt/blog/2019/11/03/difference-of-floats
// Difference of products, avoiding catastrophic cancellation
INL_HOST_DEVICE float dop(float a, float b, float c, float d)
{
    float cd = c * d;
    float err = FMA(-c, d, cd);
    float dop = FMA(a, b, -cd);
    return dop + err;
}

// Sum of products
INL_HOST_DEVICE float sop(float a, float b, float c, float d)
{
    float cd = c * d;
    float err = FMA(c, d, -cd);
    float dop = FMA(a, b, cd);
    return dop + err;
}

struct CompensatedFloat
{
public:
    // CompensatedFloat Public Methods
    INL_HOST_DEVICE CompensatedFloat(float v, float err = 0) : v(v), err(err) {}
    INL_HOST_DEVICE explicit operator float() const { return v + err; }

    float v, err;
};

INL_HOST_DEVICE CompensatedFloat TwoProd(float a, float b)
{
    float ab = a * b;
    return { ab, FMA(a, b, -ab) };
}

INL_HOST_DEVICE CompensatedFloat TwoSum(float a, float b)
{
    float s = a + b, delta = s - a;
    return { s, (a - (s - delta)) + (b - delta) };
}

INL_HOST_DEVICE CompensatedFloat InnerProduct(float a, float b)
{
    return TwoProd(a, b);
}

template <typename... T>
INL_HOST_DEVICE CompensatedFloat InnerProduct(float a, float b, T... terms)
{
    CompensatedFloat ab = TwoProd(a, b);
    CompensatedFloat tp = InnerProduct(terms...);
    CompensatedFloat sum = TwoSum(ab.v, tp.v);
    return { sum.v, ab.err + (tp.err + sum.err) };
}

INL_HOST_DEVICE float DifferenceOfProducts(float a, float b, float c, float d)
{
    auto cd = c * d;
    auto differenceOfProducts = FMA(a, b, -cd);
    auto error = FMA(-c, d, cd);
    return differenceOfProducts + error;
}

struct Float2
{
    union
    {
        struct { float x, y; };
        float _v[2];
    };

    INL_HOST_DEVICE Float2() : x(0), y(0) {}
    INL_HOST_DEVICE Float2(float _x, float _y) : x(_x), y(_y) {}
    INL_HOST_DEVICE Float2(float _x) : x(_x), y(_x) {}

    INL_HOST_DEVICE Float2  operator+(const Float2& v) const { return Float2(x + v.x, y + v.y); }
    INL_HOST_DEVICE Float2  operator-(const Float2& v) const { return Float2(x - v.x, y - v.y); }
    INL_HOST_DEVICE Float2  operator*(const Float2& v) const { return Float2(x * v.x, y * v.y); }
    INL_HOST_DEVICE Float2  operator/(const Float2& v) const { return Float2(x / v.x, y / v.y); }

    INL_HOST_DEVICE Float2  operator+(float a) const { return Float2(x + a, y + a); }
    INL_HOST_DEVICE Float2  operator-(float a) const { return Float2(x - a, y - a); }
    INL_HOST_DEVICE Float2  operator*(float a) const { return Float2(x * a, y * a); }
    INL_HOST_DEVICE Float2  operator/(float a) const { return Float2(x / a, y / a); }

    INL_HOST_DEVICE Float2& operator+=(const Float2& v) { x += v.x; y += v.y; return *this; }
    INL_HOST_DEVICE Float2& operator-=(const Float2& v) { x -= v.x; y -= v.y; return *this; }
    INL_HOST_DEVICE Float2& operator*=(const Float2& v) { x *= v.x; y *= v.y; return *this; }
    INL_HOST_DEVICE Float2& operator/=(const Float2& v) { x /= v.x; y /= v.y; return *this; }

    INL_HOST_DEVICE Float2& operator+=(const float& a) { x += a; y += a; return *this; }
    INL_HOST_DEVICE Float2& operator-=(const float& a) { x -= a; y -= a; return *this; }
    INL_HOST_DEVICE Float2& operator*=(const float& a) { x *= a; y *= a; return *this; }
    INL_HOST_DEVICE Float2& operator/=(const float& a) { x /= a; y /= a; return *this; }

    INL_HOST_DEVICE Float2 operator-() const { return Float2(-x, -y); }

    INL_HOST_DEVICE bool operator!=(const Float2& v) const { return x != v.x || y != v.y; }
    INL_HOST_DEVICE bool operator==(const Float2& v) const { return x == v.x && y == v.y; }

    INL_HOST_DEVICE float& operator[](int i) { return _v[i]; }
    INL_HOST_DEVICE float  operator[](int i) const { return _v[i]; }

    INL_HOST_DEVICE float length() const { return sqrtf(x * x + y * y); }
    INL_HOST_DEVICE float length2() const { return x * x + y * y; }

};

INL_HOST_DEVICE Float2 operator+(float a, const Float2& v) { return Float2(v.x + a, v.y + a); }
INL_HOST_DEVICE Float2 operator-(float a, const Float2& v) { return Float2(a - v.x, a - v.y); }
INL_HOST_DEVICE Float2 operator*(float a, const Float2& v) { return Float2(v.x * a, v.y * a); }
INL_HOST_DEVICE Float2 operator/(float a, const Float2& v) { return Float2(a / v.x, a / v.y); }

INL_HOST_DEVICE float length(const Float2& v) { return sqrtf(v.x * v.x + v.y * v.y); }

struct Int2
{
    union
    {
        struct { int x, y; };
        int _v[2];
    };

    INL_HOST_DEVICE Int2() : x{ 0 }, y{ 0 } {}
    INL_HOST_DEVICE Int2(int a) : x{ a }, y{ a } {}
    INL_HOST_DEVICE Int2(int x, int y) : x{ x }, y{ y } {}

    INL_HOST_DEVICE Int2 operator + (int a) const { return Int2(x + a, y + a); }
    INL_HOST_DEVICE Int2 operator - (int a) const { return Int2(x - a, y - a); }

    INL_HOST_DEVICE Int2 operator += (int a) { x += a; y += a; return *this; }
    INL_HOST_DEVICE Int2 operator -= (int a) { x -= a; y -= a; return *this; }

    INL_HOST_DEVICE Int2 operator + (const Int2& v) const { return Int2(x + v.x, y + v.y); }
    INL_HOST_DEVICE Int2 operator - (const Int2& v) const { return Int2(x - v.x, y - v.y); }

    INL_HOST_DEVICE Int2 operator += (const Int2& v) { x += v.x; y += v.y; return *this; }
    INL_HOST_DEVICE Int2 operator -= (const Int2& v) { x -= v.x; y -= v.y; return *this; }

    INL_HOST_DEVICE Int2 operator - () const { return Int2(-x, -y); }

    INL_HOST_DEVICE bool operator == (const Int2& v) { return x == v.x && y == v.y; }
    INL_HOST_DEVICE bool operator != (const Int2& v) { return x != v.x || y != v.y; }

    INL_HOST_DEVICE int& operator[] (int i) { return _v[i]; }
    INL_HOST_DEVICE int  operator[] (int i) const { return _v[i]; }
};

INL_HOST_DEVICE Float2 ToFloat2(const Int2& v) { return Float2((float)v.x, (float)v.y); }

struct UInt2
{
    union
    {
        struct { unsigned int x, y; };
        unsigned int _v[2];
    };

    INL_HOST_DEVICE UInt2() : x{ 0 }, y{ 0 } {}
    INL_HOST_DEVICE UInt2(unsigned int a) : x{ a }, y{ a } {}
    INL_HOST_DEVICE UInt2(unsigned int x, unsigned int y) : x{ x }, y{ y } {}

    INL_HOST_DEVICE UInt2 operator + (unsigned int a) const { return UInt2(x + a, y + a); }
    INL_HOST_DEVICE UInt2 operator - (unsigned int a) const { return UInt2(x - a, y - a); }

    INL_HOST_DEVICE UInt2 operator += (unsigned int a) { x += a; y += a; return *this; }
    INL_HOST_DEVICE UInt2 operator -= (unsigned int a) { x -= a; y -= a; return *this; }

    INL_HOST_DEVICE UInt2 operator + (const UInt2& v) const { return UInt2(x + v.x, y + v.y); }
    INL_HOST_DEVICE UInt2 operator - (const UInt2& v) const { return UInt2(x - v.x, y - v.y); }

    INL_HOST_DEVICE UInt2 operator += (const UInt2& v) { x += v.x; y += v.y; return *this; }
    INL_HOST_DEVICE UInt2 operator -= (const UInt2& v) { x -= v.x; y -= v.y; return *this; }

    INL_HOST_DEVICE bool operator == (const UInt2& v) { return x == v.x && y == v.y; }
    INL_HOST_DEVICE bool operator != (const UInt2& v) { return x != v.x || y != v.y; }

    INL_HOST_DEVICE unsigned int& operator[] (unsigned int i) { return _v[i]; }
    INL_HOST_DEVICE unsigned int  operator[] (unsigned int i) const { return _v[i]; }

    INL_HOST_DEVICE operator Int2() const { return Int2((int)x, (int)y); }
};

INL_HOST_DEVICE Int2 operator + (int a, const Int2& v) { return Int2(v.x + a, v.y + a); }
INL_HOST_DEVICE Int2 operator - (int a, const Int2& v) { return Int2(a - v.x, a - v.y); }
INL_HOST_DEVICE Int2 operator * (int a, const Int2& v) { return Int2(v.x * a, v.y * a); }
INL_HOST_DEVICE Int2 operator / (int a, const Int2& v) { return Int2(a / v.x, a / v.y); }

INL_HOST_DEVICE Float2 operator + (float a, const Int2& v) { return Float2((float)v.x + a, (float)v.y + a); }
INL_HOST_DEVICE Float2 operator - (float a, const Int2& v) { return Float2(a - (float)v.x, a - (float)v.y); }
INL_HOST_DEVICE Float2 operator * (float a, const Int2& v) { return Float2((float)v.x * a, (float)v.y * a); }
INL_HOST_DEVICE Float2 operator / (float a, const Int2& v) { return Float2(a / (float)v.x, a / (float)v.y); }

INL_HOST_DEVICE Float2 operator + (const Float2& vf, const Int2& vi) { return Float2(vf.x + vi.x, vf.y + vi.y); }
INL_HOST_DEVICE Float2 operator - (const Float2& vf, const Int2& vi) { return Float2(vf.x - vi.x, vf.y - vi.y); }
INL_HOST_DEVICE Float2 operator * (const Float2& vf, const Int2& vi) { return Float2(vf.x * vi.x, vf.y * vi.y); }
INL_HOST_DEVICE Float2 operator / (const Float2& vf, const Int2& vi) { return Float2(vf.x / vi.x, vf.y / vi.y); }

INL_HOST_DEVICE Float2 operator + (const Int2& vi, const Float2& vf) { return Float2(vi.x + vf.x, vi.y + vf.y); }
INL_HOST_DEVICE Float2 operator - (const Int2& vi, const Float2& vf) { return Float2(vi.x - vf.x, vi.y - vf.y); }
INL_HOST_DEVICE Float2 operator * (const Int2& vi, const Float2& vf) { return Float2(vi.x * vf.x, vi.y * vf.y); }
INL_HOST_DEVICE Float2 operator / (const Int2& vi, const Float2& vf) { return Float2(vi.x / vf.x, vi.y / vf.y); }

INL_DEVICE float  fract(float a) { float intPart; return modff(a, &intPart); }
INL_DEVICE Float2 floor(const Float2& v) { return Float2(floorf(v.x), floorf(v.y)); }
INL_DEVICE Int2   floori(const Float2& v) { return Int2((int)(floorf(v.x)), (int)(floorf(v.y))); }
INL_DEVICE Float2 fract(const Float2& v) { float intPart; return Float2(modff(v.x, &intPart), modff(v.y, &intPart)); }
INL_DEVICE Int2   roundi(const Float2& v) { return Int2((int)(rintf(v.x)), (int)(rintf(v.y))); }

INL_HOST_DEVICE float max1f(const float& a, const float& b) { return (a < b) ? b : a; }
INL_HOST_DEVICE float min1f(const float& a, const float& b) { return (a > b) ? b : a; }

struct Float3
{
    union
    {
        struct { float x, y, z; };
        struct { Float2 xy; float z; };
        float _v[3];
    };

    INL_HOST_DEVICE Float3() : x(0), y(0), z(0) {}
    INL_HOST_DEVICE Float3(float _x) : x(_x), y(_x), z(_x) {}
    INL_HOST_DEVICE Float3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    INL_HOST_DEVICE Float3(const float4& v) : x(v.x), y(v.y), z(v.z) {}

    INL_HOST_DEVICE Float2 xz() const { return Float2(x, z); }

    INL_HOST_DEVICE Float3  operator+(const Float3& v) const { return Float3(x + v.x, y + v.y, z + v.z); }
    INL_HOST_DEVICE Float3  operator-(const Float3& v) const { return Float3(x - v.x, y - v.y, z - v.z); }
    INL_HOST_DEVICE Float3  operator*(const Float3& v) const { return Float3(x * v.x, y * v.y, z * v.z); }
    INL_HOST_DEVICE Float3  operator/(const Float3& v) const { return Float3(x / v.x, y / v.y, z / v.z); }

    INL_HOST_DEVICE Float3  operator+(float a) const { return Float3(x + a, y + a, z + a); }
    INL_HOST_DEVICE Float3  operator-(float a) const { return Float3(x - a, y - a, z - a); }
    INL_HOST_DEVICE Float3  operator*(float a) const { return Float3(x * a, y * a, z * a); }
    INL_HOST_DEVICE Float3  operator/(float a) const { return Float3(x / a, y / a, z / a); }

    INL_HOST_DEVICE Float3& operator+=(const Float3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    INL_HOST_DEVICE Float3& operator-=(const Float3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    INL_HOST_DEVICE Float3& operator*=(const Float3& v) { x *= v.x; y *= v.y; z *= v.z; return *this; }
    INL_HOST_DEVICE Float3& operator/=(const Float3& v) { x /= v.x; y /= v.y; z /= v.z; return *this; }

    INL_HOST_DEVICE Float3& operator+=(const float& a) { x += a; y += a; z += a; return *this; }
    INL_HOST_DEVICE Float3& operator-=(const float& a) { x -= a; y -= a; z -= a; return *this; }
    INL_HOST_DEVICE Float3& operator*=(const float& a) { x *= a; y *= a; z *= a; return *this; }
    INL_HOST_DEVICE Float3& operator/=(const float& a) { x /= a; y /= a; z /= a; return *this; }

    INL_HOST_DEVICE Float3 operator-() const { return Float3(-x, -y, -z); }

    INL_HOST_DEVICE bool operator!=(const Float3& v) const { return x != v.x || y != v.y || z != v.z; }
    INL_HOST_DEVICE bool operator==(const Float3& v) const { return x == v.x && y == v.y && z == v.z; }

    INL_HOST_DEVICE bool operator<(const Float3& v) const { return x < v.x; }
    INL_HOST_DEVICE bool operator<=(const Float3& v) const { return x <= v.x; }
    INL_HOST_DEVICE bool operator>(const Float3& v) const { return x > v.x; }
    INL_HOST_DEVICE bool operator>=(const Float3& v) const { return x >= v.x; }

    INL_HOST_DEVICE float& operator[](int i) { return _v[i]; }
    INL_HOST_DEVICE float  operator[](int i) const { return _v[i]; }

    INL_HOST_DEVICE operator float4() { return make_float4(x, y, z, 0); }

    INL_HOST_DEVICE float   length() const { return sqrtf(x * x + y * y + z * z); }
    INL_HOST_DEVICE float   length2() const { return x * x + y * y + z * z; }
    INL_HOST_DEVICE float   getmax() const { return max(max(x, y), z); }
    INL_HOST_DEVICE float   getmin() const { return min(min(x, y), z); }
    INL_HOST_DEVICE float   norm() const { float norm = sqrtf(x * x + y * y + z * z); return norm; }
    INL_HOST_DEVICE Float3& normalize() { float norm = sqrtf(x * x + y * y + z * z); x /= norm; y /= norm; z /= norm; return *this; }
    INL_HOST_DEVICE Float3  normalized() const { float norm = sqrtf(x * x + y * y + z * z); return Float3(x / norm, y / norm, z / norm); }
};

struct Float3Hasher
{
    unsigned long long operator() (const Float3& v) const
    {
        union FloatToSizeT
        {
            FloatToSizeT(float a) : a{ a } {}
            float a;
            unsigned long long b;
        };

        unsigned long long h = 17;
        h = (h * 37) ^ FloatToSizeT(v.x).b;
        h = (h * 37) ^ FloatToSizeT(v.y).b;
        h = (h * 37) ^ FloatToSizeT(v.z).b;
        return h;
    }
};

INL_HOST_DEVICE Float3 operator+(float a, const Float3& v) { return Float3(v.x + a, v.y + a, v.z + a); }
INL_HOST_DEVICE Float3 operator-(float a, const Float3& v) { return Float3(a - v.x, a - v.y, a - v.z); }
INL_HOST_DEVICE Float3 operator*(float a, const Float3& v) { return Float3(v.x * a, v.y * a, v.z * a); }
INL_HOST_DEVICE Float3 operator/(float a, const Float3& v) { return Float3(a / v.x, a / v.y, a / v.z); }

INL_HOST_DEVICE Float3 DifferenceOfProducts(Float3 a, float b, Float3 c, float d)
{
    return Float3(DifferenceOfProducts(a.x, b, c.x, d),
        DifferenceOfProducts(a.y, b, c.y, d),
        DifferenceOfProducts(a.z, b, c.z, d));
}

struct Int3
{
    union
    {
        struct { int x, y, z; };
        int _v[3];
    };

    INL_HOST_DEVICE Int3() : x{ 0 }, y{ 0 }, z{ 0 } {}
    INL_HOST_DEVICE Int3(int a) : x{ a }, y{ a }, z{ a } {}
    INL_HOST_DEVICE Int3(int x, int y, int z) : x{ x }, y{ y }, z{ z } {}

    INL_HOST_DEVICE Int3 operator + (int a) const { return Int3(x + a, y + a, z + a); }
    INL_HOST_DEVICE Int3 operator - (int a) const { return Int3(x - a, y - a, z - a); }

    INL_HOST_DEVICE Int3 operator += (int a) { x += a; y += a; z += a; return *this; }
    INL_HOST_DEVICE Int3 operator -= (int a) { x -= a; y -= a; z -= a; return *this; }

    INL_HOST_DEVICE Int3 operator + (const Int3& v) const { return Int3(x + v.x, y + v.y, z + v.y); }
    INL_HOST_DEVICE Int3 operator - (const Int3& v) const { return Int3(x - v.x, y - v.y, z - v.y); }

    INL_HOST_DEVICE Int3 operator += (const Int3& v) { x += v.x; y += v.y; z += v.y; return *this; }
    INL_HOST_DEVICE Int3 operator -= (const Int3& v) { x -= v.x; y -= v.y; z -= v.y; return *this; }

    INL_HOST_DEVICE Int3 operator - () const { return Int3(-x, -y, -z); }

    INL_HOST_DEVICE bool operator == (const Int3& v) { return x == v.x && y == v.y && z == v.z; }
    INL_HOST_DEVICE bool operator != (const Int3& v) { return x != v.x || y != v.y || z != v.z; }

    INL_HOST_DEVICE int& operator[] (int i) { return _v[i]; }
    INL_HOST_DEVICE int  operator[] (int i) const { return _v[i]; }
};

struct UInt3
{
    union
    {
        struct { uint x, y, z; };
        uint _v[3];
    };

    INL_HOST_DEVICE UInt3() : x{ 0 }, y{ 0 }, z{ 0 } {}
    INL_HOST_DEVICE UInt3(uint a) : x{ a }, y{ a }, z{ a } {}
    INL_HOST_DEVICE UInt3(uint x, uint y, uint z) : x{ x }, y{ y }, z{ z } {}

    INL_HOST_DEVICE UInt3 operator + (uint a) const { return UInt3(x + a, y + a, z + a); }
    INL_HOST_DEVICE UInt3 operator - (uint a) const { return UInt3(x - a, y - a, z - a); }

    INL_HOST_DEVICE UInt3 operator += (uint a) { x += a; y += a; z += a; return *this; }
    INL_HOST_DEVICE UInt3 operator -= (uint a) { x -= a; y -= a; z -= a; return *this; }

    INL_HOST_DEVICE UInt3 operator + (const UInt3& v) const { return UInt3(x + v.x, y + v.y, z + v.y); }
    INL_HOST_DEVICE UInt3 operator - (const UInt3& v) const { return UInt3(x - v.x, y - v.y, z - v.y); }

    INL_HOST_DEVICE UInt3 operator += (const UInt3& v) { x += v.x; y += v.y; z += v.y; return *this; }
    INL_HOST_DEVICE UInt3 operator -= (const UInt3& v) { x -= v.x; y -= v.y; z -= v.y; return *this; }

    INL_HOST_DEVICE bool operator == (const UInt3& v) { return x == v.x && y == v.y && z == v.z; }
    INL_HOST_DEVICE bool operator != (const UInt3& v) { return x != v.x || y != v.y || z != v.z; }

    INL_HOST_DEVICE uint& operator[] (uint i) { return _v[i]; }
    INL_HOST_DEVICE uint  operator[] (uint i) const { return _v[i]; }
};

struct Float4
{
    union
    {
        struct { float x, y, z, w; };
        struct { Float3 xyz; float w; };
        struct { Float2 xy; Float2 zw; };
        float _v[4];
    };

    INL_HOST_DEVICE Float4() : x(0), y(0), z(0), w(0) {}
    INL_HOST_DEVICE Float4(float _x) : x(_x), y(_x), z(_x), w(_x) {}
    INL_HOST_DEVICE Float4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}
    INL_HOST_DEVICE Float4(const Float2& v1, const Float2& v2) : x(v1.x), y(v1.y), z(v2.x), w(v2.y) {}
    INL_HOST_DEVICE Float4(const Float3& v) : x(v.x), y(v.y), z(v.z), w(0) {}
    INL_HOST_DEVICE Float4(const Float3& v, float a) : x(v.x), y(v.y), z(v.z), w(a) {}
    INL_HOST_DEVICE Float4(const Float4& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}

    INL_HOST_DEVICE Float4  operator+(const Float4& v) const { return Float4(x + v.x, y + v.y, z + v.z, z + v.z); }
    INL_HOST_DEVICE Float4  operator-(const Float4& v) const { return Float4(x - v.x, y - v.y, z - v.z, z - v.z); }
    INL_HOST_DEVICE Float4  operator*(const Float4& v) const { return Float4(x * v.x, y * v.y, z * v.z, z * v.z); }
    INL_HOST_DEVICE Float4  operator/(const Float4& v) const { return Float4(x / v.x, y / v.y, z / v.z, z / v.z); }

    INL_HOST_DEVICE Float4  operator+(float a) const { return Float4(x + a, y + a, z + a, z + a); }
    INL_HOST_DEVICE Float4  operator-(float a) const { return Float4(x - a, y - a, z - a, z - a); }
    INL_HOST_DEVICE Float4  operator*(float a) const { return Float4(x * a, y * a, z * a, z * a); }
    INL_HOST_DEVICE Float4  operator/(float a) const { return Float4(x / a, y / a, z / a, z / a); }

    INL_HOST_DEVICE Float4& operator+=(const Float4& v) { x += v.x; y += v.y; z += v.z; w += v.w; return *this; }
    INL_HOST_DEVICE Float4& operator-=(const Float4& v) { x -= v.x; y -= v.y; z -= v.z; w -= v.w; return *this; }
    INL_HOST_DEVICE Float4& operator*=(const Float4& v) { x *= v.x; y *= v.y; z *= v.z; w *= v.w; return *this; }
    INL_HOST_DEVICE Float4& operator/=(const Float4& v) { x /= v.x; y /= v.y; z /= v.z; w /= v.w; return *this; }

    INL_HOST_DEVICE Float4& operator+=(const float& a) { x += a; y += a; z += a; w += a; return *this; }
    INL_HOST_DEVICE Float4& operator-=(const float& a) { x -= a; y -= a; z -= a; w += a; return *this; }
    INL_HOST_DEVICE Float4& operator*=(const float& a) { x *= a; y *= a; z *= a; w += a; return *this; }
    INL_HOST_DEVICE Float4& operator/=(const float& a) { x /= a; y /= a; z /= a; w += a; return *this; }

    INL_HOST_DEVICE Float4 operator-() const { return Float4(-x, -y, -z, -w); }

    INL_HOST_DEVICE bool operator!=(const Float4& v) const { return x != v.x || y != v.y || z != v.z || w != v.w; }
    INL_HOST_DEVICE bool operator==(const Float4& v) const { return x == v.x && y == v.y && z == v.z && w == v.w; }

    INL_HOST_DEVICE float& operator[](int i) { return _v[i]; }
    INL_HOST_DEVICE float  operator[](int i) const { return _v[i]; }
};

INL_HOST_DEVICE Float4 operator + (float a, const Float4& v) { return Float4(v.x + a, v.y + a, v.z + a, v.w + a); }
INL_HOST_DEVICE Float4 operator - (float a, const Float4& v) { return Float4(a - v.x, a - v.y, a - v.z, a - v.w); }
INL_HOST_DEVICE Float4 operator * (float a, const Float4& v) { return Float4(v.x * a, v.y * a, v.z * a, v.w * a); }
INL_HOST_DEVICE Float4 operator / (float a, const Float4& v) { return Float4(a / v.x, a / v.y, a / v.z, a / v.w); }

INL_HOST_DEVICE Float3 abs(const Float3& v) { return Float3(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }
INL_HOST_DEVICE Float2 normalize(const Float2& v) { float norm = sqrtf(v.x * v.x + v.y * v.y); return Float2(v.x / norm, v.y / norm); }
INL_HOST_DEVICE Float3 normalize(const Float3& v) { float norm = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); return Float3(v.x / norm, v.y / norm, v.z / norm); }
INL_HOST_DEVICE Float3 sqrt3f(const Float3& v) { return Float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z)); }
INL_HOST_DEVICE Float3 rsqrt3f(const Float3& v) { return Float3(1.0f / sqrtf(v.x), 1.0f / sqrtf(v.y), 1.0f / sqrtf(v.z)); }
INL_HOST_DEVICE Float3 min3f(const Float3& v1, const Float3& v2) { return Float3(min(v1.x, v2.x), min(v1.y, v2.y), min(v1.z, v2.z)); }
INL_HOST_DEVICE Float3 max3f(const Float3& v1, const Float3& v2) { return Float3(max(v1.x, v2.x), max(v1.y, v2.y), max(v1.z, v2.z)); }
INL_HOST_DEVICE Float3 cross(const Float3& v1, const Float3& v2) { return Float3(dop(v1.y, v2.z, v1.z, v2.y), dop(v1.z, v2.x, v1.x, v2.z), dop(v1.x, v2.y, v1.y, v2.x)); /*Float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);*/ }
INL_HOST_DEVICE Float3 pow3f(const Float3& v1, const Float3& v2) { return Float3(powf(v1.x, v2.x), powf(v1.y, v2.y), powf(v1.z, v2.z)); }
INL_HOST_DEVICE Float3 exp3f(const Float3& v) { return Float3(expf(v.x), expf(v.y), expf(v.z)); }
INL_HOST_DEVICE Float3 pow3f(const Float3& v, float a) { return Float3(powf(v.x, a), powf(v.y, a), powf(v.z, a)); }
INL_HOST_DEVICE Float3 sin3f(const Float3& v) { return Float3(sinf(v.x), sinf(v.y), sinf(v.z)); }
INL_HOST_DEVICE Float3 cos3f(const Float3& v) { return Float3(cosf(v.x), cosf(v.y), cosf(v.z)); }
INL_HOST_DEVICE Float3 mixf(const Float3& v1, const Float3& v2, float a) { return v1 * (1.0f - a) + v2 * a; }
INL_HOST_DEVICE float  mix1f(float v1, float v2, float a) { return v1 * (1.0f - a) + v2 * a; }
INL_HOST_DEVICE Float3 minf3f(const float a, const Float3& v) { return Float3(v.x < a ? v.x : a, v.y < a ? v.y : a, v.y < a ? v.y : a); }
INL_HOST_DEVICE void   swap(int& v1, int& v2) { int tmp = v1; v1 = v2; v2 = tmp; }
INL_HOST_DEVICE void   swap(float& v1, float& v2) { float tmp = v1; v1 = v2; v2 = tmp; }
INL_HOST_DEVICE void   swap(Float3& v1, Float3& v2) { Float3 tmp = v1; v1 = v2; v2 = tmp; }
INL_HOST_DEVICE int    clampi(int a, int lo = 0, int hi = 1) { return a < lo ? lo : a > hi ? hi : a; }
INL_HOST_DEVICE float  clampf(float a, float lo = 0.0f, float hi = 1.0f) { return a < lo ? lo : a > hi ? hi : a; }
INL_HOST_DEVICE Float3 clamp3f(Float3 a, Float3 lo = Float3(0.0f), Float3 hi = Float3(1.0f)) { return Float3(clampf(a.x, lo.x, hi.x), clampf(a.y, lo.y, hi.y), clampf(a.z, lo.z, hi.z)); }
INL_HOST_DEVICE Float4 clamp4f(Float4 a, Float4 lo = Float4(0.0f), Float4 hi = Float4(1.0f)) { return Float4(clampf(a.x, lo.x, hi.x), clampf(a.y, lo.y, hi.y), clampf(a.z, lo.z, hi.z), clampf(a.w, lo.w, hi.w)); }
INL_HOST_DEVICE float  dot(const Float2& v1, const Float2& v2) { return v1.x * v2.x + v1.y * v2.y; }
INL_HOST_DEVICE float  dot(const Float3& v1, const Float3& v2) { return (float)InnerProduct(v1.x, v2.x, v1.y, v2.y, v1.z, v2.z); /*v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;*/ }
INL_HOST_DEVICE float  dot(const Float4& v1, const Float4& v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w; }
INL_HOST_DEVICE float  distancesq(const Float3& v1, const Float3& v2) { return (v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z) * (v1.z - v2.z); }
INL_HOST_DEVICE float  distance(const Float3& v1, const Float3& v2) { return sqrtf((v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z) * (v1.z - v2.z)); }
INL_HOST_DEVICE Float3 lerp3f(Float3 a, Float3 b, float w) { return a + w * (b - a); }
INL_HOST_DEVICE float  lerpf(float a, float b, float w) { return a + w * (b - a); }
INL_HOST_DEVICE Float3 reflect3f(Float3 i, Float3 n) { return i - 2.0f * n * dot(n, i); }
INL_HOST_DEVICE float  pow2(float a) { return a * a; }
INL_HOST_DEVICE float  pow3(float a) { return a * a * a; }

INL_HOST_DEVICE float smoothstep1f(float a, float b, float w) { return a + (w * w * (3.0f - 2.0f * w)) * (b - a); }
INL_HOST_DEVICE Float3 smoothstep3f(Float3 a, Float3 b, float w) { return a + (w * w * (3.0f - 2.0f * w)) * (b - a); }

INL_HOST_DEVICE float AngleBetween(const Float3& a, const Float3& b) { return acos(dot(a, b) / sqrtf(a.length2() * b.length2())); }

INL_DEVICE float min3(float v1, float v2, float v3) { return min(min(v1, v2), v3); }
INL_DEVICE float max3(float v1, float v2, float v3) { return max(max(v1, v2), v3); }

struct Mat3
{
    union
    {
        struct
        {
            float m00, m10, m20;
            float m01, m11, m21;
            float m02, m12, m22;
        };
        struct
        {
            Float3 v0, v1, v2;
        };
        Float3 _v3[3];
        float _v[9];
    };
    INL_HOST_DEVICE Mat3() { for (int i = 0; i < 9; ++i) _v[i] = 0; }
    INL_HOST_DEVICE Mat3(const Float3& v0, const Float3& v1, const Float3& v2) : v0{ v0 }, v1{ v1 }, v2{ v2 } {}

    // column-major matrix construction
    INL_HOST_DEVICE constexpr Mat3(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22)
        : m00{ m00 }, m01{ m01 }, m02{ m02 }, m10{ m10 }, m11{ m11 }, m12{ m12 }, m20{ m20 }, m21{ m21 }, m22{ m22 }
    {}

    INL_HOST_DEVICE Float3& operator[](int i) { return _v3[i]; }
    INL_HOST_DEVICE const Float3& operator[](int i) const { return _v3[i]; }

    INL_HOST_DEVICE void transpose() { swap(m01, m10); swap(m20, m02); swap(m21, m12); }
};

// column major multiply
INL_HOST_DEVICE Float3 operator*(const Mat3& m, const Float3& v)
{
    return Float3((float)InnerProduct(m.m00, v[0], m.m01, v[1], m.m02, v[2]),
        (float)InnerProduct(m.m10, v[0], m.m11, v[1], m.m12, v[2]),
        (float)InnerProduct(m.m20, v[0], m.m21, v[1], m.m22, v[2])); // return m.v0 * v.x + m.v1 * v.y + m.v2 * v.z;
}

// Rotation matrix
INL_HOST_DEVICE Mat3 RotationMatrixX(float a) { return { Float3(1, 0, 0), Float3(0, cosf(a), sinf(a)), Float3(0, -sinf(a), cosf(a)) }; }
INL_HOST_DEVICE Mat3 RotationMatrixY(float a) { return { Float3(cosf(a), 0, -sinf(a)), Float3(0, 1, 0), Float3(sinf(a), 0, cosf(a)) }; }
INL_HOST_DEVICE Mat3 RotationMatrixZ(float a) { return { Float3(cosf(a), sinf(a), 0), Float3(-sinf(a), cosf(a), 0), Float3(0, 0, 1) }; }

INL_HOST_DEVICE float Determinant(const Mat3& m)
{
    float minor12 = DifferenceOfProducts(m[1][1], m[2][2], m[1][2], m[2][1]);
    float minor02 = DifferenceOfProducts(m[1][0], m[2][2], m[1][2], m[2][0]);
    float minor01 = DifferenceOfProducts(m[1][0], m[2][1], m[1][1], m[2][0]);
    return FMA(m[0][2], minor01, DifferenceOfProducts(m[0][0], minor12, m[0][1], minor02));
}

INL_HOST_DEVICE Mat3 Inverse(const Mat3& m)
{
    float det = Determinant(m);
    if (det == 0)
        return {};
    float invDet = 1 / det;

    Mat3 r;

    r[0][0] = invDet * DifferenceOfProducts(m[1][1], m[2][2], m[1][2], m[2][1]);
    r[1][0] = invDet * DifferenceOfProducts(m[1][2], m[2][0], m[1][0], m[2][2]);
    r[2][0] = invDet * DifferenceOfProducts(m[1][0], m[2][1], m[1][1], m[2][0]);
    r[0][1] = invDet * DifferenceOfProducts(m[0][2], m[2][1], m[0][1], m[2][2]);
    r[1][1] = invDet * DifferenceOfProducts(m[0][0], m[2][2], m[0][2], m[2][0]);
    r[2][1] = invDet * DifferenceOfProducts(m[0][1], m[2][0], m[0][0], m[2][1]);
    r[0][2] = invDet * DifferenceOfProducts(m[0][1], m[1][2], m[0][2], m[1][1]);
    r[1][2] = invDet * DifferenceOfProducts(m[0][2], m[1][0], m[0][0], m[1][2]);
    r[2][2] = invDet * DifferenceOfProducts(m[0][0], m[1][1], m[0][1], m[1][0]);

    return r;
}

struct Mat4
{
    union
    {
        struct
        {
            float m00, m10, m20, m30;
            float m01, m11, m21, m31;
            float m02, m12, m22, m32;
            float m03, m13, m23, m33;
        };
        float _v[16];
    };

    INL_HOST_DEVICE Mat4() { for (int i = 0; i < 16; ++i) { _v[i] = 0; } m00 = m11 = m22 = m33 = 1; }
    INL_HOST_DEVICE Mat4(const Mat4& m) { for (int i = 0; i < 16; ++i) { _v[i] = m._v[i]; } }

    // row
    INL_HOST_DEVICE void   setRow(uint i, const Float4& v) { /*assert(i < 4);*/ _v[i] = v[0]; _v[i + 4] = v[1]; _v[i + 8] = v[2]; _v[i + 12] = v[3]; }
    INL_HOST_DEVICE Float4 getRow(uint i) const { /*assert(i < 4);*/ return Float4(_v[i], _v[i + 4], _v[i + 8], _v[i + 12]); }

    // column
    INL_HOST_DEVICE void   setCol(uint i, const Float4& v) { /*assert(i < 4);*/ _v[i * 4] = v[0]; _v[i * 4 + 1] = v[1]; _v[i * 4 + 2] = v[2]; _v[i * 4 + 3] = v[3]; }
    INL_HOST_DEVICE Float4 getCol(uint i) const { /*assert(i < 4);*/ return Float4(_v[i * 4], _v[i * 4 + 1], _v[i * 4 + 2], _v[i * 4 + 3]); }

    // element
    INL_HOST_DEVICE void   set(uint r, uint c, float v) { /*assert(r < 4 && c < 4);*/ _v[r + c * 4] = v; }
    INL_HOST_DEVICE float  get(uint r, uint c) const { /*assert(r < 4 && c < 4);*/ return _v[r + c * 4]; }

    INL_HOST_DEVICE const float* operator[](uint i) const { return _v + i * 4; }
    INL_HOST_DEVICE float* operator[](uint i) { return _v + i * 4; }
};

INL_HOST_DEVICE Mat4 invert(const Mat4& m)
{
    float s0 = DifferenceOfProducts(m[0][0], m[1][1], m[1][0], m[0][1]);
    float s1 = DifferenceOfProducts(m[0][0], m[1][2], m[1][0], m[0][2]);
    float s2 = DifferenceOfProducts(m[0][0], m[1][3], m[1][0], m[0][3]);

    float s3 = DifferenceOfProducts(m[0][1], m[1][2], m[1][1], m[0][2]);
    float s4 = DifferenceOfProducts(m[0][1], m[1][3], m[1][1], m[0][3]);
    float s5 = DifferenceOfProducts(m[0][2], m[1][3], m[1][2], m[0][3]);

    float c0 = DifferenceOfProducts(m[2][0], m[3][1], m[3][0], m[2][1]);
    float c1 = DifferenceOfProducts(m[2][0], m[3][2], m[3][0], m[2][2]);
    float c2 = DifferenceOfProducts(m[2][0], m[3][3], m[3][0], m[2][3]);

    float c3 = DifferenceOfProducts(m[2][1], m[3][2], m[3][1], m[2][2]);
    float c4 = DifferenceOfProducts(m[2][1], m[3][3], m[3][1], m[2][3]);
    float c5 = DifferenceOfProducts(m[2][2], m[3][3], m[3][2], m[2][3]);

    float determinant = (float)InnerProduct(s0, c5, -s1, c4, s2, c3, s3, c2, s5, c0, -s4, c1);
    if (determinant == 0)
        return {};
    float s = 1 / determinant;

    Mat4 inv;

    inv[0][0] = s * (float)InnerProduct(m[1][1], c5, m[1][3], c3, -m[1][2], c4);
    inv[0][1] = s * (float)InnerProduct(-m[0][1], c5, m[0][2], c4, -m[0][3], c3);
    inv[0][2] = s * (float)InnerProduct(m[3][1], s5, m[3][3], s3, -m[3][2], s4);
    inv[0][3] = s * (float)InnerProduct(-m[2][1], s5, m[2][2], s4, -m[2][3], s3);

    inv[1][0] = s * (float)InnerProduct(-m[1][0], c5, m[1][2], c2, -m[1][3], c1);
    inv[1][1] = s * (float)InnerProduct(m[0][0], c5, m[0][3], c1, -m[0][2], c2);
    inv[1][2] = s * (float)InnerProduct(-m[3][0], s5, m[3][2], s2, -m[3][3], s1);
    inv[1][3] = s * (float)InnerProduct(m[2][0], s5, m[2][3], s1, -m[2][2], s2);

    inv[2][0] = s * (float)InnerProduct(m[1][0], c4, m[1][3], c0, -m[1][1], c2);
    inv[2][1] = s * (float)InnerProduct(-m[0][0], c4, m[0][1], c2, -m[0][3], c0);
    inv[2][2] = s * (float)InnerProduct(m[3][0], s4, m[3][3], s0, -m[3][1], s2);
    inv[2][3] = s * (float)InnerProduct(-m[2][0], s4, m[2][1], s2, -m[2][3], s0);

    inv[3][0] = s * (float)InnerProduct(-m[1][0], c3, m[1][1], c1, -m[1][2], c0);
    inv[3][1] = s * (float)InnerProduct(m[0][0], c3, m[0][2], c0, -m[0][1], c1);
    inv[3][2] = s * (float)InnerProduct(-m[3][0], s3, m[3][1], s1, -m[3][2], s0);
    inv[3][3] = s * (float)InnerProduct(m[2][0], s3, m[2][2], s0, -m[2][1], s1);

    return inv;
}

struct Quat
{
    union
    {
        Float3 v;
        struct
        {
            float x, y, z;
        };
    };
    float w;

    INL_HOST_DEVICE Quat() : v(), w(0) {}
    INL_HOST_DEVICE Quat(const Float3& v) : v(v), w(0) {}
    INL_HOST_DEVICE Quat(const Float3& v, float w) : v(v), w(w) {}
    INL_HOST_DEVICE Quat(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

    static INL_HOST_DEVICE Quat axisAngle(const Float3& axis, float angle) { return Quat(axis.normalized() * sinf(angle / 2), cosf(angle / 2)); }

    INL_HOST_DEVICE Quat  conj() const { return Quat(-v, w); }
    INL_HOST_DEVICE float norm2() const { return x * x + y * y + z * z + w * w; }
    INL_HOST_DEVICE Quat  inv() const { return conj() / norm2(); }
    INL_HOST_DEVICE float norm() const { return sqrtf(norm2()); }
    INL_HOST_DEVICE Quat  normalized() const { float n = norm(); return Quat(v / n, w / n); }
    INL_HOST_DEVICE Quat  pow(float a) const { return Quat::axisAngle(v, acosf(w) * a * 2); }

    INL_HOST_DEVICE Quat operator/  (float a) const { return Quat(v / a, w / a); }
    INL_HOST_DEVICE Quat operator+  (const Quat& q) const { const Quat& p = *this; return Quat(p.v + q.v, p.w + q.w); }
    INL_HOST_DEVICE Quat operator*  (const Quat& q) const { const Quat& p = *this; return Quat(p.w * q.v + q.w * p.v + cross(p.v, q.v), p.w * q.w - dot(p.v, q.v)); }
    INL_HOST_DEVICE Quat& operator+=(const Quat& q) { Quat ret = *this + q; return (*this = ret); }
    INL_HOST_DEVICE Quat& operator*=(const Quat& q) { Quat ret = *this * q; return (*this = ret); }
};

INL_HOST_DEVICE Quat rotate(const Quat& q, const Quat& v) { return q * v * q.conj(); }
INL_HOST_DEVICE Quat slerp(const Quat& q, const Quat& r, float t) { return (r * q.conj()).pow(t) * q; }
INL_HOST_DEVICE Quat rotationBetween(const Quat& p, const Quat& q) { return Quat(cross(p.v, q.v), sqrtf(p.v.length2() * q.v.length2()) + dot(p.v, q.v)).normalized(); }

INL_HOST_DEVICE Float3 rotate3f(const Float3& axis, float angle, const Float3& v) { return rotate(Quat::axisAngle(axis, angle), v).v; }
INL_HOST_DEVICE Float3 slerp3f(const Float3& q, const Float3& r, float t) { return slerp(Quat(q), Quat(r), t).v; }
INL_HOST_DEVICE Float3 rotationBetween3f(const Float3& p, const Float3& q) { return rotationBetween(Quat(p), Quat(q)).v; }

INL_HOST_DEVICE float SafeDivide(float a, float b) { float eps = 1e-20f; return a / ((fabsf(b) > eps) ? b : copysignf(eps, b)); };
INL_HOST_DEVICE Float3 SafeDivide3f1f(const Float3& a, float b) { return Float3(SafeDivide(a.x, b), SafeDivide(a.y, b), SafeDivide(a.z, b)); };
INL_HOST_DEVICE Float3 SafeDivide3f(const Float3& a, const Float3& b) { return Float3(SafeDivide(a.x, b.x), SafeDivide(a.y, b.y), SafeDivide(a.z, b.z)); };

INL_DEVICE void LocalizeSample(
    const Float3& n,
    Float3& u,
    Float3& v)
{
    Float3 w = Float3(1, 0, 0);

    if (abs(n.x) > 0.707f)
        w = Float3(0, 1, 0);

    u = cross(n, w);
    v = cross(n, u);
}

INL_DEVICE float luminance(Float3 v)
{
    return dot(v, Float3(0.2126f, 0.7152f, 0.0722f));
}

}