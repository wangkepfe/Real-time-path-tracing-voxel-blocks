#pragma once

#include <math.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "Common.h"

#define PI_OVER_4 0.7853981633974483096156608458198757210492f
#define PI_OVER_2 1.5707963267948966192313216916397514420985f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define TWO_PI 6.2831853071795864769252867665590057683943f
#define Pi_over_180 0.01745329251f
#define INV_PI 0.31830988618f
#define INV_TWO_PI 0.15915494309f
#ifndef M_PI
#define M_PI 3.1415926535897932384626422832795028841971f
#endif
#ifndef M_Ef
#define M_Ef 2.71828182845904523536f
#endif
#ifndef M_LOG2Ef
#define M_LOG2Ef 1.44269504088896340736f
#endif
#ifndef M_LOG10Ef
#define M_LOG10Ef 0.434294481903251827651f
#endif
#ifndef M_LN2f
#define M_LN2f 0.693147180559945309417f
#endif
#ifndef M_LN10f
#define M_LN10f 2.30258509299404568402f
#endif
#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif
#ifndef M_PI_2f
#define M_PI_2f 1.57079632679489661923f
#endif
#ifndef M_PI_4f
#define M_PI_4f 0.785398163397448309616f
#endif
#ifndef M_1_PIf
#define M_1_PIf 0.318309886183790671538f
#endif
#ifndef M_2_PIf
#define M_2_PIf 0.636619772367581343076f
#endif
#ifndef M_2_SQRTPIf
#define M_2_SQRTPIf 1.12837916709551257390f
#endif
#ifndef M_SQRT2f
#define M_SQRT2f 1.41421356237309504880f
#endif
#ifndef M_SQRT1_2f
#define M_SQRT1_2f 0.707106781186547524401f
#endif
#ifndef SAFE_COSINE_EPSI
#define SAFE_COSINE_EPSI 1e-5f
#endif
static constexpr float FloatOneMinusEpsilon = 0x1.fffffep-1;
static constexpr float OneMinusEpsilon = FloatOneMinusEpsilon;

struct Float3;
struct Int2;
struct UInt2;

template <typename T>
INL_HOST_DEVICE T max(const T &a, const T &b) { return a > b ? a : b; }

template <typename T>
INL_HOST_DEVICE T min(const T &a, const T &b) { return a < b ? a : b; }

template <typename T>
INL_HOST_DEVICE T max(const T &a, const T &b, const T &c) { return max(max(a, b), c); }

template <typename T>
INL_HOST_DEVICE T min(const T &a, const T &b, const T &c) { return min(min(a, b), c); }

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
    return {ab, FMA(a, b, -ab)};
}

INL_HOST_DEVICE CompensatedFloat TwoSum(float a, float b)
{
    float s = a + b, delta = s - a;
    return {s, (a - (s - delta)) + (b - delta)};
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
    return {sum.v, ab.err + (tp.err + sum.err)};
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
        struct
        {
            float x, y;
        };
        float _v[2];
    };

    INL_HOST_DEVICE Float2() : x(0), y(0) {}
    INL_HOST_DEVICE Float2(float _x, float _y) : x(_x), y(_y) {}
    INL_HOST_DEVICE explicit Float2(float _x) : x(_x), y(_x) {}
    INL_HOST_DEVICE explicit Float2(const float2 &v) : x(v.x), y(v.y) {}

    INL_HOST_DEVICE Float2 operator+(const Float2 &v) const { return Float2(x + v.x, y + v.y); }
    INL_HOST_DEVICE Float2 operator-(const Float2 &v) const { return Float2(x - v.x, y - v.y); }
    INL_HOST_DEVICE Float2 operator*(const Float2 &v) const { return Float2(x * v.x, y * v.y); }
    INL_HOST_DEVICE Float2 operator/(const Float2 &v) const { return Float2(x / v.x, y / v.y); }

    INL_HOST_DEVICE Float2 operator+(float a) const { return Float2(x + a, y + a); }
    INL_HOST_DEVICE Float2 operator-(float a) const { return Float2(x - a, y - a); }
    INL_HOST_DEVICE Float2 operator*(float a) const { return Float2(x * a, y * a); }
    INL_HOST_DEVICE Float2 operator/(float a) const { return Float2(x / a, y / a); }

    INL_HOST_DEVICE Float2 &operator+=(const Float2 &v)
    {
        x += v.x;
        y += v.y;
        return *this;
    }
    INL_HOST_DEVICE Float2 &operator-=(const Float2 &v)
    {
        x -= v.x;
        y -= v.y;
        return *this;
    }
    INL_HOST_DEVICE Float2 &operator*=(const Float2 &v)
    {
        x *= v.x;
        y *= v.y;
        return *this;
    }
    INL_HOST_DEVICE Float2 &operator/=(const Float2 &v)
    {
        x /= v.x;
        y /= v.y;
        return *this;
    }

    INL_HOST_DEVICE Float2 &operator+=(const float &a)
    {
        x += a;
        y += a;
        return *this;
    }
    INL_HOST_DEVICE Float2 &operator-=(const float &a)
    {
        x -= a;
        y -= a;
        return *this;
    }
    INL_HOST_DEVICE Float2 &operator*=(const float &a)
    {
        x *= a;
        y *= a;
        return *this;
    }
    INL_HOST_DEVICE Float2 &operator/=(const float &a)
    {
        x /= a;
        y /= a;
        return *this;
    }

    INL_HOST_DEVICE Float2 operator-() const { return Float2(-x, -y); }

    INL_HOST_DEVICE bool operator!=(const Float2 &v) const { return x != v.x || y != v.y; }
    INL_HOST_DEVICE bool operator==(const Float2 &v) const { return x == v.x && y == v.y; }

    INL_HOST_DEVICE float &operator[](int i) { return _v[i]; }
    INL_HOST_DEVICE float operator[](int i) const { return _v[i]; }

    INL_HOST_DEVICE explicit operator float2() const { return make_float2(x, y); }

    INL_HOST_DEVICE float length() const { return sqrtf(x * x + y * y); }
    INL_HOST_DEVICE float length2() const { return x * x + y * y; }
};

INL_HOST_DEVICE Float2 operator+(float a, const Float2 &v) { return Float2(v.x + a, v.y + a); }
INL_HOST_DEVICE Float2 operator-(float a, const Float2 &v) { return Float2(a - v.x, a - v.y); }
INL_HOST_DEVICE Float2 operator*(float a, const Float2 &v) { return Float2(v.x * a, v.y * a); }
INL_HOST_DEVICE Float2 operator/(float a, const Float2 &v) { return Float2(a / v.x, a / v.y); }

INL_HOST_DEVICE float length(const Float2 &v) { return sqrtf(v.x * v.x + v.y * v.y); }

struct Int2
{
    union
    {
        struct
        {
            int x, y;
        };
        int _v[2];
    };

    INL_HOST_DEVICE Int2() : x{0}, y{0} {}
    INL_HOST_DEVICE explicit Int2(int a) : x{a}, y{a} {}
    INL_HOST_DEVICE Int2(int x, int y) : x{x}, y{y} {}
    INL_HOST_DEVICE explicit Int2(const uint2 &v) : x{(int)v.x}, y{(int)v.y} {}
    INL_HOST_DEVICE explicit Int2(const uint3 &v) : x{(int)v.x}, y{(int)v.y} {}
    INL_HOST_DEVICE explicit Int2(Float2 a) : x{(int)a.x}, y{(int)a.y} {}

    INL_HOST_DEVICE Int2 operator+(int a) const { return Int2(x + a, y + a); }
    INL_HOST_DEVICE Int2 operator-(int a) const { return Int2(x - a, y - a); }
    INL_HOST_DEVICE Int2 operator*(int a) const { return Int2(x * a, y * a); }
    INL_HOST_DEVICE Int2 operator/(int a) const { return Int2(x / a, y / a); }
    INL_HOST_DEVICE Int2 operator%(int a) const { return Int2(x % a, y % a); }

    INL_HOST_DEVICE Int2 operator+=(int a)
    {
        x += a;
        y += a;
        return *this;
    }
    INL_HOST_DEVICE Int2 operator-=(int a)
    {
        x -= a;
        y -= a;
        return *this;
    }
    INL_HOST_DEVICE Int2 operator*=(int a)
    {
        x *= a;
        y *= a;
        return *this;
    }
    INL_HOST_DEVICE Int2 operator/=(int a)
    {
        x /= a;
        y /= a;
        return *this;
    }
    INL_HOST_DEVICE Int2 operator%=(int a)
    {
        x %= a;
        y %= a;
        return *this;
    }

    INL_HOST_DEVICE Int2 operator+(const Int2 &v) const { return Int2(x + v.x, y + v.y); }
    INL_HOST_DEVICE Int2 operator-(const Int2 &v) const { return Int2(x - v.x, y - v.y); }

    INL_HOST_DEVICE Int2 operator+=(const Int2 &v)
    {
        x += v.x;
        y += v.y;
        return *this;
    }
    INL_HOST_DEVICE Int2 operator-=(const Int2 &v)
    {
        x -= v.x;
        y -= v.y;
        return *this;
    }

    INL_HOST_DEVICE Int2 operator-() const { return Int2(-x, -y); }

    INL_HOST_DEVICE bool operator==(const Int2 &v) { return x == v.x && y == v.y; }
    INL_HOST_DEVICE bool operator!=(const Int2 &v) { return x != v.x || y != v.y; }

    INL_HOST_DEVICE int &operator[](int i) { return _v[i]; }
    INL_HOST_DEVICE int operator[](int i) const { return _v[i]; }
};

INL_HOST_DEVICE Float2 ToFloat2(const Int2 &v) { return Float2((float)v.x, (float)v.y); }

struct UInt2
{
    union
    {
        struct
        {
            unsigned int x, y;
        };
        unsigned int _v[2];
    };

    INL_HOST_DEVICE UInt2() : x{0}, y{0} {}
    INL_HOST_DEVICE explicit UInt2(unsigned int a) : x{a}, y{a} {}
    INL_HOST_DEVICE UInt2(unsigned int x, unsigned int y) : x{x}, y{y} {}
    INL_HOST_DEVICE explicit UInt2(Int2 a) : x{(unsigned int)a.x}, y{(unsigned int)a.y} {}
    INL_HOST_DEVICE explicit UInt2(const uint2 &v) : x{v.x}, y{v.y} {}
    INL_HOST_DEVICE explicit UInt2(const uint3 &v) : x{v.x}, y{v.y} {}

    INL_HOST_DEVICE UInt2 operator+(unsigned int a) const { return UInt2(x + a, y + a); }
    INL_HOST_DEVICE UInt2 operator-(unsigned int a) const { return UInt2(x - a, y - a); }
    INL_HOST_DEVICE UInt2 operator*(unsigned int a) const { return UInt2(x * a, y * a); }
    INL_HOST_DEVICE UInt2 operator/(unsigned int a) const { return UInt2(x / a, y / a); }
    INL_HOST_DEVICE UInt2 operator%(unsigned int a) const { return UInt2(x % a, y % a); }

    INL_HOST_DEVICE UInt2 operator+=(unsigned int a)
    {
        x += a;
        y += a;
        return *this;
    }
    INL_HOST_DEVICE UInt2 operator-=(unsigned int a)
    {
        x -= a;
        y -= a;
        return *this;
    }

    INL_HOST_DEVICE UInt2 operator+(const UInt2 &v) const { return UInt2(x + v.x, y + v.y); }
    INL_HOST_DEVICE UInt2 operator-(const UInt2 &v) const { return UInt2(x - v.x, y - v.y); }

    INL_HOST_DEVICE UInt2 operator+=(const UInt2 &v)
    {
        x += v.x;
        y += v.y;
        return *this;
    }
    INL_HOST_DEVICE UInt2 operator-=(const UInt2 &v)
    {
        x -= v.x;
        y -= v.y;
        return *this;
    }

    INL_HOST_DEVICE bool operator==(const UInt2 &v) { return x == v.x && y == v.y; }
    INL_HOST_DEVICE bool operator!=(const UInt2 &v) { return x != v.x || y != v.y; }

    INL_HOST_DEVICE unsigned int &operator[](unsigned int i) { return _v[i]; }
    INL_HOST_DEVICE unsigned int operator[](unsigned int i) const { return _v[i]; }

    INL_HOST_DEVICE explicit operator Int2() const { return Int2((int)x, (int)y); }
};

INL_HOST_DEVICE Int2 operator+(int a, const Int2 &v) { return Int2(v.x + a, v.y + a); }
INL_HOST_DEVICE Int2 operator-(int a, const Int2 &v) { return Int2(a - v.x, a - v.y); }
INL_HOST_DEVICE Int2 operator*(int a, const Int2 &v) { return Int2(v.x * a, v.y * a); }
INL_HOST_DEVICE Int2 operator/(int a, const Int2 &v) { return Int2(a / v.x, a / v.y); }

INL_HOST_DEVICE Float2 operator+(float a, const Int2 &v) { return Float2((float)v.x + a, (float)v.y + a); }
INL_HOST_DEVICE Float2 operator-(float a, const Int2 &v) { return Float2(a - (float)v.x, a - (float)v.y); }
INL_HOST_DEVICE Float2 operator*(float a, const Int2 &v) { return Float2((float)v.x * a, (float)v.y * a); }
INL_HOST_DEVICE Float2 operator/(float a, const Int2 &v) { return Float2(a / (float)v.x, a / (float)v.y); }

INL_HOST_DEVICE Float2 operator+(const Float2 &vf, const Int2 &vi) { return Float2(vf.x + vi.x, vf.y + vi.y); }
INL_HOST_DEVICE Float2 operator-(const Float2 &vf, const Int2 &vi) { return Float2(vf.x - vi.x, vf.y - vi.y); }
INL_HOST_DEVICE Float2 operator*(const Float2 &vf, const Int2 &vi) { return Float2(vf.x * vi.x, vf.y * vi.y); }
INL_HOST_DEVICE Float2 operator/(const Float2 &vf, const Int2 &vi) { return Float2(vf.x / vi.x, vf.y / vi.y); }

INL_HOST_DEVICE Float2 operator+(const Int2 &vi, const Float2 &vf) { return Float2(vi.x + vf.x, vi.y + vf.y); }
INL_HOST_DEVICE Float2 operator-(const Int2 &vi, const Float2 &vf) { return Float2(vi.x - vf.x, vi.y - vf.y); }
INL_HOST_DEVICE Float2 operator*(const Int2 &vi, const Float2 &vf) { return Float2(vi.x * vf.x, vi.y * vf.y); }
INL_HOST_DEVICE Float2 operator/(const Int2 &vi, const Float2 &vf) { return Float2(vi.x / vf.x, vi.y / vf.y); }

INL_DEVICE float fract(float a)
{
    float intPart;
    return modff(a, &intPart);
}
INL_DEVICE Float2 floor(const Float2 &v) { return Float2(floorf(v.x), floorf(v.y)); }
INL_DEVICE Int2 floori(const Float2 &v) { return Int2((int)(floorf(v.x)), (int)(floorf(v.y))); }
INL_DEVICE Float2 fract(const Float2 &v)
{
    float intPart;
    return Float2(modff(v.x, &intPart), modff(v.y, &intPart));
}
INL_DEVICE Int2 roundi(const Float2 &v) { return Int2((int)(rintf(v.x)), (int)(rintf(v.y))); }

INL_HOST_DEVICE float max1f(const float &a, const float &b) { return (a < b) ? b : a; }
INL_HOST_DEVICE float min1f(const float &a, const float &b) { return (a > b) ? b : a; }

struct Float3
{
    union
    {
        struct
        {
            float x, y, z;
        };
        struct
        {
            Float2 xy;
            float z;
        };
        float _v[3];
    };

    INL_HOST_DEVICE Float3() : x(0), y(0), z(0) {}
    INL_HOST_DEVICE explicit Float3(float _x) : x(_x), y(_x), z(_x) {}
    INL_HOST_DEVICE Float3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    INL_HOST_DEVICE explicit Float3(const Float2 &v, float z) : x(v.x), y(v.y), z(z) {}
    INL_HOST_DEVICE explicit Float3(const float3 &v) : x(v.x), y(v.y), z(v.z) {}
    INL_HOST_DEVICE explicit Float3(const float4 &v) : x(v.x), y(v.y), z(v.z) {}

    INL_HOST_DEVICE float3 to_float3() const { return make_float3(x, y, z); }

    INL_HOST_DEVICE Float2 xz() const { return Float2(x, z); }

    INL_HOST_DEVICE Float3 operator+(const Float3 &v) const { return Float3(x + v.x, y + v.y, z + v.z); }
    INL_HOST_DEVICE Float3 operator-(const Float3 &v) const { return Float3(x - v.x, y - v.y, z - v.z); }
    INL_HOST_DEVICE Float3 operator*(const Float3 &v) const { return Float3(x * v.x, y * v.y, z * v.z); }
    INL_HOST_DEVICE Float3 operator/(const Float3 &v) const { return Float3(x / v.x, y / v.y, z / v.z); }

    INL_HOST_DEVICE Float3 operator+(float a) const { return Float3(x + a, y + a, z + a); }
    INL_HOST_DEVICE Float3 operator-(float a) const { return Float3(x - a, y - a, z - a); }
    INL_HOST_DEVICE Float3 operator*(float a) const { return Float3(x * a, y * a, z * a); }
    INL_HOST_DEVICE Float3 operator/(float a) const { return Float3(x / a, y / a, z / a); }

    INL_HOST_DEVICE Float3 &operator+=(const Float3 &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    INL_HOST_DEVICE Float3 &operator-=(const Float3 &v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    INL_HOST_DEVICE Float3 &operator*=(const Float3 &v)
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }
    INL_HOST_DEVICE Float3 &operator/=(const Float3 &v)
    {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        return *this;
    }

    INL_HOST_DEVICE Float3 &operator+=(const float &a)
    {
        x += a;
        y += a;
        z += a;
        return *this;
    }
    INL_HOST_DEVICE Float3 &operator-=(const float &a)
    {
        x -= a;
        y -= a;
        z -= a;
        return *this;
    }
    INL_HOST_DEVICE Float3 &operator*=(const float &a)
    {
        x *= a;
        y *= a;
        z *= a;
        return *this;
    }
    INL_HOST_DEVICE Float3 &operator/=(const float &a)
    {
        x /= a;
        y /= a;
        z /= a;
        return *this;
    }

    INL_HOST_DEVICE Float3 operator-() const { return Float3(-x, -y, -z); }

    INL_HOST_DEVICE bool operator!=(const Float3 &v) const { return x != v.x || y != v.y || z != v.z; }
    INL_HOST_DEVICE bool operator==(const Float3 &v) const { return x == v.x && y == v.y && z == v.z; }

    INL_HOST_DEVICE bool operator<(const Float3 &v) const { return x < v.x; }
    INL_HOST_DEVICE bool operator<=(const Float3 &v) const { return x <= v.x; }
    INL_HOST_DEVICE bool operator>(const Float3 &v) const { return x > v.x; }
    INL_HOST_DEVICE bool operator>=(const Float3 &v) const { return x >= v.x; }

    INL_HOST_DEVICE float &operator[](int i) { return _v[i]; }
    INL_HOST_DEVICE float operator[](int i) const { return _v[i]; }

    INL_HOST_DEVICE explicit operator float3() const { return make_float3(x, y, z); }

    INL_HOST_DEVICE float length2() const { return (float)InnerProduct(x, x, y, y, z, z); }
    INL_HOST_DEVICE float length() const { return sqrtf(length2()); }
    INL_HOST_DEVICE float getmax() const { return max(max(x, y), z); }
    INL_HOST_DEVICE float getmin() const { return min(min(x, y), z); }
    INL_HOST_DEVICE float norm() const { return length(); }
    INL_HOST_DEVICE Float3 &normalize()
    {
        float n = norm();
        x /= n;
        y /= n;
        z /= n;
        return *this;
    }
    INL_HOST_DEVICE Float3 normalized() const
    {
        float n = norm();
        return Float3(x / n, y / n, z / n);
    }
};

struct Float3Hasher
{
    unsigned long long operator()(const Float3 &v) const
    {
        union FloatToSizeT
        {
            FloatToSizeT(float a) : a{a} {}
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

INL_HOST_DEVICE Float3 operator+(float a, const Float3 &v) { return Float3(v.x + a, v.y + a, v.z + a); }
INL_HOST_DEVICE Float3 operator-(float a, const Float3 &v) { return Float3(a - v.x, a - v.y, a - v.z); }
INL_HOST_DEVICE Float3 operator*(float a, const Float3 &v) { return Float3(v.x * a, v.y * a, v.z * a); }
INL_HOST_DEVICE Float3 operator/(float a, const Float3 &v) { return Float3(a / v.x, a / v.y, a / v.z); }

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
        struct
        {
            int x, y, z;
        };
        int _v[3];
    };

    INL_HOST_DEVICE Int3() : x{0}, y{0}, z{0} {}
    INL_HOST_DEVICE explicit Int3(int a) : x{a}, y{a}, z{a} {}
    INL_HOST_DEVICE Int3(int x, int y, int z) : x{x}, y{y}, z{z} {}

    INL_HOST_DEVICE Int3 operator+(int a) const { return Int3(x + a, y + a, z + a); }
    INL_HOST_DEVICE Int3 operator-(int a) const { return Int3(x - a, y - a, z - a); }

    INL_HOST_DEVICE Int3 operator+=(int a)
    {
        x += a;
        y += a;
        z += a;
        return *this;
    }
    INL_HOST_DEVICE Int3 operator-=(int a)
    {
        x -= a;
        y -= a;
        z -= a;
        return *this;
    }

    INL_HOST_DEVICE Int3 operator+(const Int3 &v) const { return Int3(x + v.x, y + v.y, z + v.y); }
    INL_HOST_DEVICE Int3 operator-(const Int3 &v) const { return Int3(x - v.x, y - v.y, z - v.y); }

    INL_HOST_DEVICE Int3 operator+=(const Int3 &v)
    {
        x += v.x;
        y += v.y;
        z += v.y;
        return *this;
    }
    INL_HOST_DEVICE Int3 operator-=(const Int3 &v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.y;
        return *this;
    }

    INL_HOST_DEVICE Int3 operator-() const { return Int3(-x, -y, -z); }

    INL_HOST_DEVICE bool operator==(const Int3 &v) { return x == v.x && y == v.y && z == v.z; }
    INL_HOST_DEVICE bool operator!=(const Int3 &v) { return x != v.x || y != v.y || z != v.z; }

    INL_HOST_DEVICE int &operator[](int i) { return _v[i]; }
    INL_HOST_DEVICE int operator[](int i) const { return _v[i]; }
};

struct UInt3
{
    union
    {
        struct
        {
            unsigned int x, y, z;
        };
        unsigned int _v[3];
    };

    INL_HOST_DEVICE UInt3() : x{0}, y{0}, z{0} {}
    INL_HOST_DEVICE explicit UInt3(unsigned int a) : x{a}, y{a}, z{a} {}
    INL_HOST_DEVICE UInt3(unsigned int x, unsigned int y, unsigned int z) : x{x}, y{y}, z{z} {}
    INL_HOST_DEVICE UInt3(const uint3 &v) : x{v.x}, y{v.y}, z{v.z} {}

    INL_HOST_DEVICE UInt3 operator+(unsigned int a) const { return UInt3(x + a, y + a, z + a); }
    INL_HOST_DEVICE UInt3 operator-(unsigned int a) const { return UInt3(x - a, y - a, z - a); }

    INL_HOST_DEVICE UInt3 operator+=(unsigned int a)
    {
        x += a;
        y += a;
        z += a;
        return *this;
    }
    INL_HOST_DEVICE UInt3 operator-=(unsigned int a)
    {
        x -= a;
        y -= a;
        z -= a;
        return *this;
    }

    INL_HOST_DEVICE UInt3 operator+(const UInt3 &v) const { return UInt3(x + v.x, y + v.y, z + v.z); }
    INL_HOST_DEVICE UInt3 operator-(const UInt3 &v) const { return UInt3(x - v.x, y - v.y, z - v.z); }

    INL_HOST_DEVICE UInt3 operator+=(const UInt3 &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    INL_HOST_DEVICE UInt3 operator-=(const UInt3 &v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    INL_HOST_DEVICE bool operator==(const UInt3 &v) { return x == v.x && y == v.y && z == v.z; }
    INL_HOST_DEVICE bool operator!=(const UInt3 &v) { return x != v.x || y != v.y || z != v.z; }

    INL_HOST_DEVICE unsigned int &operator[](unsigned int i) { return _v[i]; }
    INL_HOST_DEVICE unsigned int operator[](unsigned int i) const { return _v[i]; }
};

struct Int4
{
    union
    {
        struct
        {
            int x, y, z, w;
        };
        int _v[4];
    };

    INL_HOST_DEVICE Int4() : x{0}, y{0}, z{0}, w{0} {}
    INL_HOST_DEVICE explicit Int4(int a) : x{a}, y{a}, z{a}, w{a} {}
    INL_HOST_DEVICE Int4(int x, int y, int z, int w) : x{x}, y{y}, z{z}, w{w} {}

    INL_HOST_DEVICE Int4 operator+(int a) const { return Int4(x + a, y + a, z + a, w + a); }
    INL_HOST_DEVICE Int4 operator-(int a) const { return Int4(x - a, y - a, z - a, w + a); }

    INL_HOST_DEVICE Int4 operator+=(int a)
    {
        x += a;
        y += a;
        z += a;
        w += a;
        return *this;
    }
    INL_HOST_DEVICE Int4 operator-=(int a)
    {
        x -= a;
        y -= a;
        z -= a;
        w += a;
        return *this;
    }

    INL_HOST_DEVICE Int4 operator+(const Int4 &v) const { return Int4(x + v.x, y + v.y, z + v.z, w + v.w); }
    INL_HOST_DEVICE Int4 operator-(const Int4 &v) const { return Int4(x - v.x, y - v.y, z - v.z, w - v.w); }

    INL_HOST_DEVICE Int4 operator+=(const Int4 &v)
    {
        x += v.x;
        y += v.y;
        z += v.y;
        w += v.w;
        return *this;
    }
    INL_HOST_DEVICE Int4 operator-=(const Int4 &v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.y;
        w -= v.w;
        return *this;
    }

    INL_HOST_DEVICE Int4 operator-() const { return Int4(-x, -y, -z, -w); }

    INL_HOST_DEVICE bool operator==(const Int4 &v) { return x == v.x && y == v.y && z == v.z && w == v.w; }
    INL_HOST_DEVICE bool operator!=(const Int4 &v) { return x != v.x || y != v.y || z != v.z || w != v.w; }

    INL_HOST_DEVICE int &operator[](int i) { return _v[i]; }
    INL_HOST_DEVICE int operator[](int i) const { return _v[i]; }
};

struct UInt4
{
    union
    {
        struct
        {
            unsigned int x, y, z, w;
        };
        unsigned int _v[4];
    };

    INL_HOST_DEVICE UInt4() : x{0}, y{0}, z{0}, w{0} {}
    INL_HOST_DEVICE explicit UInt4(unsigned int a) : x{a}, y{a}, z{a}, w{a} {}
    INL_HOST_DEVICE UInt4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) : x{x}, y{y}, z{z}, w{w} {}
    INL_HOST_DEVICE UInt4(const uint4 &v) : x{v.x}, y{v.y}, z{v.z}, w{v.w} {}
    INL_HOST_DEVICE UInt4(const UInt2 &v1, const UInt2 &v2) : x{v1.x}, y{v1.y}, z{v2.x}, w{v2.y} {}

    INL_HOST_DEVICE UInt4 operator+(unsigned int a) const { return UInt4(x + a, y + a, z + a, w + a); }
    INL_HOST_DEVICE UInt4 operator-(unsigned int a) const { return UInt4(x - a, y - a, z - a, w - a); }

    INL_HOST_DEVICE UInt4 operator+=(unsigned int a)
    {
        x += a;
        y += a;
        z += a;
        w += a;
        return *this;
    }
    INL_HOST_DEVICE UInt4 operator-=(unsigned int a)
    {
        x -= a;
        y -= a;
        z -= a;
        w -= a;
        return *this;
    }

    INL_HOST_DEVICE UInt4 operator+(const UInt4 &v) const { return UInt4(x + v.x, y + v.y, z + v.z, w + v.w); }
    INL_HOST_DEVICE UInt4 operator-(const UInt4 &v) const { return UInt4(x - v.x, y - v.y, z - v.z, w - v.w); }

    INL_HOST_DEVICE UInt4 operator+=(const UInt4 &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
        return *this;
    }
    INL_HOST_DEVICE UInt4 operator-=(const UInt4 &v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        w -= v.w;
        return *this;
    }

    INL_HOST_DEVICE bool operator==(const UInt4 &v) { return x == v.x && y == v.y && z == v.z && w == v.w; }
    INL_HOST_DEVICE bool operator!=(const UInt4 &v) { return x != v.x || y != v.y || z != v.z || w != v.w; }

    INL_HOST_DEVICE unsigned int &operator[](unsigned int i) { return _v[i]; }
    INL_HOST_DEVICE unsigned int operator[](unsigned int i) const { return _v[i]; }
};

struct Float4
{
    union
    {
        struct
        {
            float x, y, z, w;
        };
        struct
        {
            Float3 xyz;
            float w;
        };
        struct
        {
            Float2 xy;
            Float2 zw;
        };
        float _v[4];
    };

    INL_HOST_DEVICE Float4() : x(0), y(0), z(0), w(0) {}
    INL_HOST_DEVICE explicit Float4(float _x) : x(_x), y(_x), z(_x), w(_x) {}
    INL_HOST_DEVICE Float4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}
    INL_HOST_DEVICE explicit Float4(const Float2 &v1, const Float2 &v2) : x(v1.x), y(v1.y), z(v2.x), w(v2.y) {}
    INL_HOST_DEVICE explicit Float4(const Float3 &v) : x(v.x), y(v.y), z(v.z), w(0) {}
    INL_HOST_DEVICE explicit Float4(const Float3 &v, float a) : x(v.x), y(v.y), z(v.z), w(a) {}
    INL_HOST_DEVICE explicit Float4(const float4 &v) : x(v.x), y(v.y), z(v.z), w(v.w) {}

    INL_HOST_DEVICE Float2 xz() { return Float2(x, z); }
    INL_HOST_DEVICE Float2 yw() { return Float2(y, w); }

    INL_HOST_DEVICE Float4 operator+(const Float4 &v) const { return Float4(x + v.x, y + v.y, z + v.z, z + v.z); }
    INL_HOST_DEVICE Float4 operator-(const Float4 &v) const { return Float4(x - v.x, y - v.y, z - v.z, z - v.z); }
    INL_HOST_DEVICE Float4 operator*(const Float4 &v) const { return Float4(x * v.x, y * v.y, z * v.z, z * v.z); }
    INL_HOST_DEVICE Float4 operator/(const Float4 &v) const { return Float4(x / v.x, y / v.y, z / v.z, z / v.z); }

    INL_HOST_DEVICE Float4 operator+(float a) const { return Float4(x + a, y + a, z + a, z + a); }
    INL_HOST_DEVICE Float4 operator-(float a) const { return Float4(x - a, y - a, z - a, z - a); }
    INL_HOST_DEVICE Float4 operator*(float a) const { return Float4(x * a, y * a, z * a, z * a); }
    INL_HOST_DEVICE Float4 operator/(float a) const { return Float4(x / a, y / a, z / a, z / a); }

    INL_HOST_DEVICE Float4 &operator+=(const Float4 &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
        return *this;
    }
    INL_HOST_DEVICE Float4 &operator-=(const Float4 &v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        w -= v.w;
        return *this;
    }
    INL_HOST_DEVICE Float4 &operator*=(const Float4 &v)
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        w *= v.w;
        return *this;
    }
    INL_HOST_DEVICE Float4 &operator/=(const Float4 &v)
    {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        w /= v.w;
        return *this;
    }

    INL_HOST_DEVICE Float4 &operator+=(const float &a)
    {
        x += a;
        y += a;
        z += a;
        w += a;
        return *this;
    }
    INL_HOST_DEVICE Float4 &operator-=(const float &a)
    {
        x -= a;
        y -= a;
        z -= a;
        w += a;
        return *this;
    }
    INL_HOST_DEVICE Float4 &operator*=(const float &a)
    {
        x *= a;
        y *= a;
        z *= a;
        w += a;
        return *this;
    }
    INL_HOST_DEVICE Float4 &operator/=(const float &a)
    {
        x /= a;
        y /= a;
        z /= a;
        w += a;
        return *this;
    }

    INL_HOST_DEVICE Float4 operator-() const { return Float4(-x, -y, -z, -w); }

    INL_HOST_DEVICE bool operator!=(const Float4 &v) const { return x != v.x || y != v.y || z != v.z || w != v.w; }
    INL_HOST_DEVICE bool operator==(const Float4 &v) const { return x == v.x && y == v.y && z == v.z && w == v.w; }

    INL_HOST_DEVICE float &operator[](int i) { return _v[i]; }
    INL_HOST_DEVICE float operator[](int i) const { return _v[i]; }
};

INL_HOST_DEVICE Float4 operator+(float a, const Float4 &v) { return Float4(v.x + a, v.y + a, v.z + a, v.w + a); }
INL_HOST_DEVICE Float4 operator-(float a, const Float4 &v) { return Float4(a - v.x, a - v.y, a - v.z, a - v.w); }
INL_HOST_DEVICE Float4 operator*(float a, const Float4 &v) { return Float4(v.x * a, v.y * a, v.z * a, v.w * a); }
INL_HOST_DEVICE Float4 operator/(float a, const Float4 &v) { return Float4(a / v.x, a / v.y, a / v.z, a / v.w); }

INL_HOST_DEVICE Float3 abs(const Float3 &v) { return Float3(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }
INL_HOST_DEVICE Float2 normalize(const Float2 &v)
{
    float norm = sqrtf(v.x * v.x + v.y * v.y);
    return Float2(v.x / norm, v.y / norm);
}
INL_HOST_DEVICE Float3 normalize(const Float3 &v)
{
    float norm = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return Float3(v.x / norm, v.y / norm, v.z / norm);
}
INL_HOST_DEVICE Float3 sqrt3f(const Float3 &v) { return Float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z)); }
INL_HOST_DEVICE Float3 rsqrt3f(const Float3 &v) { return Float3(1.0f / sqrtf(v.x), 1.0f / sqrtf(v.y), 1.0f / sqrtf(v.z)); }
INL_HOST_DEVICE Float3 min3f(const Float3 &v1, const Float3 &v2) { return Float3(min(v1.x, v2.x), min(v1.y, v2.y), min(v1.z, v2.z)); }
INL_HOST_DEVICE Float3 max3f(const Float3 &v1, const Float3 &v2) { return Float3(max(v1.x, v2.x), max(v1.y, v2.y), max(v1.z, v2.z)); }
INL_HOST_DEVICE Float4 min4f(const Float4 &v1, const Float4 &v2) { return Float4(min(v1.x, v2.x), min(v1.y, v2.y), min(v1.z, v2.z), min(v1.w, v2.w)); }
INL_HOST_DEVICE Float4 max4f(const Float4 &v1, const Float4 &v2) { return Float4(max(v1.x, v2.x), max(v1.y, v2.y), max(v1.z, v2.z), max(v1.w, v2.w)); }
INL_HOST_DEVICE Float3 cross(const Float3 &v1, const Float3 &v2) { return Float3(dop(v1.y, v2.z, v1.z, v2.y), dop(v1.z, v2.x, v1.x, v2.z), dop(v1.x, v2.y, v1.y, v2.x)); /*Float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);*/ }
INL_HOST_DEVICE Float3 pow3f(const Float3 &v1, const Float3 &v2) { return Float3(powf(v1.x, v2.x), powf(v1.y, v2.y), powf(v1.z, v2.z)); }
INL_HOST_DEVICE Float3 exp3f(const Float3 &v) { return Float3(expf(v.x), expf(v.y), expf(v.z)); }
INL_HOST_DEVICE Float3 pow3f(const Float3 &v, float a) { return Float3(powf(v.x, a), powf(v.y, a), powf(v.z, a)); }
INL_HOST_DEVICE Float4 pow4f(const Float4 &v, float a) { return Float4(powf(v.x, a), powf(v.y, a), powf(v.z, a), powf(v.w, a)); }
INL_HOST_DEVICE Float3 sin3f(const Float3 &v) { return Float3(sinf(v.x), sinf(v.y), sinf(v.z)); }
INL_HOST_DEVICE Float3 cos3f(const Float3 &v) { return Float3(cosf(v.x), cosf(v.y), cosf(v.z)); }
INL_HOST_DEVICE Float3 mixf(const Float3 &v1, const Float3 &v2, float a) { return v1 * (1.0f - a) + v2 * a; }
INL_HOST_DEVICE float mix1f(float v1, float v2, float a) { return v1 * (1.0f - a) + v2 * a; }
INL_HOST_DEVICE Float3 minf3f(const float a, const Float3 &v) { return Float3(v.x < a ? v.x : a, v.y < a ? v.y : a, v.y < a ? v.y : a); }
INL_HOST_DEVICE void swap(int &v1, int &v2)
{
    int tmp = v1;
    v1 = v2;
    v2 = tmp;
}
INL_HOST_DEVICE void swap(float &v1, float &v2)
{
    float tmp = v1;
    v1 = v2;
    v2 = tmp;
}
INL_HOST_DEVICE void swap(Float3 &v1, Float3 &v2)
{
    Float3 tmp = v1;
    v1 = v2;
    v2 = tmp;
}
INL_HOST_DEVICE int clampi(int a, int lo = 0, int hi = 1) { return a < lo ? lo : (a > hi ? hi : a); }
INL_HOST_DEVICE unsigned int clampu(unsigned int a, unsigned int lo = 0u, unsigned int hi = 1u) { return a < lo ? lo : (a > hi ? hi : a); }
INL_HOST_DEVICE Int2 clamp2i(Int2 a, Int2 lo, Int2 hi) { return Int2(clampi(a.x, lo.x, hi.x), clampi(a.y, lo.y, hi.y)); }
INL_HOST_DEVICE float clampf(float a, float lo = 0.0f, float hi = 1.0f) { return a < lo ? lo : a > hi ? hi
                                                                                                      : a; }
INL_HOST_DEVICE Float2 clamp2f(Float2 a, Float2 lo = Float2(0.0f), Float2 hi = Float2(1.0f)) { return Float2(clampf(a.x, lo.x, hi.x), clampf(a.y, lo.y, hi.y)); }
INL_HOST_DEVICE Float3 clamp3f(Float3 a, Float3 lo = Float3(0.0f), Float3 hi = Float3(1.0f)) { return Float3(clampf(a.x, lo.x, hi.x), clampf(a.y, lo.y, hi.y), clampf(a.z, lo.z, hi.z)); }
INL_HOST_DEVICE Float4 clamp4f(Float4 a, Float4 lo = Float4(0.0f), Float4 hi = Float4(1.0f)) { return Float4(clampf(a.x, lo.x, hi.x), clampf(a.y, lo.y, hi.y), clampf(a.z, lo.z, hi.z), clampf(a.w, lo.w, hi.w)); }
INL_HOST_DEVICE float dot(const Float2 &v1, const Float2 &v2) { return v1.x * v2.x + v1.y * v2.y; }
INL_HOST_DEVICE float dot(const Float3 &v1, const Float3 &v2) { return (float)InnerProduct(v1.x, v2.x, v1.y, v2.y, v1.z, v2.z); /*v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;*/ }
INL_HOST_DEVICE float dot(const Float4 &v1, const Float4 &v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w; }
INL_HOST_DEVICE float distancesq(const Float3 &v1, const Float3 &v2) { return (v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z) * (v1.z - v2.z); }
INL_HOST_DEVICE float distance(const Float3 &v1, const Float3 &v2) { return sqrtf((v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z) * (v1.z - v2.z)); }
INL_HOST_DEVICE Float3 lerp3f(Float3 a, Float3 b, float w) { return a + w * (b - a); }
INL_HOST_DEVICE Float4 lerp4f(Float4 a, Float4 b, float w) { return a + w * (b - a); }
INL_HOST_DEVICE float lerpf(float a, float b, float w) { return a + w * (b - a); }
INL_HOST_DEVICE float lerp(float a, float b, float w) { return a + w * (b - a); }
INL_HOST_DEVICE Float3 lerp(Float3 a, Float3 b, float w) { return a + w * (b - a); }
INL_HOST_DEVICE Float4 lerp(Float4 a, Float4 b, float w) { return a + w * (b - a); }
INL_HOST_DEVICE Float3 reflect3f(Float3 i, Float3 n) { return i - 2.0f * n * dot(n, i); }
INL_HOST_DEVICE float pow2(float a) { return a * a; }
INL_HOST_DEVICE float pow3(float a) { return a * a * a; }
INL_HOST_DEVICE float length(const Float3 &v) { return sqrtf(dot(v, v)); }

INL_HOST_DEVICE float smoothstep1f(float a, float b, float w) { return a + (w * w * (3.0f - 2.0f * w)) * (b - a); }
INL_HOST_DEVICE Float3 smoothstep3f(Float3 a, Float3 b, float w) { return a + (w * w * (3.0f - 2.0f * w)) * (b - a); }

INL_HOST_DEVICE float AngleBetween(const Float3 &a, const Float3 &b) { return acos(dot(a, b) / sqrtf(a.length2() * b.length2())); }

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
    INL_HOST_DEVICE Mat3()
    {
        for (int i = 0; i < 9; ++i)
            _v[i] = 0;
    }
    INL_HOST_DEVICE Mat3(const Mat3 &rhs) : v0{rhs.v0}, v1{rhs.v1}, v2{rhs.v2} {}
    INL_HOST_DEVICE Mat3(const Float3 &v0, const Float3 &v1, const Float3 &v2) : v0{v0}, v1{v1}, v2{v2} {}

    // column-major matrix construction
    INL_HOST_DEVICE constexpr Mat3(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22)
        : m00{m00}, m01{m01}, m02{m02}, m10{m10}, m11{m11}, m12{m12}, m20{m20}, m21{m21}, m22{m22}
    {
    }

    INL_HOST_DEVICE Float3 &operator[](int i) { return _v3[i]; }
    INL_HOST_DEVICE const Float3 &operator[](int i) const { return _v3[i]; }

    INL_HOST_DEVICE void transpose()
    {
        swap(m01, m10);
        swap(m20, m02);
        swap(m21, m12);
    }
};

INL_HOST_DEVICE Mat3 operator*(const Mat3 &A, const Mat3 &B)
{
    return Mat3(
        // First row of C
        A.m00 * B.m00 + A.m01 * B.m10 + A.m02 * B.m20, // C.m00
        A.m00 * B.m01 + A.m01 * B.m11 + A.m02 * B.m21, // C.m01
        A.m00 * B.m02 + A.m01 * B.m12 + A.m02 * B.m22, // C.m02

        // Second row of C
        A.m10 * B.m00 + A.m11 * B.m10 + A.m12 * B.m20, // C.m10
        A.m10 * B.m01 + A.m11 * B.m11 + A.m12 * B.m21, // C.m11
        A.m10 * B.m02 + A.m11 * B.m12 + A.m12 * B.m22, // C.m12

        // Third row of C
        A.m20 * B.m00 + A.m21 * B.m10 + A.m22 * B.m20, // C.m20
        A.m20 * B.m01 + A.m21 * B.m11 + A.m22 * B.m21, // C.m21
        A.m20 * B.m02 + A.m21 * B.m12 + A.m22 * B.m22  // C.m22
    );
}

// column major multiply
INL_HOST_DEVICE Float3 operator*(const Mat3 &m, const Float3 &v)
{
    return Float3((float)InnerProduct(m.m00, v[0], m.m01, v[1], m.m02, v[2]),
                  (float)InnerProduct(m.m10, v[0], m.m11, v[1], m.m12, v[2]),
                  (float)InnerProduct(m.m20, v[0], m.m21, v[1], m.m22, v[2])); // return m.v0 * v.x + m.v1 * v.y + m.v2 * v.z;
}

// Rotation matrix
INL_HOST_DEVICE Mat3 RotationMatrixX(float a) { return {Float3(1, 0, 0), Float3(0, cosf(a), sinf(a)), Float3(0, -sinf(a), cosf(a))}; }
INL_HOST_DEVICE Mat3 RotationMatrixY(float a) { return {Float3(cosf(a), 0, -sinf(a)), Float3(0, 1, 0), Float3(sinf(a), 0, cosf(a))}; }
INL_HOST_DEVICE Mat3 RotationMatrixZ(float a) { return {Float3(cosf(a), sinf(a), 0), Float3(-sinf(a), cosf(a), 0), Float3(0, 0, 1)}; }

INL_HOST_DEVICE float Determinant(const Mat3 &m)
{
    float minor12 = DifferenceOfProducts(m[1][1], m[2][2], m[1][2], m[2][1]);
    float minor02 = DifferenceOfProducts(m[1][0], m[2][2], m[1][2], m[2][0]);
    float minor01 = DifferenceOfProducts(m[1][0], m[2][1], m[1][1], m[2][0]);
    return FMA(m[0][2], minor01, DifferenceOfProducts(m[0][0], minor12, m[0][1], minor02));
}

INL_HOST_DEVICE Mat3 Inverse(const Mat3 &m)
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

    INL_HOST_DEVICE Mat4()
    {
        for (int i = 0; i < 16; ++i)
        {
            _v[i] = 0;
        }
        m00 = m11 = m22 = m33 = 1;
    }
    INL_HOST_DEVICE Mat4(const Mat4 &m)
    {
        for (int i = 0; i < 16; ++i)
        {
            _v[i] = m._v[i];
        }
    }
    
    // TRS constructor - create transformation matrix from Translation, Rotation (quaternion), Scale
    INL_HOST_DEVICE Mat4(const Float3& translation, const Float4& rotation, const Float3& scale)
    {
        // Extract quaternion components
        float x = rotation.x, y = rotation.y, z = rotation.z, w = rotation.w;

        // Calculate rotation matrix elements
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, xz = x * z, yz = y * z;
        float wx = w * x, wy = w * y, wz = w * z;

        // Build matrix with rotation and scale (column-major)
        _v[0] = scale.x * (1.0f - 2.0f * (yy + zz));    // [0][0]
        _v[1] = scale.x * (2.0f * (xy + wz));           // [0][1]
        _v[2] = scale.x * (2.0f * (xz - wy));           // [0][2]
        _v[3] = 0.0f;                                   // [0][3]

        _v[4] = scale.y * (2.0f * (xy - wz));           // [1][0]
        _v[5] = scale.y * (1.0f - 2.0f * (xx + zz));    // [1][1]
        _v[6] = scale.y * (2.0f * (yz + wx));           // [1][2]
        _v[7] = 0.0f;                                   // [1][3]

        _v[8] = scale.z * (2.0f * (xz + wy));           // [2][0]
        _v[9] = scale.z * (2.0f * (yz - wx));           // [2][1]
        _v[10] = scale.z * (1.0f - 2.0f * (xx + yy));   // [2][2]
        _v[11] = 0.0f;                                  // [2][3]

        // Translation
        _v[12] = translation.x;                         // [3][0]
        _v[13] = translation.y;                         // [3][1]
        _v[14] = translation.z;                         // [3][2]
        _v[15] = 1.0f;                                  // [3][3]
    }

    // row
    INL_HOST_DEVICE void setRow(unsigned int i, const Float4 &v)
    { /*assert(i < 4);*/
        _v[i] = v[0];
        _v[i + 4] = v[1];
        _v[i + 8] = v[2];
        _v[i + 12] = v[3];
    }
    INL_HOST_DEVICE Float4 getRow(unsigned int i) const { /*assert(i < 4);*/ return Float4(_v[i], _v[i + 4], _v[i + 8], _v[i + 12]); }

    // column
    INL_HOST_DEVICE void setCol(unsigned int i, const Float4 &v)
    { /*assert(i < 4);*/
        _v[i * 4] = v[0];
        _v[i * 4 + 1] = v[1];
        _v[i * 4 + 2] = v[2];
        _v[i * 4 + 3] = v[3];
    }
    INL_HOST_DEVICE Float4 getCol(unsigned int i) const { /*assert(i < 4);*/ return Float4(_v[i * 4], _v[i * 4 + 1], _v[i * 4 + 2], _v[i * 4 + 3]); }

    // element
    INL_HOST_DEVICE void set(unsigned int r, unsigned int c, float v) { /*assert(r < 4 && c < 4);*/ _v[r + c * 4] = v; }
    INL_HOST_DEVICE float get(unsigned int r, unsigned int c) const { /*assert(r < 4 && c < 4);*/ return _v[r + c * 4]; }

    INL_HOST_DEVICE const float *operator[](unsigned int i) const { return _v + i * 4; }
    INL_HOST_DEVICE float *operator[](unsigned int i) { return _v + i * 4; }
    
    // Matrix multiplication operator (column-major)
    INL_HOST_DEVICE Mat4 operator*(const Mat4& other) const {
        Mat4 result;
        for (int col = 0; col < 4; col++) {
            for (int row = 0; row < 4; row++) {
                result[col][row] = 0.0f;
                for (int k = 0; k < 4; k++) {
                    result[col][row] += (*this)[k][row] * other[col][k];
                }
            }
        }
        return result;
    }
    
    
    // Transform a 3D point by this matrix (homogeneous coordinates) - operator overload
    INL_HOST_DEVICE Float3 operator*(const Float3& point) const {
        Float4 result;
        // Column-major matrix transformation: result = matrix * point
        result.x = (*this)[0][0] * point.x + (*this)[1][0] * point.y + (*this)[2][0] * point.z + (*this)[3][0];
        result.y = (*this)[0][1] * point.x + (*this)[1][1] * point.y + (*this)[2][1] * point.z + (*this)[3][1];
        result.z = (*this)[0][2] * point.x + (*this)[1][2] * point.y + (*this)[2][2] * point.z + (*this)[3][2];
        result.w = (*this)[0][3] * point.x + (*this)[1][3] * point.y + (*this)[2][3] * point.z + (*this)[3][3];
        return Float3(result.x, result.y, result.z) / result.w;
    }
};

INL_HOST_DEVICE Mat4 invert(const Mat4 &m)
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
    INL_HOST_DEVICE Quat(const Float3 &v) : v(v), w(0) {}
    INL_HOST_DEVICE Quat(const Float3 &v, float w) : v(v), w(w) {}
    INL_HOST_DEVICE Quat(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

    static INL_HOST_DEVICE Quat axisAngle(const Float3 &axis, float angle) { return Quat(axis.normalized() * sinf(angle / 2), cosf(angle / 2)); }

    INL_HOST_DEVICE Quat conj() const { return Quat(-v, w); }
    INL_HOST_DEVICE float norm2() const { return x * x + y * y + z * z + w * w; }
    INL_HOST_DEVICE Quat inv() const { return conj() / norm2(); }
    INL_HOST_DEVICE float norm() const { return sqrtf(norm2()); }
    INL_HOST_DEVICE Quat normalized() const
    {
        float n = norm();
        return Quat(v / n, w / n);
    }
    INL_HOST_DEVICE Quat pow(float a) const { return Quat::axisAngle(v, acosf(w) * a * 2); }

    INL_HOST_DEVICE Quat operator/(float a) const { return Quat(v / a, w / a); }
    INL_HOST_DEVICE Quat operator+(const Quat &q) const
    {
        const Quat &p = *this;
        return Quat(p.v + q.v, p.w + q.w);
    }
    INL_HOST_DEVICE Quat operator*(const Quat &q) const
    {
        const Quat &p = *this;
        return Quat(p.w * q.v + q.w * p.v + cross(p.v, q.v), p.w * q.w - dot(p.v, q.v));
    }
    INL_HOST_DEVICE Quat &operator+=(const Quat &q)
    {
        Quat ret = *this + q;
        return (*this = ret);
    }
    INL_HOST_DEVICE Quat &operator*=(const Quat &q)
    {
        Quat ret = *this * q;
        return (*this = ret);
    }
};

INL_HOST_DEVICE Quat rotate(const Quat &q, const Quat &v) { return q * v * q.conj(); }
INL_HOST_DEVICE Quat slerp(const Quat &q, const Quat &r, float t) { return (r * q.conj()).pow(t) * q; }
INL_HOST_DEVICE Quat rotationBetween(const Quat &p, const Quat &q) { return Quat(cross(p.v, q.v), sqrtf(p.v.length2() * q.v.length2()) + dot(p.v, q.v)).normalized(); }

INL_HOST_DEVICE Float3 rotate3f(const Float3 &axis, float angle, const Float3 &v) { return rotate(Quat::axisAngle(axis, angle), v).v; }
INL_HOST_DEVICE Float3 slerp3f(const Float3 &q, const Float3 &r, float t) { return slerp(Quat(q), Quat(r), t).v; }
INL_HOST_DEVICE Float3 rotationBetween3f(const Float3 &p, const Float3 &q) { return rotationBetween(Quat(p), Quat(q)).v; }

INL_HOST_DEVICE float SafeDivide(float a, float b)
{
    float eps = 1e-20f;
    return a / ((fabsf(b) > eps) ? b : copysignf(eps, b));
};
INL_HOST_DEVICE Float3 SafeDivide3f1f(const Float3 &a, float b) { return Float3(SafeDivide(a.x, b), SafeDivide(a.y, b), SafeDivide(a.z, b)); };
INL_HOST_DEVICE Float3 SafeDivide3f(const Float3 &a, const Float3 &b) { return Float3(SafeDivide(a.x, b.x), SafeDivide(a.y, b.y), SafeDivide(a.z, b.z)); };

INL_DEVICE void LocalizeSample(
    const Float3 &n,
    Float3 &u,
    Float3 &v)
{
    Float3 w = Float3(1, 0, 0);

    if (abs(n.x) > 0.707f)
        w = Float3(0, 1, 0);

    u = cross(n, w);
    v = cross(n, u);
}

INL_DEVICE Float3 LocalizeAlignZUp(
    const Float3 &normal,
    const Float3 &localVec)
{
    Float3 u, v;

    LocalizeSample(normal, u, v);

    Float3 worldVec = localVec.x * u + localVec.y * v + localVec.z * normal;
    return worldVec;
}

/**
 *  Calculates refraction direction
 *  r   : refraction vector
 *  i   : incident vector
 *  n   : surface normal
 *  ior : index of refraction ( n2 / n1 )
 *  returns false in case of total internal reflection, in that case r is initialized to (0,0,0).
 */
INL_HOST_DEVICE bool refract(Float3 &r, Float3 const &i, Float3 const &n, const float ior)
{
    Float3 nn = n;
    float negNdotV = dot(i, nn);
    float eta;

    if (negNdotV > 0.0f)
    {
        eta = ior;
        nn = -n;
        negNdotV = -negNdotV;
    }
    else
    {
        eta = 1.f / ior;
    }

    const float k = 1.f - eta * eta * (1.f - negNdotV * negNdotV);

    if (k < 0.0f)
    {
        // Initialize this value, so that r always leaves this function initialized.
        r = Float3(0.f);
        return false;
    }
    else
    {
        r = normalize(eta * i - (eta * negNdotV + sqrtf(k)) * nn);
        return true;
    }
}

// Tangent-Bitangent-Normal orthonormal space.
struct TBN
{
    // Default constructor to be able to include it into other structures when needed.
    INL_HOST_DEVICE TBN()
    {
    }

    INL_HOST_DEVICE TBN(const Float3 &n)
        : normal(n)
    {
        if (fabsf(normal.z) < fabsf(normal.x))
        {
            tangent.x = normal.z;
            tangent.y = 0.0f;
            tangent.z = -normal.x;
        }
        else
        {
            tangent.x = 0.0f;
            tangent.y = normal.z;
            tangent.z = -normal.y;
        }
        tangent = normalize(tangent);
        bitangent = cross(normal, tangent);
    }

    // Constructor for cases where tangent, bitangent, and normal are given as ortho-normal basis.
    INL_HOST_DEVICE TBN(const Float3 &t, const Float3 &b, const Float3 &n)
        : tangent(t), bitangent(b), normal(n)
    {
    }

    // Normal is kept, tangent and bitangent are calculated.
    // Normal must be normalized.
    // Must not be used with degenerated vectors!
    INL_HOST_DEVICE TBN(const Float3 &tangent_reference, const Float3 &n)
        : normal(n)
    {
        bitangent = normalize(cross(normal, tangent_reference));
        tangent = cross(bitangent, normal);
    }

    INL_HOST_DEVICE void negate()
    {
        tangent = -tangent;
        bitangent = -bitangent;
        normal = -normal;
    }

    INL_HOST_DEVICE Float3 transformToLocal(const Float3 &p) const
    {
        return Float3(dot(p, tangent),
                      dot(p, bitangent),
                      dot(p, normal));
    }

    INL_HOST_DEVICE Float3 transformToWorld(const Float3 &p) const
    {
        return p.x * tangent + p.y * bitangent + p.z * normal;
    }

    Float3 tangent;
    Float3 bitangent;
    Float3 normal;
};

INL_HOST_DEVICE float luminance(const Float3 &rgb)
{
    const Float3 ntsc_luminance = {0.2126f, 0.7152f, 0.0722f}; // {0.30f, 0.59f, 0.11f};
    return dot(rgb, ntsc_luminance);
}

INL_HOST_DEVICE float intensity(const Float3 &rgb)
{
    return (rgb.x + rgb.y + rgb.z) * 0.3333333333f;
}

INL_HOST_DEVICE float cube(const float x)
{
    return x * x * x;
}

INL_HOST_DEVICE bool isNull(const Float3 &v)
{
    return (v.x == 0.0f && v.y == 0.0f && v.z == 0.0f);
}

INL_HOST_DEVICE bool isNotNull(const Float3 &v)
{
    return (v.x != 0.0f || v.y != 0.0f || v.z != 0.0f);
}

// Used for Multiple Importance Sampling.
INL_HOST_DEVICE float powerHeuristic(const float a, const float b)
{
    const float t = a * a;
    return t / (t + b * b);
}

INL_HOST_DEVICE float balanceHeuristic(const float a, const float b)
{
    return a / (a + b);
}

// Tiny Encryption Algorithm (TEA) to calculate a the seed per launch index and iteration.
// This results in a ton of integer instructions! Use the smallest N necessary.
template <unsigned int N>
INL_DEVICE unsigned int tea(const unsigned int val0, const unsigned int val1)
{
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for (unsigned int n = 0; n < N; ++n)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xA341316C) ^ (v1 + s0) ^ ((v1 >> 5) + 0xC8013EA4);
        v1 += ((v0 << 4) + 0xAD90777D) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7E95761E);
    }
    return v0;
}

INL_DEVICE float rng(unsigned int &previous)
{
    previous = previous * 1664525u + 1013904223u;

    return float(previous & 0x00FFFFFF) / float(0x01000000u);
}

INL_DEVICE Float2 rng2(unsigned int &previous)
{
    Float2 s;

    previous = previous * 1664525u + 1013904223u;
    s.x = float(previous & 0x00FFFFFF) / float(0x01000000u);

    previous = previous * 1664525u + 1013904223u;
    s.y = float(previous & 0x00FFFFFF) / float(0x01000000u);

    return s;
}

INL_DEVICE Float2 ConcentricSampleDisk(Float2 u)
{
    // Map uniform random numbers to [-1, 1]
    Float2 uOffset = 2.0 * u - 1.0;

    // Handle degeneracy at the origin
    if (abs(uOffset.x) < 1e-10f && abs(uOffset.y) < 1e-10f)
    {
        return Float2(0, 0);
    }

    // Apply concentric mapping to point
    float theta;
    float r;

    if (abs(uOffset.x) > abs(uOffset.y))
    {
        r = uOffset.x;
        theta = PI_OVER_4 * (uOffset.y / uOffset.x);
    }
    else
    {
        r = uOffset.y;
        theta = PI_OVER_2 - PI_OVER_4 * (uOffset.x / uOffset.y);
    }

    return r * Float2(cosf(theta), sinf(theta));
}

INL_HOST_DEVICE Float3 YawPitchToDir(float yaw, float pitch)
{
    return normalize(Float3(sinf(yaw) * cosf(pitch), sinf(pitch), cosf(yaw) * cosf(pitch)));
}

INL_HOST_DEVICE Float2 DirToYawPitch(Float3 dir)
{
    dir.normalize();
    return Float2(atan2f(dir.x, dir.z), asinf(dir.y));
}

// INL_DEVICE Float3 RgbToYcocg(const Float3 &rgb)
// {
//     float tmp1 = rgb.x + rgb.z;
//     float tmp2 = rgb.y * 2.0f;
//     return Float3(tmp1 + tmp2, (rgb.x - rgb.z) * 2.0f, tmp2 - tmp1);
// }

// INL_DEVICE Float3 YcocgToRgb(const Float3 &ycocg)
// {
//     float tmp = ycocg.x - ycocg.z;
//     return Float3(tmp + ycocg.y, ycocg.x + ycocg.z, tmp - ycocg.y) * 0.25f;
// }

template <typename T>
INL_DEVICE void WarpReduceSum(T &v)
{
    const int warpSize = 32;
    for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
}

INL_DEVICE Float2 copysignf2(const Float2 &a, const Float2 &b)
{
    return {copysignf(a.x, b.x), copysignf(a.y, b.y)};
}

INL_DEVICE Float3 CatmulRom(float T, Float3 D, Float3 C, Float3 B, Float3 A)
{
    return 0.5f * ((2.0f * B) + (-A + C) * T + (2.0f * A - 5.0f * B + 4.0f * C - D) * T * T + (-A + 3.0f * B - 3.0f * C + D) * T * T * T);
}

INL_DEVICE Float3 ColorRampBSpline(float T, Float4 A, Float4 B, Float4 C, Float4 D)
{
    float AB = B.w - A.w;
    float BC = C.w - B.w;
    float CD = D.w - C.w;

    float iAB = clampf((T - A.w) / AB);
    float iBC = clampf((T - B.w) / BC);
    float iCD = clampf((T - C.w) / CD);

    Float4 p = Float4(1.0f - iAB, iAB - iBC, iBC - iCD, iCD);
    Float3 cA = CatmulRom(p.x, A.xyz, A.xyz, B.xyz, C.xyz);
    Float3 cB = CatmulRom(p.y, A.xyz, B.xyz, C.xyz, D.xyz);
    Float3 cC = CatmulRom(p.z, B.xyz, C.xyz, D.xyz, D.xyz);
    Float3 cD = D.xyz;

    if (T < B.w)
        return cA;
    if (T < C.w)
        return cB;
    if (T < D.w)
        return cC;
    return cD;
}

INL_DEVICE Float3 ColorHexToFloat3(int Hex)
{
    // 0xABCDEF
    int AB = (Hex & 0x00FF0000) >> 16;
    int CD = (Hex & 0x0000FF00) >> 8;
    int EF = Hex & 0x000000FF;
    return pow3f(Float3(AB, CD, EF) / 255.0f, Float3(2.2f));
}

INL_DEVICE Float3 ColorRampVisualization(float t)
{
    Float4 A = Float4(ColorHexToFloat3(0x6a2c70), 0.05);
    Float4 B = Float4(ColorHexToFloat3(0xb83b5e), 0.22);
    Float4 C = Float4(ColorHexToFloat3(0xf08a5d), 0.5);
    Float4 D = Float4(ColorHexToFloat3(0xf9ed69), 0.9);
    return ColorRampBSpline(t, A, B, C, D);
}

INL_DEVICE void alignVector(Float3 const &axis, Float3 &w)
{
    // Align w with axis.
    const float s = copysignf(1.0f, axis.z);
    w.z *= s;
    const Float3 h = Float3(axis.x, axis.y, axis.z + s);
    const float k = dot(w, h) / (1.0f + fabsf(axis.z));
    w = k * h - w;
}

INL_DEVICE float rcp(float a) { return 1.0f / a; }

INL_DEVICE Float3 EqualRectMap(float u, float v)
{
    float theta = u * TWO_PI;
    float phi = v * PI_OVER_2;

    float x = cos(theta) * cos(phi);
    float y = sin(phi);
    float z = sin(theta) * cos(phi);

    return Float3(x, y, z);
}

INL_DEVICE Float2 EqualRectMap(Float3 dir)
{
    float phi = asin(dir.y);
    float theta = acos(dir.x / cos(phi));

    float u = theta / TWO_PI;
    float v = phi / PI_OVER_2;

    return Float2(u, v);
}

INL_DEVICE Float3 EqualAreaHemisphereMap(float u, float v)
{
    float z = v;
    float r = sqrtf(1.0f - v * v);
    float phi = TWO_PI * u;

    return Float3(r * cosf(phi), z, r * sinf(phi));
}

INL_DEVICE Float2 EqualAreaHemisphereMap(Float3 dir)
{
    float u = atan2f(-dir.z, -dir.x) / TWO_PI + 0.5f;
    float v = max(dir.y, 0.001f);
    return Float2(u, v);
}

INL_DEVICE Float3 EqualAreaSphereMap(float u, float v)
{
    float y = 2.0f * v - 1.0f;
    float r = sqrtf(1.0f - y * y);
    float phi = TWO_PI * u;
    return Float3(r * cosf(phi), y, r * sinf(phi));
}

INL_DEVICE Float2 EqualAreaSphereMap(Float3 dir)
{
    float u = atan2f(-dir.z, -dir.x) / TWO_PI + 0.5f;
    float v = (dir.y + 1.0f) * 0.5f;
    return Float2(u, v);
}

INL_DEVICE Float3 EqualAreaMapCone(const Float3 &sunDir, float u, float v, float cosThetaMax)
{
    float cosTheta = (1.0f - u) + u * cosThetaMax;
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
    float phi = v * TWO_PI;

    Float3 t, b;
    LocalizeSample(sunDir, t, b);
    Mat3 trans(t, sunDir, b);

    Float3 coords = Float3(cosf(phi) * sinTheta, cosTheta, sinf(phi) * sinTheta);

    return trans * coords;
}

INL_DEVICE bool EqualAreaMapCone(Float2 &uv, const Float3 &sunDir, const Float3 &rayDir, float cosThetaMax)
{
    Float3 t, b;
    LocalizeSample(sunDir, t, b);
    Mat3 trans(t, sunDir, b);
    trans.transpose();

    Float3 coords = trans * rayDir;
    float cosTheta = coords.y;
    if (cosTheta < cosThetaMax)
    {
        return false;
    }
    float u = (1.0f - cosTheta) / (1.0f - cosThetaMax);

    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
    if (sinTheta < 1e-5f || (coords.x / sinTheta) < -1.0f || (coords.x / sinTheta) > 1.0f)
    {
        return false;
    }

    float v = acosf(coords.x / sinTheta) * INV_TWO_PI;

    uv = Float2(u, v);

    return true;
}

INL_HOST_DEVICE void RayAabbIntersectCore(const Float3 &rayOrig, const Float3 &rayDir, const Float3 &aabbMin, const Float3 &aabbMax, float &tMin, float &tMax)
{
    Float3 inverseRayDir = SafeDivide3f(Float3(1.0f), rayDir);

    Float3 tNear = (aabbMin - rayOrig) * inverseRayDir;
    Float3 tFar = (aabbMax - rayOrig) * inverseRayDir;

    tNear = min3f(tNear, tFar);
    tFar = max3f(tNear, tFar);

    tMin = tNear.getmax();
    tMax = tFar.getmin();
}

INL_HOST_DEVICE bool RayAabbIntersect(const Float3 &rayOrig, const Float3 &rayDir, const Float3 &aabbMin, const Float3 &aabbMax, float &t, bool &inside)
{
    float tMin;
    float tMax;

    RayAabbIntersectCore(rayOrig, rayDir, aabbMin, aabbMax, tMin, tMax);

    bool hit = (tMin <= tMax) && (tMax > 0);

    inside = (tMin < 0);
    t = inside ? tMax : tMin;

    return hit;
}

INL_HOST_DEVICE float RayVoxelGridIntersect(const Float3 &rayOrig, const Float3 &rayDir, const Float3 &aabbMin, const Float3 &aabbMax, int &axis)
{
    Float3 inverseRayDir = SafeDivide3f(Float3(1.0f), rayDir);

    Float3 tNear = (aabbMin - rayOrig) * inverseRayDir;
    Float3 tFar = (aabbMax - rayOrig) * inverseRayDir;

    tFar = max3f(tNear, tFar);

    float t;

    if (tFar.x < tFar.y)
    {
        if (tFar.x < tFar.z)
        {
            axis = 0;
            t = tFar.x;
        }
        else
        {
            axis = 2;
            t = tFar.z;
        }
    }
    else
    {
        if (tFar.y < tFar.z)
        {
            axis = 1;
            t = tFar.y;
        }
        else
        {
            axis = 2;
            t = tFar.z;
        }
    }

    return t;
}

INL_HOST_DEVICE float PointToSegmentDistance(const Float3 &P, const Float3 &A, const Float3 &B, Float3 &closestPoint)
{
    Float3 v = B - A;
    Float3 w = P - A;
    float c1 = dot(w, v);

    // If projection falls before A on the line
    if (c1 <= 0.0f)
    {
        closestPoint = A;
        return length(P - A);
    }

    float c2 = dot(v, v);
    // If projection falls beyond B on the line
    if (c2 <= c1)
    {
        closestPoint = B;
        return length(P - B);
    }

    // Projection falls between A and B
    float t = c1 / c2;
    closestPoint = A + v * t;
    return length(P - closestPoint);
}

INL_HOST_DEVICE float radians(float degrees)
{
    return degrees * (M_PI / 180.0f);
}

INL_HOST_DEVICE float saturate(float x)
{
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

INL_HOST_DEVICE Float2 saturate(const Float2 &v)
{
    return Float2(saturate(v.x), saturate(v.y));
}

INL_HOST_DEVICE float square(float x) { return x * x; }
INL_HOST_DEVICE Float2 square(Float2 x) { return x * x; }
INL_HOST_DEVICE Float3 square(Float3 x) { return x * x; }
INL_HOST_DEVICE Float4 square(Float4 x) { return x * x; }

INL_DEVICE float pow5(float e)
{
    float e2 = e * e;
    return e2 * e2 * e;
}

INL_DEVICE Float3 FresnelShlick(const Float3 &F0, float cosTheta)
{
    return F0 + (Float3(1.0f) - F0) * pow5(1.0f - cosTheta);
}

INL_DEVICE float FresnelShlick(float F0, float cosTheta)
{
    return F0 + (1.0f - F0) * pow5(1.0f - cosTheta);
}

INL_DEVICE Float3 SampleTriangle(Float2 rndSample)
{
    float sqrtx = sqrt(rndSample.x);

    return Float3(
        1.0f - sqrtx,
        sqrtx * (1.0f - rndSample.y),
        sqrtx * rndSample.y);
}

// Inverse of SampleTriangle
INL_DEVICE Float2 InverseTriangleSample(Float2 hitUV)
{
    Float3 barycentric = Float3(1.0f - hitUV.x - hitUV.y, hitUV.x, hitUV.y);
    float sqrtx = 1 - barycentric.x;
    return Float2(sqrtx * sqrtx, barycentric.z / sqrtx);
}

// Converts a point in the octahedral map to a normalized direction (non-equal area, signed)
// p - signed position in octahedral map [-1, 1] for each component
// Returns normalized direction
INL_DEVICE Float3 octToNdirSigned(Float2 p)
{
    // https://twitter.com/Stubbesaurus/status/937994790553227264
    Float3 n = Float3(p.x, p.y, 1.0f - abs(p.x) - abs(p.y));
    float t = max(0.0f, -n.z);
    n.x += n.x >= 0.0f ? -t : t;
    n.y += n.y >= 0.0f ? -t : t;
    return normalize(n);
}

// Converts a point in the octahedral map (non-equal area, unsigned normalized) to normalized direction
// pNorm - a packed 32 bit unsigned normalized position in octahedral map
// Returns normalized direction
INL_DEVICE Float3 octToNdirUnorm32(unsigned int pUnorm)
{
    Float2 p;
    p.x = saturate(float(pUnorm & 0xffff) / 0xfffe);
    p.y = saturate(float(pUnorm >> 16) / 0xfffe);
    p = p * 2.0f - 1.0f;
    return octToNdirSigned(p);
}

// Helper function to reflect the folds of the lower hemisphere
// over the diagonals in the octahedral map
INL_DEVICE Float2 octWrap(Float2 v)
{
    return Float2((1.0f - abs(v.y)) * (v.x >= 0.0f ? 1.0f : -1.0f),
                  (1.0f - abs(v.x)) * (v.y >= 0.0f ? 1.0f : -1.0f));
}

/**********************/
// Signed encodings
// Converts a normalized direction to the octahedral map (non-equal area, signed)
// n - normalized direction
// Returns a signed position in octahedral map [-1, 1] for each component
INL_DEVICE Float2 ndirToOctSigned(Float3 n)
{
    // Project the sphere onto the octahedron (|x|+|y|+|z| = 1) and then onto the xy-plane
    Float2 p = n.xy * (1.f / (abs(n.x) + abs(n.y) + abs(n.z)));
    return (n.z < 0.f) ? octWrap(p) : p;
}

/**********************/
// Unorm 32 bit encodings
// Converts a normalized direction to the octahedral map (non-equal area, unsigned normalized)
// n - normalized direction
// Returns a packed 32 bit unsigned normalized position in octahedral map
// The two components of the result are stored in UNORM16 format, [0..1]
INL_DEVICE unsigned int ndirToOctUnorm32(Float3 n)
{
    Float2 p = ndirToOctSigned(n);
    p = saturate(Float2(p.x, p.y) * 0.5f + 0.5f);
    return unsigned int(p.x * 0xfffe) | (unsigned int(p.y * 0xfffe) << 16);
}

// Converts area measure PDF to solid angle measure PDF
INL_DEVICE float PdfAtoW(float pdfA, float distance_, float cosTheta)
{
    return pdfA * (distance_ * distance_) / cosTheta;
}

// Unpack two 16-bit floats (packed into a single 32-bit unsigned int)
// into a Float2.
INL_DEVICE Float2 Unpack_R16G16_FLOAT(unsigned int rg)
{
    // Extract lower 16 bits and upper 16 bits.
    unsigned short h0_bits = static_cast<unsigned short>(rg & 0xFFFF);
    unsigned short h1_bits = static_cast<unsigned short>((rg >> 16) & 0xFFFF);

    // Reinterpret the 16-bit bit patterns as __half.
    __half h0, h1;
    *((unsigned short *)&h0) = h0_bits;
    *((unsigned short *)&h1) = h1_bits;

    // Convert each __half to float.
    float f0 = __half2float(h0);
    float f1 = __half2float(h1);

    return Float2(f0, f1);
}

// Unpack two unsigned int values (each containing two 16-bit floats)
// into a Float4.
// This mirrors the HLSL function that takes a UInt2 where:
//    rgba.x contains two half-precision values (R16, G16)
//    rgba.y contains two half-precision values (B16, A16)
INL_DEVICE Float4 Unpack_R16G16B16A16_FLOAT(UInt2 rgba)
{
    // Unpack first pair (from rgba.x) and second pair (from rgba.y).
    Float2 lower = Unpack_R16G16_FLOAT(rgba.x);
    Float2 upper = Unpack_R16G16_FLOAT(rgba.y);

    return Float4(lower.x, lower.y, upper.x, upper.y);
}

// Convert a float to a 16-bit half in bit representation.
INL_DEVICE unsigned int pack_f32_to_f16_bits(float f)
{
    // Convert float to half with round-to-nearest (returns __half)
    __half h = __float2half_rn(f);
    // Reinterpret the __half's underlying bits as an unsigned short.
    return static_cast<unsigned int>(*reinterpret_cast<unsigned short *>(&h));
}

// Packs two 32-bit floats (in a Float2) into one 32-bit unsigned integer
// by converting each to half precision and packing one in the lower 16 bits
// and the other in the upper 16 bits.
INL_DEVICE unsigned int Pack_R16G16_FLOAT(Float2 rg)
{
    unsigned int r = pack_f32_to_f16_bits(rg.x);       // lower 16 bits
    unsigned int g = pack_f32_to_f16_bits(rg.y) << 16; // upper 16 bits
    return r | g;
}

// Packs a Float4 into a UInt2 by packing each pair of floats into a 32-bit unsigned int.
// For example, rgba.rg is packed into the low unsigned int and rgba.ba into the high unsigned int.
INL_DEVICE UInt2 Pack_R16G16B16A16_FLOAT(Float4 rgba)
{
    unsigned int low = Pack_R16G16_FLOAT(Float2(rgba.x, rgba.y));
    unsigned int high = Pack_R16G16_FLOAT(Float2(rgba.z, rgba.w));
    return UInt2(low, high);
}