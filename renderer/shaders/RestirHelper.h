#pragma once

#include <cuda_fp16.h>

#include "LinearMath.h"

INL_DEVICE Float3 sampleTriangle(Float2 rndSample)
{
    float sqrtx = sqrt(rndSample.x);

    return Float3(
        1.0f - sqrtx,
        sqrtx * (1.0f - rndSample.y),
        sqrtx * rndSample.y);
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