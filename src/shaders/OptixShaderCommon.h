#pragma once

#include <optix.h>
#include "LinearMath.h"

namespace jazzfusion
{

static constexpr int BounceLimit = 6;
static constexpr float DenominatorEpsilon = 1.0e-6f;


static constexpr int FLAG_SHADOW = 0x00000001;

static constexpr int FLAG_FRONTFACE = 0x00000010;
static constexpr int FLAG_THINWALLED = 0x00000020;
static constexpr int FLAG_TRANSMISSION = 0x00000100;
static constexpr int FLAG_VOLUME = 0x00001000;

static constexpr int FLAG_TERMINATE = 0x80000000;

static constexpr int FLAG_CLEAR_MASK = 0x00000000;

// Currently only containing some vertex attributes in world coordinates.
struct State
{
    Float3 normalGeo;
    Float3 normal;
    Float3 texcoord;
    Float3 albedo;
};

// Note that the fields are ordered by CUDA alignment restrictions.
struct __align__(16) PerRayData
{
    Float4 absorption_ior; // The absorption coefficient and IOR of the currently hit material.

    Float3 pos;     // Current surface hit point or volume sample point, in world space
    float distance; // Distance from the ray origin to the current position, in world space. Needed for absorption of nested materials.

    Float3 wo; // Outgoing direction, to observer, in world space.
    uint seed; // Random number generator input.

    Float3 wi; // Incoming direction, to light, in world space.
    uint depth;

    Float3 radiance; // Radiance along the current path segment.
    int flags;       // Bitfield with flags. See FLAG_* defines for its contents.

    Float3 f_over_pdf; // BSDF sample throughput, pre-multiplied f_over_pdf = bsdf.f * fabsf(dot(wi, ns) / bsdf.pdf;
    float pdf;         // The last BSDF sample's pdf, tracked for multiple importance sampling.

    Float3 extinction; // The current volume's extinction coefficient. (Only absorption in this implementation.)
    float opacity;     // Cutout opacity result.

    Float2 ior;            // .x = IOR the ray currently is inside, .y = the IOR of the surrounding volume. The IOR of the current material is in absorption_ior.w!
    uint material;
    uint padding1;

    Float3 normal;
    uint padding2;

    Float3 albedo;
    uint padding3;
};

// Alias the PerRayData pointer and an UInt2 for the payload split and merge operations. Generates just move instructions.

union Payload
{
    INL_DEVICE Payload(PerRayData* ptrIn) : ptr{ ptrIn } {}
    INL_DEVICE Payload(const UInt2& datIn) : dat{ datIn } {}

    PerRayData* ptr;
    UInt2 dat;
};

INL_DEVICE UInt2 splitPointer(PerRayData* ptr)
{
    Payload payload{ ptr };
    return payload.dat;
}

INL_DEVICE PerRayData* mergePointer(uint p0, uint p1)
{
    Payload payload{ UInt2(p0, p1) };
    return payload.ptr;
}

}