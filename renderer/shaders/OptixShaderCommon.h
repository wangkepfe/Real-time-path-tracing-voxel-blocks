#pragma once

#include <optix.h>
#include "LinearMath.h"
#include "SystemParameter.h"

namespace jazzfusion
{

static constexpr int BounceLimit = 6;
static constexpr float DenominatorEpsilon = 1.0e-6f;


static constexpr int FLAG_SHADOW = 0x00000001;
static constexpr int FLAG_DIFFUSED = 0x00000002;

static constexpr int FLAG_FRONTFACE = 0x00000010;
static constexpr int FLAG_THINWALLED = 0x00000020;
static constexpr int FLAG_TRANSMISSION = 0x00000100;
static constexpr int FLAG_VOLUME = 0x00001000;

static constexpr int FLAG_SHADOW_HIT = 0x00002000;
static constexpr int FLAG_SHADOW_GLASS_HIT = 0x00004000;

static constexpr int FLAG_TERMINATE = 0x80000000;

static constexpr int FLAG_CLEAR_MASK = FLAG_DIFFUSED | FLAG_SHADOW;

// Currently only containing some vertex attributes in world coordinates.
struct State
{
    // Float3 normalGeo;
    Float3 normal;
    Float3 texcoord;
};

// Note that the fields are ordered by CUDA alignment restrictions.
struct __align__(16) PerRayData
{
    Float4 absorption_ior; // The absorption coefficient and IOR of the currently hit material.

    Float3 pos;     // Current surface hit point or volume sample point, in world space
    float distance; // Distance from the ray origin to the current position, in world space. Needed for absorption of nested materials.

    Float3 wo; // Outgoing direction, to observer, in world space.
    uint flags;

    Float3 wi; // Incoming direction, to light, in world space.
    uint depth;

    Float3 radiance; // Radiance along the current path segment.
    uint material;

    Float3 f_over_pdf; // BSDF sample throughput, pre-multiplied f_over_pdf = bsdf.f * fabsf(dot(wi, ns) / bsdf.pdf;
    float pdf;         // The last BSDF sample's pdf, tracked for multiple importance sampling.

    Float2 ior;            // .x = IOR the ray currently is inside, .y = the IOR of the surrounding volume. The IOR of the current material is in absorption_ior.w!
    float rayConeSpread;
    float rayConeWidth;

    Float3 normal;
    int randIdx;

    Float3 albedo;
    uint sampleIdx;

    // float totalDistance;
    // Float3 lightEmission;
    // float lightPdf;
    // Float3 centerRayDir;
    // uint unused;
    // float randNums[8];

    INL_DEVICE float rand(const SystemParameter & sysParam)
    {
        UInt2 idx = UInt2(optixGetLaunchIndex());
        return sysParam.randGen.rand(idx.x, idx.y, sysParam.iterationIndex * sysParam.samplePerIteration + sysParam.sampleIndex, randIdx++);
    }

    INL_DEVICE Float2 rand2(const SystemParameter & sysParam)
    {
        return Float2(rand(sysParam), rand(sysParam));
    }
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