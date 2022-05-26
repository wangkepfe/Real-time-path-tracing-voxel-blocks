#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include "Common.h"

static constexpr int BounceLimit = 6;
static constexpr float RayMax = 1.e27f;

// Prevent that division by very small floating point values results in huge values, for example dividing by pdf.
static constexpr float DenominatorEpsilon = 1.0e-6f;

static constexpr int MaterialStackEmpty = -1;
static constexpr int MaterialStackFirst = 0;
static constexpr int MaterialStackLast = 0;
static constexpr int MaterialStackSize = 1;

// Set when reaching a closesthit program. Unused in this demo
static constexpr int FLAG_HIT = 0x00000001;
// Set when reaching the __anyhit__shadow program. Indicates visibility test failed.
static constexpr int FLAG_SHADOW = 0x00000002;
// Set by BSDFs which support direct lighting. Not set means specular interaction. Cleared in the closesthit program.
// Used to decide when to do direct lighting and multuiple importance sampling on implicit light hits.
static constexpr int FLAG_DIFFUSE = 0x00000004;
// Set if (0.0f <= wo_dot_ng), means looking onto the front face. (Edge-on is explicitly handled as frontface for the material stack.)
static constexpr int FLAG_FRONTFACE = 0x00000010;
// Pass down material.flags through to the BSDFs.
static constexpr int FLAG_THINWALLED = 0x00000020;
// FLAG_TRANSMISSION is set if there is a transmission. (Can't happen when FLAG_THINWALLED is set.)
static constexpr int FLAG_TRANSMISSION = 0x00000100;
// Set if the material stack is not empty.
static constexpr int FLAG_VOLUME = 0x00001000;
// Highest bit set means terminate path.
static constexpr int FLAG_TERMINATE = 0x80000000;
// Keep flags active in a path segment which need to be tracked along the path.
// In this case only the last surface interaction is kept.
// It's needed to track the last bounce's diffuse state in case a ray hits a light implicitly for multiple importance sampling.
// FLAG_DIFFUSE is reset in the closesthit program.
static constexpr int FLAG_CLEAR_MASK = FLAG_DIFFUSE;

// Currently only containing some vertex attributes in world coordinates.
struct State
{
    float3 normalGeo;
    float3 normal;
    float3 texcoord;
    float3 albedo;
};

// Note that the fields are ordered by CUDA alignment restrictions.
struct PerRayData
{
    float4 absorption_ior; // The absorption coefficient and IOR of the currently hit material.
    float2 ior;            // .x = IOR the ray currently is inside, .y = the IOR of the surrounding volume. The IOR of the current material is in absorption_ior.w!

    float3 pos;     // Current surface hit point or volume sample point, in world space
    float distance; // Distance from the ray origin to the current position, in world space. Needed for absorption of nested materials.

    float3 wo; // Outgoing direction, to observer, in world space.
    float3 wi; // Incoming direction, to light, in world space.

    float3 radiance; // Radiance along the current path segment.
    int flags;       // Bitfield with flags. See FLAG_* defines for its contents.

    float3 f_over_pdf; // BSDF sample throughput, pre-multiplied f_over_pdf = bsdf.f * fabsf(dot(wi, ns) / bsdf.pdf;
    float pdf;         // The last BSDF sample's pdf, tracked for multiple importance sampling.

    float3 extinction; // The current volume's extinction coefficient. (Only absorption in this implementation.)
    float opacity;     // Cutout opacity result.

    unsigned int seed; // Random number generator input.
};

// Alias the PerRayData pointer and an uint2 for the payload split and merge operations. Generates just move instructions.
typedef union
{
    PerRayData* ptr;
    uint2 dat;
} Payload;

INL_DEVICE uint2 splitPointer(PerRayData* ptr)
{
    Payload payload;

    payload.ptr = ptr;

    return payload.dat;
}

INL_DEVICE PerRayData* mergePointer(unsigned int p0, unsigned int p1)
{
    Payload payload;

    payload.dat.x = p0;
    payload.dat.y = p1;

    return payload.ptr;
}
