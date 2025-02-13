#pragma once

#include <optix.h>
#include "LinearMath.h"
#include "SystemParameter.h"

static const unsigned int DIReservoir_LightValidBit = 0x80000000;
static const unsigned int DIReservoir_LightIndexMask = 0x7FFFFFFF;
static const unsigned int InvalidLightIndex = 0x7FFFFFFF;
static const unsigned int SkyLightIndex = 0x7FFFFFFE;
static const unsigned int SunLightIndex = 0x7FFFFFFD;

// Currently only containing some vertex attributes in world coordinates.
struct MaterialState
{
    Float3 normal;
    Float2 texcoord;
    Float3 geometricNormal;
    Float3 albedo;
    float roughness;
    float metallic;
    Float3 wo;
};

// Note that the fields are ordered by CUDA alignment restrictions.
struct __align__(16) RayData
{
    Float4 absorptionIor; // The absorption coefficient and IOR of the currently hit material.

    Float3 pos;     // Current surface hit point or volume sample point, in world space
    float distance; // Distance from the ray origin to the current position, in world space. Needed for absorption of nested materials.

    Float3 wo; // Outgoing direction, point away from surface, to observer, in world space.
    float roughness;

    Float3 wi; // Incoming direction, point away from surface, to light, in world space.
    unsigned int depth;

    Float3 radiance; // Radiance along the current path segment.
    float material;

    Float3 f_over_pdf; // The last BSDF sample's throughput, pre-multiplied f_over_pdf = bsdf.f * fabsf(dot(wi, ns) / bsdf.pdf;
    float pdf;         // The last BSDF sample's pdf, tracked for multiple importance sampling.

    Float3 normal;
    int randIdx;

    Float3 albedo;
    unsigned int sampleIdx;

    float rayConeSpread;
    float rayConeWidth;
    unsigned int shadowRayLightIdx;

    bool hitFirstDiffuseSurface;
    bool shouldTerminate;
    bool isShadowRay;
    bool hasShadowRayHitAnything;
    bool hasShadowRayHitTransmissiveSurface;
    bool hasShadowRayHitThinfilmSurface;
    bool hasShadowRayHitLocalLight;
    bool isCurrentBounceDiffuse;
    bool isLastBounceDiffuse;
    bool hitFrontFace;
    bool transmissionEvent;
    bool isInsideVolume;
};

struct __align__(16) ShadowRayData
{
    unsigned int lightIdx;
    Float2 bary;
};

union Payload
{
    INL_DEVICE Payload(void *ptrIn) : ptr{ptrIn} {}
    INL_DEVICE Payload(const UInt2 &datIn) : dat{datIn} {}

    void *ptr;
    UInt2 dat;
};

INL_DEVICE UInt2 splitPointer(void *ptr)
{
    Payload payload{ptr};
    return payload.dat;
}

INL_DEVICE void *mergePointer(unsigned int p0, unsigned int p1)
{
    Payload payload{UInt2(p0, p1)};
    return payload.ptr;
}