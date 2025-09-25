#pragma once

#include <optix.h>
#include "LinearMath.h"
#include "SystemParameter.h"

static const unsigned int DIReservoir_LightValidBit = 0x80000000;
static const unsigned int DIReservoir_LightIndexMask = 0x7FFFFFFF;
static const unsigned int InvalidLightIndex = 0x7FFFFFFF;
static const unsigned int SkyLightIndex = 0x7FFFFFFE;
static const unsigned int SunLightIndex = 0x7FFFFFFD;

// Note that the fields are ordered by CUDA alignment restrictions.
struct __align__(16) RayData
{
    Float3 pos;     // Current surface hit point or volume sample point, in world space
    float distance; // Distance from the ray origin to the current position, in world space. Needed for absorption of nested materials.
    float firstToSecondHitDistance; // Distance from the first hit to the second hit
    Float3 wo;      // Outgoing direction, point away from surface, to observer, in world space.
    Float3 wi;      // Incoming direction, point away from surface, to light, in world space.
    unsigned int depth;
    Float3 radiance;    // Total radiance along the current path segment.
    Float3 diffuseRadiance;  // Diffuse radiance contribution
    Float3 specularRadiance; // Specular radiance contribution
    Float3 bsdfOverPdf; // The last BSDF sample's throughput, pre-multiplied bsdfOverPdf = bsdf.f * fabsf(dot(wi, ns) / bsdf.pdf;
    float pdf;          // The last BSDF sample's pdf, tracked for multiple importance sampling.
    int randIdx;
    float rayConeSpread;
    float rayConeWidth;
    bool hitFirstDiffuseSurface;
    bool shouldTerminate;
    bool isCurrentBounceDiffuse;
    bool isLastBounceDiffuse;
    bool hitFrontFace;
    bool transmissionEvent;
    bool isInsideVolume;
    bool isCurrentSampleSpecular;
    bool lastMissWasEnvironment;
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


