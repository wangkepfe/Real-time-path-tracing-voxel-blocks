#pragma once

#include <optix.h>
#include "LinearMath.h"
#include "SystemParameter.h"

namespace jazzfusion
{
    static constexpr float DenominatorEpsilon = 1.0e-6f;

    // Currently only containing some vertex attributes in world coordinates.
    struct MaterialState
    {
        // Float3 normalGeo;
        Float3 normal;
        Float2 texcoord;
        Float3 geometricNormal;
        float roughness;
        Float3 wo;
    };

    // Note that the fields are ordered by CUDA alignment restrictions.
    struct __align__(16) PerRayData
    {
        Float4 absorption_ior; // The absorption coefficient and IOR of the currently hit material.

        Float3 pos;     // Current surface hit point or volume sample point, in world space
        float distance; // Distance from the ray origin to the current position, in world space. Needed for absorption of nested materials.

        Float3 wo; // Outgoing direction, to observer, in world space.

        Float3 wi; // Incoming direction, to light, in world space.
        unsigned int depth;

        Float3 radiance; // Radiance along the current path segment.
        float material;

        Float3 f_over_pdf; // BSDF sample throughput, pre-multiplied f_over_pdf = bsdf.f * fabsf(dot(wi, ns) / bsdf.pdf;
        float pdf;         // The last BSDF sample's pdf, tracked for multiple importance sampling.

        float rayConeSpread;
        float rayConeWidth;

        Float3 normal;
        int randIdx;

        Float3 albedo;
        unsigned int sampleIdx;

        float roughness;

        // float totalDistance;
        bool hitFirstDiffuseSurface;
        bool shouldTerminate;
        bool isShadowRay;
        bool hasShadowRayHitAnything;
        bool hasShadowRayHitTransmissiveSurface;
        bool isCurrentBounceDiffuse;
        bool isLastBounceDiffuse;
        bool isHitFrontFace;
        bool isHitTransmission;
        bool isInsideVolume;

        // Float3 lightEmission;
        // float lightPdf;
        // Float3 centerRayDir;
        // unsigned int unused;
        // float randNums[8];

        INL_DEVICE float rand(const SystemParameter &sysParam)
        {
            UInt2 idx = UInt2(optixGetLaunchIndex());
            return sysParam.randGen.rand(idx.x, idx.y, sysParam.iterationIndex * sysParam.samplePerIteration + sysParam.sampleIndex, randIdx++);
        }

        INL_DEVICE Float2 rand2(const SystemParameter &sysParam)
        {
            return Float2(rand(sysParam), rand(sysParam));
        }
    };

    // Alias the PerRayData pointer and an UInt2 for the payload split and merge operations. Generates just move instructions.

    union Payload
    {
        INL_DEVICE Payload(PerRayData *ptrIn) : ptr{ptrIn} {}
        INL_DEVICE Payload(const UInt2 &datIn) : dat{datIn} {}

        PerRayData *ptr;
        UInt2 dat;
    };

    INL_DEVICE UInt2 splitPointer(PerRayData *ptr)
    {
        Payload payload{ptr};
        return payload.dat;
    }

    INL_DEVICE PerRayData *mergePointer(unsigned int p0, unsigned int p1)
    {
        Payload payload{UInt2(p0, p1)};
        return payload.ptr;
    }

}