#include "SystemParameter.h"
#include "OptixShaderCommon.h"
#include "ShaderDebugUtils.h"
#include "Sampler.h"

namespace jazzfusion
{

extern "C" __constant__ SystemParameter sysParam;

__device__ static constexpr int MaterialStackEmpty = -1;
__device__ static constexpr int MaterialStackFirst = 0;
__device__ static constexpr int MaterialStackLast = 0;
__device__ static constexpr int MaterialStackSize = 1;

__device__ void inline GenerateRay(
    Float3& orig,
    Float3& dir,
    Float3& centerDir,
    Float2& sampleUv,
    Float2& centerUv,
    Camera camera,
    Int2   idx,
    Int2   imgSize,
    Float2 randPixelOffset,
    Float2 randAperture)
{
    // [0, 1] coordinates
    Float2 uv = (Float2(idx.x, idx.y) + randPixelOffset) * camera.inversedResolution;
    Float2 uvCenter = (Float2(idx.x, idx.y) + 0.5f) * camera.inversedResolution;
    sampleUv = uv;
    centerUv = uvCenter;

    // [0, 1] -> [1, -1], since left/up vector should be 1 when uv is 0
    uv = uv * Float2(-2.0f, 2.0f) + Float2(1.0f, -1.0f);

    // Point on the image plane
    Float3 pointOnImagePlane = camera.adjustedFront + camera.adjustedLeft * uv.x + camera.adjustedUp * uv.y;

    // Point on the aperture
    Float2 diskSample = ConcentricSampleDisk(randAperture);
    Float3 pointOnAperture = diskSample.x * camera.apertureLeft + diskSample.y * camera.apertureUp;

    // ray
    orig = camera.pos + pointOnAperture;
    dir = normalize(pointOnImagePlane - pointOnAperture);

    // center
    uvCenter = uvCenter * Float2(-2.0f, 2.0f) + Float2(1.0f, -1.0f);
    Float3 pointOnImagePlaneCenter = camera.adjustedFront + camera.adjustedLeft * uvCenter.x + camera.adjustedUp * uvCenter.y;
    centerDir = normalize(pointOnImagePlaneCenter);
}

__device__ float inline GetRayConeWidth(Camera camera, Int2 idx)
{
    Float2 pixelCenter = (Float2(idx.x, idx.y) + 0.5f) - Float2(camera.resolution.x, camera.resolution.y) / 2;
    Float2 pixelOffset = copysignf2(Float2(0.5f), pixelCenter);

    Float2 uvNear = (pixelCenter - pixelOffset) * camera.inversedResolution * 2; // [-1, 1]
    Float2 uvFar = (pixelCenter + pixelOffset) * camera.inversedResolution * 2;

    Float2 halfFovLength = Float2(tanf(camera.fov.x / 2), tanf(camera.fov.y / 2));

    Float2 pointOnPlaneNear = uvNear * halfFovLength;
    Float2 pointOnPlaneFar = uvFar * halfFovLength;

    float angleNear = atanf(pointOnPlaneNear.length());
    float angleFar = atanf(pointOnPlaneFar.length());
    float pixelAngleWidth = angleFar - angleNear;

    return pixelAngleWidth;
}

__device__ __inline__ bool TraceNextPath(
    PerRayData* rayData,
    Float4* absorptionStack,
    Float3& radiance,
    Float3& throughput,
    int& stackIdx)
{
    rayData->wo = -rayData->wi;              // Direction to observer.
    rayData->ior = Float2(1.0f);   // Reset the volume IORs.
    rayData->distance = RayMax; // Shoot the next ray with maximum length.
    rayData->flags &= FLAG_CLEAR_MASK;  // Clear all non-persistent flags. In this demo only the last diffuse surface interaction stays.
    Float3 extinction;

    // Handle volume absorption of nested materials.
    if (MaterialStackFirst <= stackIdx) // Inside a volume?
    {
        rayData->flags |= FLAG_VOLUME;                                // Indicate that we're inside a volume. => At least absorption calculation needs to happen.
        extinction = absorptionStack[stackIdx].xyz; // There is only volume absorption in this demo, no volume scattering.
        rayData->ior.x = absorptionStack[stackIdx].w;                 // The IOR of the volume we're inside. Needed for eta calculations in transparent materials.
        if (MaterialStackFirst <= stackIdx - 1)
        {
            rayData->ior.y = absorptionStack[stackIdx - 1].w; // The IOR of the surrounding volume. Needed when potentially leaving a volume to calculate eta in transparent materials.
        }
    }

    // Note that the primary rays (or volume scattering miss cases) wouldn't normally offset the ray t_min by sysSceneEpsilon. Keep it simple here.

    // Put radiance payload pointer into two unsigned integers.
    UInt2 payload = splitPointer(rayData);

    optixTrace(sysParam.topObject,
        (float3)rayData->pos, (float3)rayData->wi,                               // origin, direction
        sysParam.sceneEpsilon, rayData->distance, 0.0f, // tmin, tmax, time
        OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0,
        payload.x, payload.y);

    // This renderer supports nested volumes.
    if (rayData->flags & FLAG_VOLUME)
    {
        // We're inside a volume. Calculate the extinction along the current path segment in any case.
        // The transmittance along the current path segment inside a volume needs to attenuate the ray throughput with the extinction
        // before it modulates the radiance of the hitpoint.
        throughput *= exp3f(-rayData->distance * extinction);
    }

    radiance += throughput * rayData->radiance;

    // Path termination by miss shader or sample() routines.
    // If terminate is true, f_over_pdf and pdf might be undefined.
    if ((rayData->flags & FLAG_TERMINATE) || rayData->pdf <= 0.0f || isNull(rayData->f_over_pdf))
    {
        return false;
    }

    // PERF f_over_pdf already contains the proper throughput adjustment for diffuse materials: f * (fabsf(optix::dot(rayData->wi, state.normal)) / rayData->pdf);
    throughput *= rayData->f_over_pdf;

    // Adjust the material volume stack if the geometry is not thin-walled but a border between two volumes
    // and the outgoing ray direction was a transmission.
    if ((rayData->flags & (FLAG_THINWALLED | FLAG_TRANSMISSION)) == FLAG_TRANSMISSION)
    {
        // Transmission.
        if (rayData->flags & FLAG_FRONTFACE) // Entered a new volume?
        {
            // Push the entered material's volume properties onto the volume stack.
            // rtAssert((stackIdx < MaterialStackLast), 1); // Overflow?
            stackIdx = min(stackIdx + 1, MaterialStackLast);
            absorptionStack[stackIdx] = rayData->absorption_ior;
        }
        else // Exited the current volume?
        {
            // Pop the top of stack material volume.
            // This assert fires and is intended because I tuned the frontface checks so that there are more exits than enters at silhouettes.
            // rtAssert((MaterialStackEmpty < stackIdx), 0); // Underflow?
            stackIdx = max(stackIdx - 1, MaterialStackEmpty);
        }
    }

    return true;
}

__forceinline__ __device__ unsigned int IntegerHash(unsigned int a)
{
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

extern "C" __global__ void __raygen__pathtracer()
{
    PerRayData perRayData;
    PerRayData* rayData = &perRayData;

    const Int2 imgSize = Int2(optixGetLaunchDimensions());
    Int2 idx = Int2(optixGetLaunchIndex());

    rayData->randIdx = 0;

    const Float2 screen = Float2(imgSize.x, imgSize.y);
    const Float2 pixel = Float2(idx.x, idx.y);

    Float2 samplePixelJitterOffset = rayData->rand2(sysParam);//Float2(sysParam.randGen.rand(idx.x, idx.y, sysParam.iterationIndex, rayData->randIdx++), sysParam.randGen.rand(idx.x, idx.y, sysParam.iterationIndex, rayData->randIdx++));
    Float2 sampleApertureJitterOffset = rayData->rand2(sysParam);

    Float2 sampleUv;
    Float3 centerRayDir;
    Float2 centerUv;
    GenerateRay(rayData->pos, rayData->wi, centerRayDir, sampleUv, centerUv, sysParam.camera, idx, imgSize, samplePixelJitterOffset, sampleApertureJitterOffset);

    // This renderer supports nested volumes. The absorption coefficient and IOR of the volume the ray is currently inside.
    Float4 absorptionStack[MaterialStackSize]; // .xyz == absorptionCoefficient (sigma_a), .w == index of refraction

    Float3 radiance = Float3(0.0f);
    Float3 throughput = Float3(1.0f);

    int stackIdx = MaterialStackEmpty;
    uint depth = 0;

    rayData->absorption_ior = Float4(0.0f, 0.0f, 0.0f, 1.0f); // Assume primary ray starts in vacuum.
    rayData->flags = 0;
    rayData->albedo = Float3(1.0f);
    rayData->normal = Float3(0.0f, 1.0f, 0.0f);
    rayData->rayConeWidth = 0.0f;
    rayData->rayConeSpread = GetRayConeWidth(sysParam.camera, idx);
    rayData->material = 0u;
    rayData->sampleIdx = 0;
    rayData->depth = 0;

    bool pathTerminated = false;

    Float2 outMotionVector = Float2(0.5f);
    Float3 outNormal = Float3(0.0f, 0.0f, 0.0f);
    Float3 outAlbedo = Float3(1.0f);
    float outDepth = RayMax;
    bool hitFirstDefuseSurface = false;

    while (!pathTerminated)
    {
        // Trace next path
        pathTerminated = !TraceNextPath(rayData, absorptionStack, radiance, throughput, stackIdx);

        if (depth == BounceLimit - 1)
        {
            pathTerminated = true;
        }

        // First hit
        if (depth == 0)
        {
            if (sysParam.sampleIndex == 0)
            {
                outNormal = rayData->normal;
                outDepth = rayData->distance;

                Float2 lastFrameSampleUv;
                if (rayData->material == 0)
                {
                    lastFrameSampleUv = sysParam.historyCamera.WorldToScreenSpace(rayData->wi, sysParam.camera.tanHalfFov);
                }
                else
                {
                    lastFrameSampleUv = sysParam.historyCamera.WorldToScreenSpace(rayData->pos - sysParam.historyCamera.pos, sysParam.camera.tanHalfFov);
                }
                outMotionVector += lastFrameSampleUv - sampleUv;
            }
        }

        if (!hitFirstDefuseSurface)
        {
            ++rayData->depth;
        }

        // First diffuse hit
        if (!hitFirstDefuseSurface && ((rayData->flags & FLAG_DIFFUSED) || pathTerminated))
        {
            hitFirstDefuseSurface = true;
            outAlbedo = rayData->albedo;
            rayData->albedo = Float3(1.0f);
        }

        ++depth; // Next path segment.
    }

    // if (OPTIX_CENTER_PIXEL() && outMaterial == 3)
    // {
    //     OPTIX_DEBUG_PRINT(Float4(depth, pathTerminated, needWriteOutput, 0));
    // }
    radiance *= rayData->albedo;

    Float3 tempRadiance = Float3(0);

    // if (OPTIX_CENTER_PIXEL())
    // {
    //     radiance = Float3(100.0f, 0.0f, 0.0f);
    // }

    bool hasGlass = false;
    for (uint traverseDepth = 0; traverseDepth < 16; ++traverseDepth)
    {
        uint currentMat = (rayData->material >> (traverseDepth * 2)) & 0x3;
        if (currentMat == RAY_MAT_FLAG_REFR_AND_REFL)
        {
            hasGlass = true;
            break;
        }
        else if (currentMat == RAY_MAT_FLAG_DIFFUSE || currentMat == RAY_MAT_FLAG_SKY)
        {
            break;
        }
    }

    if (hasGlass)
    {
        samplePixelJitterOffset = rayData->rand2(sysParam);
        sampleApertureJitterOffset = rayData->rand2(sysParam);
        GenerateRay(rayData->pos, rayData->wi, centerRayDir, sampleUv, centerUv, sysParam.camera, idx, imgSize, samplePixelJitterOffset, sampleApertureJitterOffset);
        throughput = Float3(1.0f);
        stackIdx = MaterialStackEmpty;
        depth = 0;
        rayData->absorption_ior = Float4(0.0f, 0.0f, 0.0f, 1.0f);
        rayData->flags = 0;
        rayData->albedo = Float3(1.0f);
        rayData->normal = Float3(0.0f, -1.0f, 0.0f);
        rayData->rayConeWidth = 0.0f;
        rayData->rayConeSpread = GetRayConeWidth(sysParam.camera, idx);
        rayData->sampleIdx = 1;
        hitFirstDefuseSurface = false;
        pathTerminated = false;
        while (!pathTerminated)
        {
            pathTerminated = !TraceNextPath(rayData, absorptionStack, tempRadiance, throughput, stackIdx);
            if (depth == BounceLimit - 1)
            {
                pathTerminated = true;
            }
            if (!hitFirstDefuseSurface)
            {
                ++rayData->depth;
            }
            if (!hitFirstDefuseSurface && ((rayData->flags & FLAG_DIFFUSED) || pathTerminated))
            {
                hitFirstDefuseSurface = true;
                outAlbedo = lerp3f(outAlbedo, rayData->albedo, 0.5f);
                rayData->albedo = Float3(1.0f);
            }
            ++depth;

        }
        tempRadiance *= rayData->albedo;
        if (isnan(tempRadiance.x) || isnan(tempRadiance.y) || isnan(tempRadiance.z))
        {
            tempRadiance = Float3(0.5f);
        }
        radiance = lerp3f(radiance, tempRadiance, 0.5f);
    }

    /// Debug visualization
    // radiance = outNormal * 0.5f + 0.5f;
    // radiance = ColorRampVisualization(clampf((float)((ushort)rayData->material) / 8192.0f));
    // radiance = ColorRampVisualization(expf(-outDepth * 0.1f));
    // radiance = Float3(((outMotionVector - 0.5f) * 10.0f) + 0.5f, 0.0f);
    // outAlbedo = Float3(1.0f);

    // if (OPTIX_CENTER_PIXEL())
    // {
    //     OPTIX_DEBUG_PRINT(rayData->material);
    //     outAlbedo = Float3(100.0f, 0.0f, 0.0f);
    // }

    if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
    {
        radiance = Float3(0.5f);
    }

    if (sysParam.sampleIndex == 0)
    {
        Store2DUshort1((ushort)rayData->material, sysParam.outMaterial, idx);
        Store2DHalf4(Float4(outNormal, 0.0f), sysParam.outNormal, idx);
        Store2DHalf1(outDepth, sysParam.outDepth, idx);
        Store2DHalf2(outMotionVector, sysParam.outMotionVector, idx);
    }

    if (sysParam.sampleIndex > 0)
    {
        Float3 accumulatedAlbedo = Load2DHalf4(sysParam.outAlbedo, idx).xyz;
        Float3 accumulatedRadiance = Load2DHalf4(sysParam.outputBuffer, idx).xyz;

        outAlbedo = lerp3f(accumulatedAlbedo, outAlbedo, 1.0f / (float)(sysParam.sampleIndex + 1));
        radiance = lerp3f(accumulatedRadiance, radiance, 1.0f / (float)(sysParam.sampleIndex + 1));
    }

    Store2DHalf4(Float4(outAlbedo, 1.0f), sysParam.outAlbedo, idx);
    Store2DHalf4(Float4(radiance, 1.0f), sysParam.outputBuffer, idx);
}

}