#include "SystemParameter.h"
#include "OptixShaderCommon.h"
#include "ShaderDebugUtils.h"

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
    Camera camera,
    Int2   idx,
    Float2 randPixelOffset,
    Float2 randAperture)
{
    // [0, 1] coordinates
    Float2 uv = (Float2(idx.x, idx.y) + randPixelOffset) * camera.inversedResolution;
    Float2 uvCenter = (Float2(idx.x, idx.y) + 0.5f) * camera.inversedResolution;
    sampleUv = uv;

    // [0, 1] -> [1, -1], since left/up vector should be 1 when uv is 0
    uv = uv * -2.0f + 1.0f;
    uvCenter = uvCenter * -2.0f + 1.0f;

    // Point on the image plane
    Float3 pointOnImagePlane = camera.adjustedFront + camera.adjustedLeft * uv.x + camera.adjustedUp * uv.y;
    Float3 pointOnImagePlaneCenter = camera.adjustedFront + camera.adjustedLeft * uvCenter.x + camera.adjustedUp * uvCenter.y;

    // Point on the aperture
    Float2 diskSample = ConcentricSampleDisk(randAperture);
    Float3 pointOnAperture = diskSample.x * camera.apertureLeft + diskSample.y * camera.apertureUp;

    // ray
    orig = camera.pos + pointOnAperture;
    dir = normalize(pointOnImagePlane - pointOnAperture);
    centerDir = normalize(pointOnImagePlaneCenter);
}

__device__ __inline__ bool TraceNextPath(PerRayData& rayData, Float4* absorptionStack, Float3& radiance, Float3& throughput, int& stackIdx, int& depth)
{
    rayData.wo = -rayData.wi;              // Direction to observer.
    rayData.ior = Float2(1.0f);   // Reset the volume IORs.
    rayData.distance = RayMax; // Shoot the next ray with maximum length.
    rayData.flags &= FLAG_CLEAR_MASK;  // Clear all non-persistent flags. In this demo only the last diffuse surface interaction stays.

    // Handle volume absorption of nested materials.
    if (MaterialStackFirst <= stackIdx) // Inside a volume?
    {
        rayData.flags |= FLAG_VOLUME;                                // Indicate that we're inside a volume. => At least absorption calculation needs to happen.
        rayData.extinction = absorptionStack[stackIdx].xyz; // There is only volume absorption in this demo, no volume scattering.
        rayData.ior.x = absorptionStack[stackIdx].w;                 // The IOR of the volume we're inside. Needed for eta calculations in transparent materials.
        if (MaterialStackFirst <= stackIdx - 1)
        {
            rayData.ior.y = absorptionStack[stackIdx - 1].w; // The IOR of the surrounding volume. Needed when potentially leaving a volume to calculate eta in transparent materials.
        }
    }

    // Note that the primary rays (or volume scattering miss cases) wouldn't normally offset the ray t_min by sysSceneEpsilon. Keep it simple here.

    // Put radiance payload pointer into two unsigned integers.
    UInt2 payload = splitPointer(&rayData);

    optixTrace(sysParam.topObject,
        (float3)rayData.pos, (float3)rayData.wi,                               // origin, direction
        sysParam.sceneEpsilon, rayData.distance, 0.0f, // tmin, tmax, time
        OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0,
        payload.x, payload.y);

    // This renderer supports nested volumes.
    if (rayData.flags & FLAG_VOLUME)
    {
        // We're inside a volume. Calculate the extinction along the current path segment in any case.
        // The transmittance along the current path segment inside a volume needs to attenuate the ray throughput with the extinction
        // before it modulates the radiance of the hitpoint.
        throughput *= exp3f(-rayData.distance * rayData.extinction);
    }

    radiance += throughput * rayData.radiance;

    // Path termination by miss shader or sample() routines.
    // If terminate is true, f_over_pdf and pdf might be undefined.
    if ((rayData.flags & FLAG_TERMINATE) || rayData.pdf <= 0.0f || isNull(rayData.f_over_pdf))
    {
        return false;
    }

    // PERF f_over_pdf already contains the proper throughput adjustment for diffuse materials: f * (fabsf(optix::dot(rayData.wi, state.normal)) / rayData.pdf);
    throughput *= rayData.f_over_pdf;

    // Adjust the material volume stack if the geometry is not thin-walled but a border between two volumes
    // and the outgoing ray direction was a transmission.
    if ((rayData.flags & (FLAG_THINWALLED | FLAG_TRANSMISSION)) == FLAG_TRANSMISSION)
    {
        // Transmission.
        if (rayData.flags & FLAG_FRONTFACE) // Entered a new volume?
        {
            // Push the entered material's volume properties onto the volume stack.
            // rtAssert((stackIdx < MaterialStackLast), 1); // Overflow?
            stackIdx = min(stackIdx + 1, MaterialStackLast);
            absorptionStack[stackIdx] = rayData.absorption_ior;
        }
        else // Exited the current volume?
        {
            // Pop the top of stack material volume.
            // This assert fires and is intended because I tuned the frontface checks so that there are more exits than enters at silhouettes.
            // rtAssert((MaterialStackEmpty < stackIdx), 0); // Underflow?
            stackIdx = max(stackIdx - 1, MaterialStackEmpty);
        }
    }

    ++depth; // Next path segment.

    return true;
}

extern "C" __global__ void __raygen__pathtracer()
{
    PerRayData rayData;

    const UInt2 imgSize = UInt2(optixGetLaunchDimensions());
    const UInt2 idx = UInt2(optixGetLaunchIndex());

    rayData.seed = tea<4>(idx.y * imgSize.x + idx.x, sysParam.iterationIndex);

    const Float2 screen = Float2(imgSize.x, imgSize.y);
    const Float2 pixel = Float2(idx.x, idx.y);
    const Float2 samplePixelJitterOffset = rng2(rayData.seed);
    const Float2 sampleApertureJitterOffset = rng2(rayData.seed);

    Float3 centerDir;
    Float2 sampleUv;
    GenerateRay(rayData.pos, rayData.wi, centerDir, sampleUv, sysParam.camera, idx, samplePixelJitterOffset, sampleApertureJitterOffset);

    // This renderer supports nested volumes. The absorption coefficient and IOR of the volume the ray is currently inside.
    Float4 absorptionStack[MaterialStackSize]; // .xyz == absorptionCoefficient (sigma_a), .w == index of refraction

    Float3 radiance = Float3(1.0f);
    Float3 throughput = Float3(1.0f);

    int stackIdx = MaterialStackEmpty;
    int depth = 0;

    rayData.absorption_ior = Float4(0.0f, 0.0f, 0.0f, 1.0f); // Assume primary ray starts in vacuum.
    rayData.flags = 0;

    while (depth < BounceLimit)
    {
        if (TraceNextPath(rayData, absorptionStack, radiance, throughput, stackIdx, depth) == false)
            break;
    }

    if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
    {
        radiance = Float3(10000.0f, 0.0f, 0.0f);
    }

    surf2Dwrite(make_float4(radiance.x, radiance.y, radiance.z, 1.0f), sysParam.outputBuffer, idx.x * sizeof(float4), idx.y, cudaBoundaryModeClamp);
}

}