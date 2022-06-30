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

//--------------------------------
//
//          ----------> u (0,1)
//        |
//        |
//        |
//        V
//
//        v (0,1)

//--------------------------------
//
//         ^  y
//         |
//         |
//         |
//        / --------->  x
//       /
//      /
//     v  z

__device__ void inline GenerateRay(
    Float3& orig,
    Float3& dir,
    Float3& centerDir,
    Float2& sampleUv,
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
    int& stackIdx,
    int& depth)
{
    rayData->wo = -rayData->wi;              // Direction to observer.
    rayData->ior = Float2(1.0f);   // Reset the volume IORs.
    rayData->distance = RayMax; // Shoot the next ray with maximum length.
    rayData->flags &= FLAG_CLEAR_MASK;  // Clear all non-persistent flags. In this demo only the last diffuse surface interaction stays.
    rayData->depth = depth;

    // Handle volume absorption of nested materials.
    if (MaterialStackFirst <= stackIdx) // Inside a volume?
    {
        rayData->flags |= FLAG_VOLUME;                                // Indicate that we're inside a volume. => At least absorption calculation needs to happen.
        rayData->extinction = absorptionStack[stackIdx].xyz; // There is only volume absorption in this demo, no volume scattering.
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
        throughput *= exp3f(-rayData->distance * rayData->extinction);
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

extern "C" __global__ void __raygen__pathtracer()
{
    PerRayData perRayData;
    PerRayData* rayData = &perRayData;

    const UInt2 imgSize = UInt2(optixGetLaunchDimensions());
    UInt2 idx = UInt2(optixGetLaunchIndex());

    if constexpr (0)
    {
        BlueNoiseRandGenerator randGen = sysParam.randGen;
        randGen.idx.x = idx.x;
        randGen.idx.y = idx.y;
        uint seed = tea<4>(idx.y * imgSize.x + idx.x, sysParam.iterationIndex);
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            randGen.sampleIdx = sysParam.iterationIndex * 4 + i;

#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float blueNoise = randGen.Rand(j);
                float pureRand = rng(seed);
                rayData->randNums[i * 4 + j] = lerpf(blueNoise, pureRand, sysParam.noiseBlend);
            }
        }
    }

    uint seed = tea<4>(idx.y * imgSize.x + idx.x, sysParam.iterationIndex);
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        rayData->randNums[i] = rng(seed);
    }
    rayData->randNumIdx = 0;

    const Float2 screen = Float2(imgSize.x, imgSize.y);
    const Float2 pixel = Float2(idx.x, idx.y);

    const Float2 samplePixelJitterOffset = rayData->rand2();
    const Float2 sampleApertureJitterOffset = rayData->rand2();

    Float2 sampleUv;
    GenerateRay(rayData->pos, rayData->wi, rayData->centerRayDir, sampleUv, sysParam.camera, idx, imgSize, samplePixelJitterOffset, sampleApertureJitterOffset);

    // This renderer supports nested volumes. The absorption coefficient and IOR of the volume the ray is currently inside.
    Float4 absorptionStack[MaterialStackSize]; // .xyz == absorptionCoefficient (sigma_a), .w == index of refraction

    Float3 radiance = Float3(0.0f);
    Float3 throughput = Float3(1.0f);

    int stackIdx = MaterialStackEmpty;
    int depth = 0;

    rayData->absorption_ior = Float4(0.0f, 0.0f, 0.0f, 1.0f); // Assume primary ray starts in vacuum.
    rayData->flags = 0;
    rayData->albedo = Float3(1.0f);
    rayData->normal = Float3(0.0f, -1.0f, 0.0f);
    rayData->rayConeWidth = 0.0f;
    rayData->rayConeSpread = GetRayConeWidth(sysParam.camera, idx);
    rayData->totalDistance = 0.0f;

    bool pathTerminated = false;

    Float2 outMotionVector = Float2(0.5f);
    Float3 outNormal = Float3(0.0f, 1.0f, 0.0f);
    Float3 outAlbedo = Float3(1.0f);
    float outDepth = RayMax;
    ushort outMaterial = SKY_MATERIAL_ID;
    uint multipliedMaterial = 1;
    bool needWriteOutput = true;

    while (!pathTerminated)
    {
        pathTerminated = !TraceNextPath(rayData, absorptionStack, radiance, throughput, stackIdx, depth);

        if (depth == BounceLimit - 1)
        {
            pathTerminated = true;
        }

        // Multiply material ID
        multipliedMaterial = multipliedMaterial * NUM_MATERIALS + rayData->material;

        if (needWriteOutput && ((rayData->flags & FLAG_DIFFUSED) || pathTerminated))
        {
            needWriteOutput = false;

            // outNormal = rayData->normal;
            // outDepth = rayData->totalDistance;
            outMaterial = (ushort)multipliedMaterial;

            // Only denoise the first diffuse albedo, reset the albedo and apply the other albedo to radiance
            outAlbedo = rayData->albedo;
            rayData->albedo = Float3(1.0f);
        }

        if (depth == 0)
        {
            outNormal = rayData->normal;
            outDepth = rayData->totalDistance;
            // Store2DHalf4(Float4(rayData->normal, 1.0f), sysParam.outNormalFront, idx);
            // Store2DHalf1(rayData->totalDistance, sysParam.outDepthFront, idx);
            // Store2DUshort1(rayData->material, sysParam.outMaterialFront, idx);
        }

        if (depth == 0 && !pathTerminated)
        {
            Float2 lastFrameSampleUv = sysParam.historyCamera.WorldToScreenSpace(rayData->pos, sysParam.camera.tanHalfFov);
            outMotionVector += lastFrameSampleUv - sampleUv;
        }

        ++depth; // Next path segment.
    }

    // if (OPTIX_CENTER_PIXEL() && outMaterial == 3)
    // {
    //     OPTIX_DEBUG_PRINT(Float4(depth, pathTerminated, needWriteOutput, 0));
    // }

    if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
    {
        radiance = Float3(0.5f);
    }

    radiance *= rayData->albedo;

    // Debug visualization
    // radiance = outNormal * 0.5f + 0.5f;
    // radiance = ColorRampVisualization(clampf((float)outMaterial / 24.0f));
    // radiance = ColorRampVisualization(expf(-outDepth * 0.1f));
    // outAlbedo = Float3(1.0f);

    // if (OPTIX_CENTER_PIXEL())
    // {
    //     // OPTIX_DEBUG_PRINT((float)outMaterial);
    //     outAlbedo = Float3(100.0f, 0.0f, 0.0f);
    // }

    Store2DHalf4(Float4(outNormal, 1.0f), sysParam.outNormal, idx);
    Store2DHalf1(outDepth, sysParam.outDepth, idx);
    Store2DUshort1(outMaterial, sysParam.outMaterial, idx);
    Store2DHalf2(outMotionVector, sysParam.outMotionVector, idx);
    Store2DHalf4(Float4(outAlbedo, 1.0f), sysParam.outAlbedo, idx);
    Store2DHalf4(Float4(radiance, 1.0f), sysParam.outputBuffer, idx);
}

}