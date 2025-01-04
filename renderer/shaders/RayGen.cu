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

    __device__ __inline__ bool TraceNextPath(
        PerRayData *rayData,
        Float4 *absorptionStack,
        Float3 &radiance,
        Float3 &throughput,
        int &stackIdx)
    {
        rayData->f_over_pdf = Float3(0.0f);
        rayData->pdf = 0.0f;
        rayData->radiance = Float3(0.0f);
        rayData->wo = -rayData->wi;        // Direction to observer.
        rayData->ior = Float2(1.0f);       // Reset the volume IORs.
        rayData->distance = RayMax;        // Shoot the next ray with maximum length.
        rayData->flags &= FLAG_CLEAR_MASK; // Clear all non-persistent flags. In this demo only the last diffuse surface interaction stays.
        Float3 extinction;

        // Handle volume absorption of nested materials.
        if (MaterialStackFirst <= stackIdx) // Inside a volume?
        {
            rayData->flags |= FLAG_VOLUME;                // Indicate that we're inside a volume. => At least absorption calculation needs to happen.
            extinction = absorptionStack[stackIdx].xyz;   // There is only volume absorption in this demo, no volume scattering.
            rayData->ior.x = absorptionStack[stackIdx].w; // The IOR of the volume we're inside. Needed for eta calculations in transparent materials.
            if (MaterialStackFirst <= stackIdx - 1)
            {
                rayData->ior.y = absorptionStack[stackIdx - 1].w; // The IOR of the surrounding volume. Needed when potentially leaving a volume to calculate eta in transparent materials.
            }
        }

        // Note that the primary rays (or volume scattering miss cases) wouldn't normally offset the ray t_min by sysSceneEpsilon. Keep it simple here.

        // Put radiance payload pointer into two unsigned integers.
        UInt2 payload = splitPointer(rayData);

        optixTrace(sysParam.topObject,
                   (float3)rayData->pos, (float3)rayData->wi,      // origin, direction
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
        PerRayData *rayData = &perRayData;

        const Int2 imgSize = Int2(optixGetLaunchDimensions());
        const Int2 idx = Int2(optixGetLaunchIndex());
        const Float2 pixelIdx = Float2(idx.x, idx.y);

        rayData->randIdx = 0;

        Float2 samplePixelJitterOffset = rayData->rand2(sysParam);
        samplePixelJitterOffset = Float2(0.5f);

        Float2 sampleUv = (pixelIdx + samplePixelJitterOffset) * sysParam.camera.inversedResolution;
        Float2 centerUv = (pixelIdx + 0.5f) * sysParam.camera.inversedResolution;

        Float3 rayDir = sysParam.camera.uvToWorldDirection(sampleUv);
        Float3 centerRayDir = sysParam.camera.uvToWorldDirection(centerUv);

        rayData->pos = sysParam.camera.pos;
        rayData->wi = rayDir;

        Float4 absorptionStack[MaterialStackSize]; // .xyz == absorptionCoefficient (sigma_a), .w == index of refraction

        Float3 radiance = Float3(0.0f);
        Float3 throughput = Float3(1.0f);

        int stackIdx = MaterialStackEmpty;
        unsigned int depth = 0;

        rayData->absorption_ior = Float4(0.0f, 0.0f, 0.0f, 1.0f); // Assume primary ray starts in vacuum.
        rayData->flags = 0;
        rayData->albedo = Float3(1.0f);
        rayData->normal = Float3(0.0f, 1.0f, 0.0f);
        rayData->roughness = 0.0f;
        rayData->rayConeWidth = 0.0f;
        rayData->rayConeSpread = sysParam.camera.getRayConeWidth(idx);
        rayData->material = 100.0f;
        rayData->sampleIdx = 0;
        rayData->depth = 0;

        bool pathTerminated = false;

        // Float2 outMotionVector = Float2(0.5f);
        Float3 outNormal = Float3(0.0f, 0.0f, 0.0f);
        float outRoughness = 0.0f;
        float outDepth = RayMax;
        float outMaterial = 100.0f;

        rayData->hitFirstDefuseSurface = false;

        static constexpr int BounceLimit = 6;

        while (!pathTerminated)
        {
            // Trace next path
            pathTerminated = !TraceNextPath(rayData, absorptionStack, radiance, throughput, stackIdx);

            if (rayData->depth == BounceLimit - 1)
            {
                pathTerminated = true;
            }

            if (rayData->depth == 0 && sysParam.sampleIndex == 0)
            {
                outNormal = rayData->normal;
                outDepth = rayData->distance;
                outRoughness = rayData->roughness;
                outMaterial = rayData->material;
            }

            ++rayData->depth;
        }

        // Float3 tempRadiance = Float3(0);
        // bool hasGlass = false;
        // for (unsigned int traverseDepth = 0; traverseDepth < 16; ++traverseDepth)
        // {
        //     unsigned int currentMat = (rayData->material >> (traverseDepth * 2)) & 0x3;
        //     if (currentMat == RAY_MAT_FLAG_REFR_AND_REFL)
        //     {
        //         hasGlass = true;
        //         break;
        //     }
        //     else if (currentMat == RAY_MAT_FLAG_DIFFUSE || currentMat == RAY_MAT_FLAG_SKY)
        //     {
        //         break;
        //     }
        // }
        // if (hasGlass)
        // {
        //     samplePixelJitterOffset = rayData->rand2(sysParam);

        //     sampleUv = (pixelIdx + samplePixelJitterOffset) * sysParam.camera.inversedResolution;
        //     centerUv = (pixelIdx + 0.5f) * sysParam.camera.inversedResolution;

        //     rayDir = sysParam.camera.uvToWorldDirection(sampleUv);
        //     centerRayDir = sysParam.camera.uvToWorldDirection(centerUv);

        //     rayData->pos = sysParam.camera.pos;
        //     rayData->wi = rayDir;

        //     throughput = Float3(1.0f);
        //     stackIdx = MaterialStackEmpty;
        //     depth = 0;
        //     rayData->absorption_ior = Float4(0.0f, 0.0f, 0.0f, 1.0f);
        //     rayData->flags = 0;
        //     rayData->albedo = Float3(1.0f);
        //     rayData->normal = Float3(0.0f, -1.0f, 0.0f);
        //     rayData->rayConeWidth = 0.0f;
        //     rayData->rayConeSpread = sysParam.camera.getRayConeWidth(idx);
        //     rayData->sampleIdx = 1;
        //     hitFirstDefuseSurface = false;
        //     pathTerminated = false;
        //     while (!pathTerminated)
        //     {
        //         pathTerminated = !TraceNextPath(rayData, absorptionStack, tempRadiance, throughput, stackIdx);
        //         if (depth == BounceLimit - 1)
        //         {
        //             pathTerminated = true;
        //         }
        //         if (!hitFirstDefuseSurface)
        //         {
        //             ++rayData->depth;
        //         }
        //         if (!hitFirstDefuseSurface && ((rayData->flags & FLAG_DIFFUSED) || pathTerminated))
        //         {
        //             hitFirstDefuseSurface = true;
        //             outAlbedo = lerp3f(outAlbedo, rayData->albedo, 0.5f);
        //             rayData->albedo = Float3(1.0f);
        //         }
        //         ++depth;
        //     }
        //     tempRadiance *= rayData->albedo;
        //     if (isnan(tempRadiance.x) || isnan(tempRadiance.y) || isnan(tempRadiance.z))
        //     {
        //         tempRadiance = Float3(0.5f);
        //     }
        //     radiance = lerp3f(radiance, tempRadiance, 0.5f);
        // }

        /// Debug visualization
        // radiance = outNormal * 0.5f + 0.5f;
        // radiance = ColorRampVisualization(clampf((float)((unsigned short)rayData->material) / 8192.0f));
        // radiance = ColorRampVisualization(expf(-outDepth * 0.1f));
        // radiance = Float3(((outMotionVector - 0.5f) * 10.0f) + 0.5f, 0.0f);
        // outAlbedo = Float3(1.0f);

        if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
        {
            radiance = Float3(0.5f);
        }

        if (sysParam.sampleIndex == 0)
        {
            Store2DFloat1(outMaterial, sysParam.outMaterial, idx);
            Store2DFloat4(Float4(outNormal, outRoughness), sysParam.outNormal, idx);
            Store2DFloat1(outDepth, sysParam.outDepth, idx);
        }

        if (sysParam.sampleIndex > 0)
        {
            Float3 accumulatedAlbedo = Load2DFloat4(sysParam.outAlbedo, idx).xyz;
            Float3 accumulatedRadiance = Load2DFloat4(sysParam.outputBuffer, idx).xyz;

            rayData->albedo = lerp3f(accumulatedAlbedo, rayData->albedo, 1.0f / (float)(sysParam.sampleIndex + 1));
            radiance = lerp3f(accumulatedRadiance, radiance, 1.0f / (float)(sysParam.sampleIndex + 1));
        }

        Store2DFloat4(Float4(rayData->albedo, 1.0f), sysParam.outAlbedo, idx);
        Store2DFloat4(Float4(radiance, outDepth), sysParam.outputBuffer, idx);
    }

}