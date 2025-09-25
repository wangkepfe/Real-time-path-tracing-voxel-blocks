#include "SystemParameter.h"
#include "Common.h"
#include "OptixShaderCommon.h"
#include "ShaderDebugUtils.h"
#include "Sampler.h"

extern "C" __constant__ SystemParameter sysParam;

__device__ __inline__ bool TraceNextPath(
    RayData *rayData,
    Float4 &absorptionIor,
    int &volumnIdx,
    Float3 &radiance,
    Float3 &throughput,
    Float3 &diffuseRadiance,
    Float3 &specularRadiance,
    bool isFirstBounceSpecular)
{
    rayData->bsdfOverPdf = Float3(1.0f);
    rayData->pdf = 0.0f;
    rayData->radiance = Float3(0.0f);
    rayData->diffuseRadiance = Float3(0.0f);
    rayData->specularRadiance = Float3(0.0f);
    rayData->wo = -rayData->wi;

    rayData->distance = RayMax;

    rayData->shouldTerminate = false;

    rayData->isLastBounceDiffuse = rayData->isCurrentBounceDiffuse;
    rayData->isCurrentBounceDiffuse = false;

    rayData->hitFrontFace = false;
    rayData->transmissionEvent = false;
    rayData->isInsideVolume = false;
    rayData->lastMissWasEnvironment = false;

    // Float3 extinction;
    // if (volumnIdx > 0)
    // {
    //     rayData->isInsideVolume = true;
    //     extinction = absorptionIor.xyz;
    // }

    UInt2 payload = splitPointer(rayData);

#if 0
    optixTrace(sysParam.topObject,
                (float3)rayData->pos, (float3)rayData->wi, // origin, direction
                0.0f, RayMax, 0.0f,                        // tmin, tmax, time
                OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                0, 2, 0, // SBToffset, SBTstride, missSBTIndex
                payload.x, payload.y);
#endif

    optixTraverse(sysParam.topObject,
                  (float3)rayData->pos, (float3)rayData->wi, // origin, direction
                  0.0f, RayMax, 0.0f,                        // tmin, tmax, time
                  OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                  0, 2, 0, // SBToffset, SBTstride, missSBTIndex
                  payload.x, payload.y);
    unsigned int hint = 0;
    if (optixHitObjectIsHit())
    {
        const GeometryInstanceData *instanceData = reinterpret_cast<const GeometryInstanceData *>(optixHitObjectGetSbtDataPointer());
        const MaterialParameter &parameters = sysParam.materialParameters[instanceData->materialIndex];
        int materialId = parameters.materialId;
        hint = materialId;
    }
    optixReorder(hint, NumOfBitsMaxBsdfIndex);
    optixInvoke(payload.x, payload.y);

    // if (rayData->isInsideVolume)
    // {
    //     throughput *= exp3f(-rayData->distance * extinction);
    // }

    radiance += throughput * rayData->radiance;
    
    // Accumulate lighting contributions based on bounce depth and first bounce type
    if (rayData->depth == 0)
    {
        // First surface: use separated diffuse/specular from BSDF evaluation
        // But for mirror surfaces, all contribution should go to specular
        if (isFirstBounceSpecular)
        {
            // First bounce is specular (mirror) - all lighting goes to specular
            specularRadiance += throughput * (rayData->diffuseRadiance + rayData->specularRadiance);
        }
        else
        {
            // First bounce is diffuse - respect BSDF separation
            diffuseRadiance += throughput * rayData->diffuseRadiance;
            specularRadiance += throughput * rayData->specularRadiance;
        }
    }
    else
    {
        // Later bounces: all lighting goes to buffer determined by first bounce type
        if (isFirstBounceSpecular)
        {
            specularRadiance += throughput * rayData->radiance;
        }
        else
        {
            diffuseRadiance += throughput * rayData->radiance;
        }
    }

    // if (OPTIX_CENTER_PIXEL())
    // {
    //     OPTIX_DEBUG_PRINT(throughput);
    //     OPTIX_DEBUG_PRINT(rayData->radiance);
    // }

    if (rayData->shouldTerminate || rayData->pdf <= 0.0f || isNull(rayData->bsdfOverPdf))
    {
        return false;
    }

    throughput *= rayData->bsdfOverPdf;

    // if (rayData->transmissionEvent)
    // {
    //     if (rayData->hitFrontFace) // Enter
    //     {
    //         volumnIdx = 1;
    //         // absorptionIor = rayData->absorptionIor;
    //     }
    //     else // Exit
    //     {
    //         volumnIdx = 0;
    //     }
    // }

    return true;
}

extern "C" __global__ void __raygen__pathtracer()
{
    RayData perRayData;
    RayData *rayData = &perRayData;

    const Int2 imgSize = Int2(optixGetLaunchDimensions());
    const Int2 idx = Int2(optixGetLaunchIndex());
    const Float2 pixelIdx = Float2(idx.x, idx.y);

    rayData->randIdx = 0;

    Float2 samplePixelJitterOffset = rand2(sysParam, rayData->randIdx);

    const Float2 sampleUv = (pixelIdx + samplePixelJitterOffset) * sysParam.camera.inversedResolution;
    const Float2 centerUv = (pixelIdx + 0.5f) * sysParam.camera.inversedResolution;
    const Float3 rayDir = sysParam.camera.uvToWorldDirection(sampleUv);
    const Float3 centerRayDir = sysParam.camera.uvToWorldDirection(centerUv); // unused

    // if (OPTIX_CENTER_PIXEL())
    // {
    //     OPTIX_DEBUG_PRINT(sampleUv);
    // }

    rayData->pos = sysParam.camera.pos;
    rayData->wi = rayDir;

    Float4 absorptionIor = Float4(0.0f, 0.0f, 0.0f, 1.0f); // .xyz == absorptionCoefficient (sigma_a), .w == index of refraction
    int volumnIdx = 0;

    Float3 radiance = Float3(0.0f);
    Float3 throughput = Float3(1.0f);
    
    // Separate diffuse and specular accumulation
    Float3 diffuseRadiance = Float3(0.0f);
    Float3 specularRadiance = Float3(0.0f);
    bool isFirstBounceSpecular = false;

    rayData->rayConeWidth = 0.0f;
    rayData->rayConeSpread = sysParam.camera.getRayConeWidth(idx);
    rayData->depth = 0;
    rayData->isCurrentBounceDiffuse = false;
    rayData->isLastBounceDiffuse = false;
    rayData->isCurrentSampleSpecular = false; // Primary ray starts as camera ray
    rayData->firstToSecondHitDistance = 0.0f;
    bool secondHitCaptured = false;

    bool pathTerminated = false;

    float primaryRayHitDist = RayMax;

    rayData->hitFirstDiffuseSurface = false;

    int totalBounceLimit = 5;
    int diffuseBounceLimit = 1;

    int totalBounce = 0;
    int diffuseBounce = 0;

    while (!pathTerminated)
    {
        pathTerminated = !TraceNextPath(rayData, absorptionIor, volumnIdx, radiance, throughput, diffuseRadiance, specularRadiance, isFirstBounceSpecular);

        // Capture first bounce type AFTER the first trace
        if (totalBounce == 0)
        {
            isFirstBounceSpecular = rayData->isCurrentSampleSpecular;
        }

        ++totalBounce;
        if (rayData->isCurrentBounceDiffuse)
        {
            ++diffuseBounce;
        }

        if (totalBounce == totalBounceLimit || diffuseBounce == diffuseBounceLimit)
        {
            pathTerminated = true;
        }

        if (rayData->depth == 0)
        {
            primaryRayHitDist = rayData->distance;

            if (!secondHitCaptured && rayData->lastMissWasEnvironment)
            {
                rayData->firstToSecondHitDistance = 0.0f;
                secondHitCaptured = true;
            }
        }
        else if (!secondHitCaptured && rayData->depth == 1)
        {
            if (rayData->lastMissWasEnvironment)
            {
                rayData->firstToSecondHitDistance = RayMax;
            }
            else
            {
                rayData->firstToSecondHitDistance = rayData->distance;
            }
            secondHitCaptured = true;
        }

        ++rayData->depth;
    }

    if (!secondHitCaptured)
    {
        if (rayData->lastMissWasEnvironment)
        {
            rayData->firstToSecondHitDistance = 0.0f;
        }
        else
        {
            rayData->firstToSecondHitDistance = RayMax;
        }
    }

    if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
    {
        radiance = Float3(0.5f);
    }
    
    // Handle NaN for separate diffuse/specular radiance
    if (isnan(diffuseRadiance.x) || isnan(diffuseRadiance.y) || isnan(diffuseRadiance.z))
    {
        diffuseRadiance = Float3(0.0f);
    }
    if (isnan(specularRadiance.x) || isnan(specularRadiance.y) || isnan(specularRadiance.z))
    {
        specularRadiance = Float3(0.0f);
    }

    Store2DFloat1(primaryRayHitDist, sysParam.depthBuffer, idx);
    Store2DFloat4(Float4(radiance, primaryRayHitDist), sysParam.illuminationBuffer, idx);
    
    // Store separated diffuse and specular radiance from path tracing
    Store2DFloat4(Float4(diffuseRadiance, primaryRayHitDist), sysParam.diffuseIlluminationBuffer, idx);
    Store2DFloat4(Float4(specularRadiance, rayData->firstToSecondHitDistance), sysParam.specularIlluminationBuffer, idx);
}


