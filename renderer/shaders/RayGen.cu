#include "SystemParameter.h"
#include "OptixShaderCommon.h"
#include "ShaderDebugUtils.h"
#include "Sampler.h"

extern "C" __constant__ SystemParameter sysParam;

__device__ __inline__ bool TraceNextPath(
    RayData *rayData,
    Float4 &absorptionIor,
    int &volumnIdx,
    Float3 &radiance,
    Float3 &throughput)
{
    rayData->f_over_pdf = Float3(0.0f);
    rayData->pdf = 0.0f;
    rayData->radiance = Float3(0.0f);
    rayData->wo = -rayData->wi;

    rayData->distance = RayMax;

    rayData->shouldTerminate = false;

    rayData->isLastBounceDiffuse = rayData->isCurrentBounceDiffuse;
    rayData->isCurrentBounceDiffuse = false;

    rayData->hitFrontFace = false;
    rayData->transmissionEvent = false;
    rayData->isInsideVolume = false;

    Float3 extinction;
    if (volumnIdx > 0)
    {
        rayData->isInsideVolume = true;
        extinction = absorptionIor.xyz;
    }

    UInt2 payload = splitPointer(rayData);

    // {
    //     optixTrace(sysParam.topObject,
    //                (float3)rayData->pos, (float3)rayData->wi, // origin, direction
    //                0.0f, RayMax, 0.0f,                        // tmin, tmax, time
    //                OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
    //                0, 2, 0, // SBToffset, SBTstride, missSBTIndex
    //                payload.x, payload.y);
    // }

    {
        optixTraverse(sysParam.topObject,
                      (float3)rayData->pos, (float3)rayData->wi, // origin, direction
                      0.0f, RayMax, 0.0f,                        // tmin, tmax, time
                      OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                      0, 2, 0, // SBToffset, SBTstride, missSBTIndex
                      payload.x, payload.y);
        unsigned int hint = 0;
        if (optixHitObjectIsHit())
        {
            const GeometryInstanceData *instanceData = reinterpret_cast<const GeometryInstanceData *>(optixHitObjectGetSbtDataPointer());
            const MaterialParameter &parameters = sysParam.materialParameters[instanceData->materialIndex];
            int materialId = parameters.indexBSDF;
            hint = materialId;
        }
        optixReorder(hint, NumOfBitsMaxBsdfIndex);
        optixInvoke(payload.x, payload.y);
    }

    if (rayData->isInsideVolume)
    {
        throughput *= exp3f(-rayData->distance * extinction);
    }

    radiance += throughput * rayData->radiance;

    if (rayData->shouldTerminate || rayData->pdf <= 0.0f || isNull(rayData->f_over_pdf))
    {
        return false;
    }

    throughput *= rayData->f_over_pdf;

    if (rayData->transmissionEvent)
    {
        if (rayData->hitFrontFace) // Enter
        {
            volumnIdx = 1;
            absorptionIor = rayData->absorptionIor;
        }
        else // Exit
        {
            volumnIdx = 0;
        }
    }

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

    rayData->pos = sysParam.camera.pos;
    rayData->wi = rayDir;

    Float4 absorptionIor; // .xyz == absorptionCoefficient (sigma_a), .w == index of refraction
    int volumnIdx = 0;

    Float3 radiance = Float3(0.0f);
    Float3 throughput = Float3(1.0f);

    rayData->absorptionIor = Float4(0.0f, 0.0f, 0.0f, 1.0f);
    rayData->albedo = Float3(1.0f);
    rayData->normal = Float3(0.0f, 1.0f, 0.0f);
    rayData->roughness = 0.0f;
    rayData->rayConeWidth = 0.0f;
    rayData->rayConeSpread = sysParam.camera.getRayConeWidth(idx);
    rayData->material = 100.0f;
    rayData->sampleIdx = 0;
    rayData->depth = 0;
    rayData->isShadowRay = false;
    rayData->isCurrentBounceDiffuse = false;
    rayData->isLastBounceDiffuse = false;

    bool pathTerminated = false;

    Float3 outNormal = Float3(0.0f, 0.0f, 0.0f);
    Float3 outGeoNormal = Float3(0.0f, 0.0f, 0.0f);
    float outThinfilm = 0.0f;
    float outRoughness = 0.0f;
    float outDepth = RayMax;
    float outMaterial = 100.0f;

    rayData->hitFirstDiffuseSurface = false;

    static constexpr int BounceLimit = 1;

    while (!pathTerminated)
    {
        pathTerminated = !TraceNextPath(rayData, absorptionIor, volumnIdx, radiance, throughput);

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
            outGeoNormal = rayData->geoNormal;
            outThinfilm = rayData->hitThinfilm ? 1.0f : 0.0f;
        }

        ++rayData->depth;
    }

    if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
    {
        radiance = Float3(0.5f);
    }

    if (sysParam.sampleIndex == 0)
    {
        Store2DFloat1(outMaterial, sysParam.outMaterial, idx);
        Store2DFloat4(Float4(outNormal, outRoughness), sysParam.outNormal, idx);
        Store2DFloat1(outDepth, sysParam.outDepth, idx);

        Store2DFloat4(Float4(outGeoNormal, outThinfilm), sysParam.outGeoNormalThinfilmBuffer, idx);
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