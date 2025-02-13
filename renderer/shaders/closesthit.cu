#include "SystemParameter.h"
#include "OptixShaderCommon.h"
#include "ShaderDebugUtils.h"
#include "Sampler.h"
#include "SelfHit.h"
#include "Restir.h"

extern "C" __constant__ SystemParameter sysParam;

extern "C" __global__ void __closesthit__radiance()
{
    RayData *rayData = (RayData *)mergePointer(optixGetPayload_0(), optixGetPayload_1());

    // Ray travel distance/time t
    rayData->distance = optixGetRayTmax();

    // Get triangle data
    const GeometryInstanceData *instanceData = reinterpret_cast<const GeometryInstanceData *>(optixGetSbtDataPointer());

    const unsigned int instanceId = optixGetInstanceId();
    const unsigned int meshTriangleIndex = optixGetPrimitiveIndex();

    const Int3 tri = instanceData->indices[meshTriangleIndex];

    const VertexAttributes &va0 = instanceData->attributes[tri.x];
    const VertexAttributes &va1 = instanceData->attributes[tri.y];
    const VertexAttributes &va2 = instanceData->attributes[tri.z];

    const Float3 v0 = va0.vertex;
    const Float3 v1 = va1.vertex;
    const Float3 v2 = va2.vertex;

    float2 bary = optixGetTriangleBarycentrics();

    // Get numerically good hit pos
    Float3 frontPos;
    Float3 backPos;
    Float3 geometricNormal;
    {
        float3 objPos, objNorm;
        float objOffset;
        SelfIntersectionAvoidance::getSafeTriangleSpawnOffset(
            /*out*/ objPos,
            /*out*/ objNorm,
            /*out*/ objOffset,
            v0.to_float3(), v1.to_float3(), v2.to_float3(), // v0, v1, v2
            bary);

        float3 safePos, safeNorm;
        float safeOffset;
        SelfIntersectionAvoidance::transformSafeSpawnOffset(
            /*out*/ safePos,
            /*out*/ safeNorm,
            /*out*/ safeOffset,
            objPos, // from step 4
            objNorm,
            objOffset);

        float3 tmpFrontPos, tmpBackPos;
        SelfIntersectionAvoidance::offsetSpawnPoint(
            /*out*/ tmpFrontPos,
            /*out*/ tmpBackPos,
            /*position =*/safePos,
            /*direction=*/safeNorm,
            /*offset   =*/safeOffset);

        frontPos = Float3(tmpFrontPos);
        backPos = Float3(tmpBackPos);
        geometricNormal = Float3(safeNorm);
    }

    // Default pos
    rayData->pos = frontPos;

    bool hitFrontFace = dot(rayData->wo, geometricNormal) > 0.0f;

    const MaterialParameter &parameters = sysParam.materialParameters[instanceData->materialIndex];
    int materialId = parameters.indexBSDF;

    rayData->material = (float)materialId;

    bool isDiffuse = materialId >= NUM_SPECULAR_BSDF;
    bool isEmissive = materialId == INDEX_BSDF_EMISSIVE;
    // bool isTransmissive = materialId == INDEX_BSDF_SPECULAR_REFLECTION_TRANSMISSION;
    bool isThinfilm = materialId == INDEX_BSDF_DIFFUSE_REFLECTION_TRANSMISSION_THINFILM;

    if (isEmissive)
    {
        if (!rayData->hitFirstDiffuseSurface)
        {
            rayData->radiance = parameters.albedo;
        }

        rayData->shouldTerminate = true;
        return;
    }

    // In the case of thin film, make sure the normal is always pointing towards the incoming ray
    if (isThinfilm && !hitFrontFace)
    {
        geometricNormal = -geometricNormal;
        hitFrontFace = true;

        Float3 tmp = frontPos;
        frontPos = backPos;
        backPos = tmp;
    }

    // UI Box
    if (0)
    {
        if (rayData->depth == 0)
        {
            Float3 highlightPoint[4];
            highlightPoint[0] = sysParam.edgeToHighlight[0];
            highlightPoint[1] = sysParam.edgeToHighlight[1];
            highlightPoint[2] = sysParam.edgeToHighlight[2];
            highlightPoint[3] = sysParam.edgeToHighlight[3];

            const float tolerance = 0.005f;
            Float3 dummy;
            float d0 = PointToSegmentDistance(rayData->pos, highlightPoint[0], highlightPoint[1], dummy);
            float d1 = PointToSegmentDistance(rayData->pos, highlightPoint[1], highlightPoint[2], dummy);
            float d2 = PointToSegmentDistance(rayData->pos, highlightPoint[2], highlightPoint[3], dummy);
            float d3 = PointToSegmentDistance(rayData->pos, highlightPoint[3], highlightPoint[0], dummy);

            if (d0 < tolerance || d1 < tolerance || d2 < tolerance || d3 < tolerance)
            {
                Store2DFloat4(Float4(1.0f), sysParam.outUiBuffer, Int2(optixGetLaunchIndex().x, optixGetLaunchIndex().y));
            }
        }
    }

    MaterialState state;
    state.geometricNormal = geometricNormal;
    state.wo = rayData->wo;

    const Float2 theBarycentrics = Float2(bary);
    const float alpha = 1.0f - theBarycentrics.x - theBarycentrics.y;

    // Texture coordinates
    if (parameters.flags == 2) // use texture coordinates
    {
        state.texcoord = va0.texcoord * alpha + va1.texcoord * theBarycentrics.x + va2.texcoord * theBarycentrics.y;
    }
    else
    {
        if (abs(state.geometricNormal.x) > 0.9f)
        {
            state.texcoord.x = fmodf(rayData->pos.z, parameters.uvScale);
            state.texcoord.y = fmodf(rayData->pos.y, parameters.uvScale);
        }
        else if (abs(state.geometricNormal.y) > 0.9f)
        {
            state.texcoord.x = fmodf(rayData->pos.x, parameters.uvScale);
            state.texcoord.y = fmodf(rayData->pos.z, parameters.uvScale);
        }
        else if (abs(state.geometricNormal.z) > 0.9f)
        {
            state.texcoord.x = fmodf(rayData->pos.x, parameters.uvScale);
            state.texcoord.y = fmodf(rayData->pos.y, parameters.uvScale);
        }
    }

    rayData->hitFrontFace = hitFrontFace;

    // Ray cone spread
    rayData->rayConeWidth += rayData->rayConeSpread * rayData->distance; // +surfaceRayConeSpread; // @TODO Based on the local surface curvature

    // Texture LOD
    state.texcoord /= parameters.uvScale;
    float texMip0Size = parameters.texSize.length();
    float lod = log2f(rayData->rayConeWidth / max(dot(state.geometricNormal, rayData->wo), 0.2f) / parameters.uvScale * 2.0f * texMip0Size) - 3.0f;

    // Albedo
    Float3 albedo = parameters.albedo;
    if (parameters.textureAlbedo != 0)
    {
        const Float3 texColor = Float3(tex2DLod<float4>(parameters.textureAlbedo, state.texcoord.x, state.texcoord.y, lod));
        albedo *= texColor;
    }
    albedo = max3f(albedo, Float3(0.001f)); // Prevent divide-by-zero at demodulation
    state.albedo = albedo;

    // Roughness
    state.roughness = 1.0f;
    if (parameters.textureRoughness != 0)
    {
        state.roughness = tex2DLod<float1>(parameters.textureRoughness, state.texcoord.x, state.texcoord.y, lod).x;
    }

    // Roughness control path regulization: After the first diffuse, all BSDF increase its roughness
    if (rayData->hitFirstDiffuseSurface && state.roughness < 0.5f)
    {
        state.roughness *= 2.0f;
    }

    // Metallic
    state.metallic = 0.0f;
    if (parameters.textureMetallic != 0)
    {
        state.metallic = tex2DLod<float1>(parameters.textureMetallic, state.texcoord.x, state.texcoord.y, lod).x;
    }

    // Normal
    if (parameters.textureNormal != 0)
    {
        Float3 texNormal = Float3(tex2DLod<float4>(parameters.textureNormal, state.texcoord.x, state.texcoord.y, lod));
        state.normal = normalize(texNormal - 0.5f);
        state.normal.x = -state.normal.x;
        state.normal.y = -state.normal.y;
        alignVector(state.geometricNormal, state.normal);
    }
    else
    {
        state.normal = state.geometricNormal;
    }

    // Water
    if (parameters.flags == 1)
    {
        if ((abs(state.geometricNormal.x) > 0.9f) || (abs(state.geometricNormal.z) > 0.9f))
        {
            state.normal = state.geometricNormal;
        }
        else
        {
            Float2 texcoord1 = state.texcoord;
            Float2 texcoord2 = state.texcoord;
            texcoord1.x += sysParam.timeInSecond * 0.1f;
            texcoord2 *= 2.0f;
            texcoord2.y += sysParam.timeInSecond * 0.05f;
            Float3 normal1 = Float3(tex2DLod<float4>(parameters.textureNormal, texcoord1.x, texcoord1.y, lod)) - 0.5f;
            Float3 normal2 = Float3(tex2DLod<float4>(parameters.textureNormal, texcoord2.x, texcoord2.y, lod)) - 0.5f;
            state.normal = normalize(normal1 + normal2 * 2.0f);
            alignVector(state.geometricNormal, state.normal);
        }
    }

    rayData->normal = state.normal;
    rayData->roughness = state.roughness;
    rayData->isCurrentBounceDiffuse = isDiffuse;

    const int indexBsdfSample = materialId;

    Float3 surfWi;
    Float3 surfBsdfOverPdf;
    float surfSampleSurfPdf;

    optixDirectCall<void, MaterialParameter const &, MaterialState const &, RayData *, Float3 &, Float3 &, float &>(indexBsdfSample, parameters, state, rayData, surfWi, surfBsdfOverPdf, surfSampleSurfPdf);

    // Find the correct front face offset position
    bool thinfilmTransmissionEvent = dot(surfWi, state.normal) < 0.0f;
    bool useFrontPos = isThinfilm
                           ? !thinfilmTransmissionEvent
                           : (rayData->hitFrontFace
                                  ? (rayData->isInsideVolume ? true : !rayData->transmissionEvent)
                                  : (rayData->isInsideVolume ? rayData->transmissionEvent : true));
    rayData->pos = useFrontPos ? frontPos : backPos;
    Float3 backfacePos = useFrontPos ? backPos : frontPos;

    // Demodulate albedo and save it in the albedo map
    if (!rayData->hitFirstDiffuseSurface && isDiffuse)
    {
        rayData->hitFirstDiffuseSurface = true;

        // Record the first ever albedo in ray data for output
        if (rayData->sampleIdx == 0)
        {
            rayData->albedo = albedo;
        }
        else
        {
            rayData->albedo = lerp3f(rayData->albedo, albedo, 1.0f / (float)(rayData->sampleIdx + 1));
        }

        // Demodulate the first ever albedo contribution
        surfBsdfOverPdf /= albedo;
        state.albedo = Float3(1.0f); // This albedo is used for later shadow ray contribution
    }

    if (isDiffuse)
    {
        Surface surface;
        surface.pos = rayData->pos;
        surface.backfacePos = backfacePos;
        surface.wo = rayData->wo;
        surface.depth = rayData->distance;
        surface.normal = rayData->normal;
        surface.geoNormal = state.geometricNormal;
        surface.albedo = state.albedo;
        surface.roughness = state.roughness;
        surface.thinfilm = isThinfilm;

        ReSTIRDIParameters params = GetDefaultReSTIRDIParams();
        SampleParameters sampleParams = InitSampleParameters(params);

        LightSample lightSample = {};
        DIReservoir reservoir = SampleLightsForSurface(sysParam, rayData->randIdx, surface, sampleParams, lightSample);
        unsigned int sampledLightIdx = GetDIReservoirLightIndex(reservoir);

        if (lightSample.lightType != LightTypeInvalid)
        {
            Float3 sampleDir = (lightSample.lightType == LightTypeLocalTriangle) ? normalize(lightSample.position - surface.pos) : lightSample.position;
            float maxDistance = (lightSample.lightType == LightTypeLocalTriangle) ? length(lightSample.position - surface.pos) - 1e-2f : RayMax;

            bool isLightVisible = false;
            UInt2 visibilityRayPayload = splitPointer(&isLightVisible);
            Float3 shadowRayOrig = surface.thinfilm ? (dot(sampleDir, surface.normal) > 0.0f ? surface.pos : surface.backfacePos) : surface.pos;
            optixTrace(sysParam.topObject,
                       (float3)shadowRayOrig, (float3)sampleDir,
                       0.0f, maxDistance, 0.0f, // tmin, tmax, time
                       OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                       0, 2, 2,
                       visibilityRayPayload.x, visibilityRayPayload.y);

            if (isLightVisible)
            {
                const int indexBsdfEval = indexBsdfSample + 1;
                const Float4 bsdfPdf = optixDirectCall<Float4, MaterialParameter const &, MaterialState const &, RayData const *, const Float3>(indexBsdfEval, parameters, state, rayData, sampleDir);
                Float3 bsdf = bsdfPdf.xyz;
                float pdf = bsdfPdf.w;

                float cosTheta = fmaxf(0.0f, dot(sampleDir, state.normal));
                Float3 bsdfOverPdf = bsdf * cosTheta;

                lightSample.radiance *= GetDIReservoirInvPdf(reservoir) / lightSample.solidAnglePdf;

                Float3 shadowRayRadiance = bsdfOverPdf * lightSample.radiance;

                // if (OPTIX_CENTER_PIXEL())
                // {
                //     OPTIX_DEBUG_PRINT(bsdf);
                //     OPTIX_DEBUG_PRINT(cosTheta);
                //     OPTIX_DEBUG_PRINT(pdf);
                //     OPTIX_DEBUG_PRINT(GetDIReservoirInvPdf(reservoir));
                //     OPTIX_DEBUG_PRINT(lightSample.radiance);
                //     OPTIX_DEBUG_PRINT(bsdfOverPdf);
                //     OPTIX_DEBUG_PRINT(shadowRayRadiance);
                // }

                rayData->radiance += shadowRayRadiance;
            }
        }
    }

    rayData->wi = surfWi;
    rayData->f_over_pdf = surfBsdfOverPdf;
    rayData->pdf = surfSampleSurfPdf;
}

extern "C" __global__ void __closesthit__bsdf_light()
{
    ShadowRayData *rayData = (ShadowRayData *)mergePointer(optixGetPayload_0(), optixGetPayload_1());

    const GeometryInstanceData *instanceData = reinterpret_cast<const GeometryInstanceData *>(optixGetSbtDataPointer());

    const unsigned int instanceId = optixGetInstanceId();
    const unsigned int meshTriangleIndex = optixGetPrimitiveIndex();

    float2 bary = optixGetTriangleBarycentrics();

    rayData->bary = Float2(bary);

    const MaterialParameter &parameters = sysParam.materialParameters[instanceData->materialIndex];
    int materialId = parameters.indexBSDF;

    bool isEmissive = materialId == INDEX_BSDF_EMISSIVE;

    if (isEmissive)
    {
        int idx = -1;
        {
            int left = 0;
            int right = sysParam.numInstancedLightMesh - 1;

            while (left <= right)
            {
                int mid = (left + right) / 2;
                unsigned int midVal = sysParam.instanceLightMapping[mid].instanceId;

                if (midVal == instanceId)
                {
                    idx = mid;
                    break;
                }
                else if (midVal < instanceId)
                {
                    left = mid + 1;
                }
                else
                {
                    right = mid - 1;
                }
            }
        }
        if (idx != -1)
        {
            rayData->lightIdx = sysParam.instanceLightMapping[idx].lightOffset + meshTriangleIndex;
        }
    }
}