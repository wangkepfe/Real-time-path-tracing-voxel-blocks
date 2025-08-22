#include "SystemParameter.h"
#include "OptixShaderCommon.h"
#include "ShaderDebugUtils.h"
#include "Sampler.h"
#include "SelfHit.h"
#include "Restir.h"

extern "C" __constant__ SystemParameter sysParam;

extern "C" __global__ void __closesthit__radiance()
{
    Int2 pixelPosition = Int2(optixGetLaunchIndex());

    RayData *rayData = (RayData *)mergePointer(optixGetPayload_0(), optixGetPayload_1());

    // Ray travel distance/time t
    rayData->distance = optixGetRayTmax();
    int &randIdx = rayData->randIdx;

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
    Float3 geoNormal;
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
        geoNormal = Float3(safeNorm);
    }

    // Default pos
    rayData->pos = frontPos;

    bool hitFrontFace = dot(rayData->wo, geoNormal) > 0.0f;

    const MaterialParameter &parameters = sysParam.materialParameters[instanceData->materialIndex];

    int materialId = parameters.materialId;
    bool isEmissive = parameters.isEmissive;
    bool isThinfilm = parameters.isThinfilm;

    if (isEmissive)
    {
        if (!rayData->hitFirstDiffuseSurface)
        {
            rayData->radiance = parameters.albedo;

            Store2DFloat4(Float4(1.0f), sysParam.albedoBuffer, pixelPosition);
            Store2DFloat1((float)(0xFFFF), sysParam.materialBuffer, pixelPosition);
            Store2DFloat4(Float4(0.0f, -1.0f, 0.0f, 0.0f), sysParam.normalRoughnessBuffer, pixelPosition);
            Store2DFloat4(Float4(0.0f, -1.0f, 0.0f, 0.0f), sysParam.geoNormalThinfilmBuffer, pixelPosition);
            Store2DFloat4(Float4(0.0f, 0.0f, 0.0f, 0.0f), sysParam.materialParameterBuffer, pixelPosition);
        }

        rayData->shouldTerminate = true;
        return;
    }

    // In the case of thin film, make sure the normal is always pointing towards the incoming ray
    if (isThinfilm && !hitFrontFace)
    {
        geoNormal = -geoNormal;
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
                Store2DFloat4(Float4(1.0f), sysParam.UIBuffer, Int2(optixGetLaunchIndex().x, optixGetLaunchIndex().y));
            }
        }
    }

    MaterialState state;
    state.geoNormal = geoNormal;
    state.wo = rayData->wo;

    const Float2 theBarycentrics = Float2(bary);
    const float alpha = 1.0f - theBarycentrics.x - theBarycentrics.y;

    // Texture coordinates
    Float2 texCoords;
    if (parameters.useWorldGridUV) // use texture coordinates
    {
        if (abs(state.geoNormal.x) > 0.9f)
        {
            texCoords.x = fmodf(rayData->pos.z, parameters.uvScale);
            texCoords.y = fmodf(rayData->pos.y, parameters.uvScale);
        }
        else if (abs(state.geoNormal.y) > 0.9f)
        {
            texCoords.x = fmodf(rayData->pos.x, parameters.uvScale);
            texCoords.y = fmodf(rayData->pos.z, parameters.uvScale);
        }
        else if (abs(state.geoNormal.z) > 0.9f)
        {
            texCoords.x = fmodf(rayData->pos.x, parameters.uvScale);
            texCoords.y = fmodf(rayData->pos.y, parameters.uvScale);
        }
    }
    else
    {
        texCoords = va0.texcoord * alpha + va1.texcoord * theBarycentrics.x + va2.texcoord * theBarycentrics.y;
    }

    rayData->hitFrontFace = hitFrontFace;

    // Ray cone spread
    rayData->rayConeWidth += rayData->rayConeSpread * rayData->distance; // +surfaceRayConeSpread; // @TODO Based on the local surface curvature

    // Texture LOD
    texCoords /= parameters.uvScale;
    float texMip0Size = parameters.texSize.length();
    float lod = log2f(rayData->rayConeWidth / max(dot(state.geoNormal, rayData->wo), 0.2f) / parameters.uvScale * 2.0f * texMip0Size) - 3.0f;

    // Albedo
    state.albedo = parameters.albedo;
    if (parameters.textureAlbedo != 0)
    {
        const Float3 texColor = Float3(tex2DLod<float4>(parameters.textureAlbedo, texCoords.x, texCoords.y, lod));
        state.albedo *= texColor;
    }
    state.albedo = max3f(state.albedo, Float3(0.001f)); // Prevent divide-by-zero at demodulation

    // Roughness
    state.roughness = parameters.roughness;
    if (parameters.textureRoughness != 0)
    {
        state.roughness = tex2DLod<float1>(parameters.textureRoughness, texCoords.x, texCoords.y, lod).x;
    }

    // Roughness control path regulization: After the first diffuse, all BSDF increase its roughness
    if (rayData->hitFirstDiffuseSurface)
    {
        state.roughness = min(state.roughness * 2.0f + 0.1f, 1.0f);
    }

    bool isDiffuse = state.roughness > roughnessThreshold;

    // Metallic
    state.metallic = parameters.metallic;
    if (parameters.textureMetallic != 0)
    {
        state.metallic = tex2DLod<float1>(parameters.textureMetallic, texCoords.x, texCoords.y, lod).x > 0.5f ? true : false;
    }

    state.translucency = parameters.translucency;

    // Normal
    if (parameters.textureNormal != 0)
    {
        Float3 texNormal = Float3(tex2DLod<float4>(parameters.textureNormal, texCoords.x, texCoords.y, lod));
        state.normal = normalize(texNormal - 0.5f);
        state.normal.x = -state.normal.x;
        state.normal.y = -state.normal.y;
        alignVector(state.geoNormal, state.normal);
    }
    else
    {
        state.normal = state.geoNormal;
    }
    constexpr float normalMapStrength = 0.2f;
    state.normal = lerp3f(state.geoNormal, state.normal, normalMapStrength);

    rayData->isCurrentBounceDiffuse = isDiffuse;

    // Write Gbuffer data
    if (rayData->depth == 0)
    {
        Store2DFloat1((float)parameters.materialId, sysParam.materialBuffer, pixelPosition);
        Store2DFloat4(Float4(state.normal, state.roughness), sysParam.normalRoughnessBuffer, pixelPosition);
        Store2DFloat4(Float4(state.normal, isThinfilm ? 1.0f : 0.0f), sysParam.geoNormalThinfilmBuffer, pixelPosition);
        Store2DFloat4(Float4(state.metallic ? 1.0f : 0.0f, state.translucency, 0.0f, 0.0f), sysParam.materialParameterBuffer, pixelPosition);
    }

    // const int indexBsdfSample = materialId;

    Float3 bsdfSampleWi;
    Float3 bsdfSampleBsdfOverPdf;
    float bsdfSamplePdf;
    // optixDirectCall<void, MaterialParameter const &, MaterialState const &, RayData *, int &, Float3 &, Float3 &, float &>(indexBsdfSample, parameters, state, rayData, rayData->randIdx, bsdfSampleWi, bsdfSampleBsdfOverPdf, bsdfSamplePdf);

    rayData->transmissionEvent = false;

    DisneyBSDFSample(rand4(sysParam, randIdx), state.normal, state.geoNormal, state.wo, state.albedo, state.metallic, state.translucency, state.roughness, bsdfSampleWi, bsdfSampleBsdfOverPdf, bsdfSamplePdf, rayData->transmissionEvent);

    if (bsdfSamplePdf <= 0.0f)
    {
        rayData->shouldTerminate = true;
    }

    // Find the correct front face offset position
    // bool useFrontPos = isThinfilm
    //                        ? (dot(bsdfSampleWi, state.normal) >= 0.0f)
    //                        : (rayData->hitFrontFace
    //                               ? (rayData->isInsideVolume ? true : !rayData->transmissionEvent)
    //                               : (rayData->isInsideVolume ? rayData->transmissionEvent : true));
    rayData->pos = isThinfilm ? (dot(bsdfSampleWi, state.normal) > 0.0f ? frontPos : backPos) : frontPos;

    rayData->wi = bsdfSampleWi;
    rayData->bsdfOverPdf = bsdfSampleBsdfOverPdf;
    rayData->pdf = bsdfSamplePdf;

    // Demodulate albedo and save it in the albedo map
    bool skipAlbedoInShadowRayContribution = false;
    if (rayData->depth == 0)
    {
        Store2DFloat4(Float4(state.albedo, 1.0f), sysParam.albedoBuffer, pixelPosition);

        // Demodulate the first albedo contribution
        bsdfSampleBsdfOverPdf /= state.albedo;
        skipAlbedoInShadowRayContribution = true;
    }

    bool enableRIS = false;
    if (!rayData->hitFirstDiffuseSurface && isDiffuse) // TODO
    {
        rayData->hitFirstDiffuseSurface = true;
        enableRIS = true;
    }

    bool enableReSTIR = true && enableRIS;

    // Specular hit = no shadow ray
    if (!isDiffuse)
    {
        if (enableReSTIR)
        {
            StoreDIReservoir(EmptyDIReservoir(), pixelPosition);
        }

        return;
    }

    if (!enableRIS)
    {
        return;
    }

    Surface surface;
    surface.state = state;
    surface.materialId = materialId;
    surface.pos = rayData->pos;
    surface.depth = rayData->distance;
    surface.isThinfilm = isThinfilm;

    LightSample lightSample = {};
    DIReservoir risReservoir = EmptyDIReservoir();

    bool skipSunSample = !isThinfilm && (dot(state.normal, sysParam.sunDir) < 0.0f || dot(state.geoNormal, sysParam.sunDir) < 0.0f);

    const int numLocalLightSamples = sysParam.accumulatedLocalLightLuminance > 0.0f ? 8 : 0;
    const int numSunLightSamples = skipSunSample ? 0 : 1;
    const int numSkyLightSamples = 1;
    const int numBrdfSamples = 1;

    const int numMisSamples = numLocalLightSamples + numSunLightSamples + numSkyLightSamples + numBrdfSamples;

    const float localLightMisWeight = float(numLocalLightSamples) / numMisSamples;
    const float sunLightMisWeight = float(numSunLightSamples) / numMisSamples;
    const float skyLightMisWeight = float(numSkyLightSamples) / numMisSamples;
    const float brdfMisWeight = float(numBrdfSamples) / numMisSamples;

    const float totalSceneLuminance = sysParam.accumulatedLocalLightLuminance + sysParam.accumulatedSkyLuminance + sysParam.accumulatedSunLuminance;
    // const float localLightSourcePdf = sysParam.accumulatedLocalLightLuminance / totalSceneLuminance;
    // const float sunLightSourcePdf = sysParam.accumulatedSunLuminance / totalSceneLuminance;
    // const float skyLightSourcePdf = sysParam.accumulatedSkyLuminance / totalSceneLuminance;

    const float brdfCutoff = 0.0f; // 0.0001f;

    // Local light
    DIReservoir localReservoir = EmptyDIReservoir();
    LightSample localSample = LightSample{};

    for (unsigned int i = 0; i < numLocalLightSamples; i++)
    {
        float sourcePdf;
        int lightIndex = sysParam.lightAliasTable->sample(rand(sysParam, randIdx), sourcePdf);

        // Boundary check for light index
        if (lightIndex >= sysParam.numLights)
            continue;

        LightInfo lightInfo = sysParam.lights[lightIndex];

        Float2 uv = rand2(sysParam, randIdx);

        TriangleLight triLight = TriangleLight::Create(lightInfo);
        LightSample candidateSample = triLight.calcSample(uv, surface.pos);

        float blendedSourcePdf = LightBrdfMisWeight(surface, candidateSample, sourcePdf, localLightMisWeight, false, brdfMisWeight, brdfCutoff);
        float targetPdf = GetLightSampleTargetPdfForSurface(candidateSample, surface);
        float risRnd = rand(sysParam, randIdx);

        if (blendedSourcePdf != 0.0f)
        {
            bool selected = StreamSample(localReservoir, lightIndex, uv, risRnd, targetPdf, 1.0f / blendedSourcePdf);
            if (selected)
            {
                localSample = candidateSample;
            }
        }
    }
    FinalizeResampling(localReservoir, 1.0, numMisSamples);
    localReservoir.M = 1;

    // Sun
    DIReservoir sunLightReservoir = EmptyDIReservoir();
    LightSample sunLightSample = LightSample{};
    for (unsigned int i = 0; i < numSunLightSamples; i++)
    {
        float sourcePdf;
        int sampledSunIdx = sysParam.sunAliasTable->sample(rand(sysParam, randIdx), sourcePdf);

        // Create candidate sample using the helper function.
        LightSample candidateSample = createSunLightSample(sampledSunIdx);

        // Recompute UV for stream sampling.
        Int2 sunIdx(sampledSunIdx % sysParam.sunRes.x, sampledSunIdx / sysParam.sunRes.x);
        Float2 uv((sunIdx.x + 0.5f) / float(sysParam.sunRes.x), (sunIdx.y + 0.5f) / float(sysParam.sunRes.y));

        float blendedSourcePdf = LightBrdfMisWeight(surface, candidateSample, sourcePdf, sunLightMisWeight, true, brdfMisWeight, brdfCutoff);
        float targetPdf = GetLightSampleTargetPdfForSurface(candidateSample, surface);
        float risRnd = rand(sysParam, randIdx);

        bool selected = StreamSample(sunLightReservoir, SunLightIndex, uv, risRnd, targetPdf, 1.0f / blendedSourcePdf);
        if (selected)
        {
            sunLightSample = candidateSample;
        }
    }
    FinalizeResampling(sunLightReservoir, 1.0, numMisSamples);
    sunLightReservoir.M = 1;

    // Sky
    DIReservoir skyLightReservoir = EmptyDIReservoir();
    LightSample skyLightSample = LightSample{};
    for (unsigned int i = 0; i < numSkyLightSamples; i++)
    {
        float sourcePdf;
        int sampledSkyIdx = sysParam.skyAliasTable->sample(rand16bits(sysParam, randIdx), sourcePdf);

        // Create candidate sample using the helper function.
        LightSample candidateSample = createSkyLightSample(sampledSkyIdx);

        // Recompute UV for stream sampling.
        Int2 skyIdx(sampledSkyIdx % sysParam.skyRes.x, sampledSkyIdx / sysParam.skyRes.x);
        Float2 uv((skyIdx.x + 0.5f) / float(sysParam.skyRes.x), (skyIdx.y + 0.5f) / float(sysParam.skyRes.y));

        float blendedSourcePdf = LightBrdfMisWeight(surface, candidateSample, sourcePdf, skyLightMisWeight, true, brdfMisWeight, brdfCutoff);
        float targetPdf = GetLightSampleTargetPdfForSurface(candidateSample, surface);
        float risRnd = rand(sysParam, randIdx);

        bool selected = StreamSample(skyLightReservoir, SkyLightIndex, uv, risRnd, targetPdf, 1.0f / blendedSourcePdf);
        if (selected)
        {
            skyLightSample = candidateSample;
        }
    }
    FinalizeResampling(skyLightReservoir, 1.0, numMisSamples);
    skyLightReservoir.M = 1;

    // BSDF sample
    DIReservoir brdfReservoir = EmptyDIReservoir();
    LightSample brdfSample = LightSample{};
    for (unsigned int i = 0; i < numBrdfSamples; ++i)
    {
        float lightSourcePdf = 0.0f;
        Float3 sampleDir;
        unsigned int lightIndex = InvalidLightIndex;
        Float2 uv = Float2(0, 0);
        LightSample candidateSample = LightSample{};

        float brdfPdf;
        bool isTransmissiveEvent = false;
        if (GetSurfaceBrdfSample(surface, randIdx, sampleDir, brdfPdf, isTransmissiveEvent))
        {
            float maxDistance = BrdfMaxDistanceFromPdf(brdfCutoff, brdfPdf);

            ShadowRayData shadowRayData;
            shadowRayData.lightIdx = InvalidLightIndex;
            UInt2 payload = splitPointer(&shadowRayData);

            Float3 shadowRayOrig = surface.isThinfilm ? (dot(sampleDir, surface.state.normal) > 0.0f ? frontPos : backPos) : frontPos;
            optixTraverse(sysParam.topObject,
                          (float3)shadowRayOrig,
                          (float3)sampleDir,
                          0.0f,        // tmin
                          maxDistance, // tmax
                          0.0f,
                          OptixVisibilityMask(0xFF),
                          OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
                          1, 2, 1,
                          payload.x, payload.y);
            optixInvoke(payload.x, payload.y);

            if (shadowRayData.lightIdx == InvalidLightIndex)
            {
                lightIndex = InvalidLightIndex;
            }
            else if (shadowRayData.lightIdx == SkyLightIndex)
            {
                const float sunAngle = 0.51f; // angular diagram in degrees
                const float sunAngleCosThetaMax = cosf(sunAngle * M_PI / 180.0f / 2.0f);
                bool hitDisk = EqualAreaMapCone(uv, sysParam.sunDir, sampleDir, sunAngleCosThetaMax);
                if (hitDisk)
                {
                    lightIndex = SunLightIndex;

                    // Compute UV from sampleDir for sun.
                    Int2 sunIdx((int)(uv.x * sysParam.sunRes.x - 0.5f), (int)(uv.y * sysParam.sunRes.y - 0.5f));
                    if (sunIdx.x >= sysParam.sunRes.x)
                    {
                        sunIdx.x %= sysParam.sunRes.x;
                    }
                    if (sunIdx.x < 0)
                    {
                        sunIdx.x = sysParam.sunRes.x - ((-sunIdx.x) % sysParam.sunRes.x);
                    }

                    sunIdx.y = clampi(sunIdx.y, 0, sysParam.sunRes.y - 1);
                    int sampledSunIdx = sunIdx.y * sysParam.sunRes.x + sunIdx.x;

                    candidateSample = createSunLightSample(sampledSunIdx);
                    candidateSample.position = sampleDir; // override with the BSDF sample direction

                    lightSourcePdf = sysParam.sunAliasTable->PMF(sampledSunIdx);
                }
                else
                {
                    lightIndex = SkyLightIndex;
                    uv = EqualAreaSphereMap(sampleDir);

                    Int2 skyIdx((int)(uv.x * sysParam.skyRes.x - 0.5f), (int)(uv.y * sysParam.skyRes.y - 0.5f));
                    clamp2i(skyIdx, Int2(0), sysParam.skyRes - 1);

                    int sampledSkyIdx = skyIdx.y * sysParam.skyRes.x + skyIdx.x;

                    candidateSample = createSkyLightSample(sampledSkyIdx);
                    candidateSample.position = sampleDir; // override with the BSDF sample direction

                    lightSourcePdf = sysParam.skyAliasTable->PMF(sampledSkyIdx);
                }
            }
            else if (numLocalLightSamples > 0)
            {
                lightIndex = shadowRayData.lightIdx;

                // Boundary check for light index
                if (lightIndex >= sysParam.numLights)
                {
                    lightIndex = InvalidLightIndex;
                }
                else
                {
                    LightInfo lightInfo = sysParam.lights[lightIndex];

                    TriangleLight triLight = TriangleLight::Create(lightInfo);

                    uv = InverseTriangleSample(shadowRayData.bary);
                    candidateSample = triLight.calcSample(uv, surface.pos);

                    if (brdfCutoff > 0.0f)
                    {
                        float lightDistance = length(candidateSample.position - surface.pos);

                        float maxDistance = BrdfMaxDistanceFromPdf(brdfCutoff, brdfPdf);
                        if (lightDistance > maxDistance)
                        {
                            lightIndex = InvalidLightIndex;
                        }
                    }

                    if (lightIndex != InvalidLightIndex)
                    {
                        lightSourcePdf = sysParam.lightAliasTable->PMF(lightIndex);
                    }
                }
            }
        }

        if (lightSourcePdf == 0.0f)
        {
            continue;
        }

        float targetPdf = GetLightSampleTargetPdfForSurface(candidateSample, surface);

        bool isEnvMapSample = lightIndex == SkyLightIndex || lightIndex == SunLightIndex;
        float misWeight = (lightIndex == SkyLightIndex) ? skyLightMisWeight : ((lightIndex == SunLightIndex) ? sunLightMisWeight : localLightMisWeight);

        float blendedSourcePdf = LightBrdfMisWeight(surface, candidateSample, lightSourcePdf, misWeight, isEnvMapSample, brdfMisWeight, brdfCutoff);
        float risRnd = rand(sysParam, randIdx);

        bool selected = StreamSample(brdfReservoir, lightIndex, uv, risRnd, targetPdf, 1.0f / blendedSourcePdf);
        if (selected)
        {
            brdfSample = candidateSample;
        }
    }
    FinalizeResampling(brdfReservoir, 1.0f, numMisSamples);
    brdfReservoir.M = 1;

    // Merge samples
    CombineDIReservoirs(risReservoir, localReservoir, 0.5f, localReservoir.targetPdf);
    bool selectSunLight = CombineDIReservoirs(risReservoir, sunLightReservoir, rand(sysParam, randIdx), sunLightReservoir.targetPdf);
    bool selectSkyLight = CombineDIReservoirs(risReservoir, skyLightReservoir, rand(sysParam, randIdx), skyLightReservoir.targetPdf);
    bool selectBrdf = CombineDIReservoirs(risReservoir, brdfReservoir, rand(sysParam, randIdx), brdfReservoir.targetPdf);

    FinalizeResampling(risReservoir, 1.0f, 1.0f);
    risReservoir.M = 1;

    if (selectBrdf)
    {
        lightSample = brdfSample;
    }
    else if (selectSkyLight)
    {
        lightSample = skyLightSample;
    }
    else if (selectSunLight)
    {
        lightSample = sunLightSample;
    }
    else
    {
        lightSample = localSample;
    }

    // Initial visibility
    bool isLightVisible = false;
    if (lightSample.lightType != LightTypeInvalid && IsValidDIReservoir(risReservoir))
    {
        constexpr float rayLengthEpsilon = 0.01f;
        constexpr float extraRayOffset = 0.0f;
        constexpr bool usePrevBvh = false;

        Float3 sampleDir = (lightSample.lightType == LightTypeLocalTriangle) ? normalize(lightSample.position - surface.pos) : lightSample.position;
        float maxDistance = (lightSample.lightType == LightTypeLocalTriangle) ? length(lightSample.position - surface.pos) - rayLengthEpsilon - extraRayOffset : RayMax;

        UInt2 visibilityRayPayload = splitPointer(&isLightVisible);
        Float3 shadowRayOrig = surface.isThinfilm ? (dot(sampleDir, surface.state.normal) > 0.0f ? frontPos : backPos) : frontPos;

        optixTraverse(usePrevBvh ? sysParam.prevTopObject : sysParam.topObject,
                      (float3)shadowRayOrig,
                      (float3)sampleDir,
                      extraRayOffset, // tmin
                      maxDistance,    // tmax
                      0.0f,
                      OptixVisibilityMask(0xFE),
                      OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                      0, 2, 2,
                      visibilityRayPayload.x, visibilityRayPayload.y);
        optixInvoke(visibilityRayPayload.x, visibilityRayPayload.y);

        if (!isLightVisible)
        {
            // Keep M for correct resampling, remove the actual sample
            risReservoir.lightData = 0;
            risReservoir.weightSum = 0;
        }
    }

    DIReservoir restirReservoir = EmptyDIReservoir();
    if (enableReSTIR)
    {
        CombineDIReservoirs(restirReservoir, risReservoir, 0.5f, risReservoir.targetPdf);

        Float3 currentWorldPos = surface.pos;
        Float3 prevWorldPos = currentWorldPos; // TODO: movement of the world
        Float2 prevUV = sysParam.prevCamera.worldDirectionToUV(normalize(prevWorldPos - sysParam.prevCamera.pos));

        Int2 prevPixelPos = Int2(prevUV.x * sysParam.prevCamera.resolution.x, prevUV.y * sysParam.prevCamera.resolution.y);
        float expectedPrevLinearDepth = distance(prevWorldPos, sysParam.prevCamera.pos);

        constexpr unsigned int numTemporalSamples = 3;
        constexpr float mCap = 20.0f;

        Int2 temporalOffsets[numTemporalSamples];
        temporalOffsets[0] = prevPixelPos - pixelPosition;
        temporalOffsets[1] = prevPixelPos - pixelPosition + Int2(ConcentricSampleDisk(rand2(sysParam, randIdx)) * 64.0f);
        temporalOffsets[2] = Int2(ConcentricSampleDisk(rand2(sysParam, randIdx)) * 64.0f);

        unsigned int cachedResult = 0;
        int selectedLoopIdx = -1;

        for (unsigned int i = 0; i < numTemporalSamples; ++i)
        {
            Int2 offset = temporalOffsets[i];
            Int2 idx = pixelPosition + offset;
            idx = ClampSamplePositionIntoView(idx);

            Surface temporalSurface;
            if (!GetPrevSurface(temporalSurface, idx))
            {
                continue;
            }

            bool isNormalValid = dot(surface.state.normal, temporalSurface.state.geoNormal) >= 0.5f;
            bool isDepthValid = abs(expectedPrevLinearDepth - temporalSurface.depth) <= 0.1f * max(expectedPrevLinearDepth, temporalSurface.depth);
            bool isRoughValid = abs(surface.state.roughness - temporalSurface.state.roughness) <= 0.5f * max(surface.state.roughness, temporalSurface.state.roughness);

            if (!(isNormalValid && isDepthValid && isRoughValid))
            {
                continue;
            }

            cachedResult |= (1u << i);

            DIReservoir prevReservoir = LoadDIReservoir(idx);
            if (isnan(prevReservoir.weightSum) || isinf(prevReservoir.weightSum))
            {
                prevReservoir = EmptyDIReservoir();
            }

            if (prevReservoir.M > mCap)
            {
                prevReservoir.M = mCap;
            }

            LightSample candidateLightSample = {};
            float neighborWeight = 0;

            if (IsValidDIReservoir(prevReservoir))
            {
                if (!GetLightSampleFromReservoir(candidateLightSample, prevReservoir, surface, numLocalLightSamples > 0))
                {
                    prevReservoir = EmptyDIReservoir();
                }
                neighborWeight = GetLightSampleTargetPdfForSurface(candidateLightSample, surface);
            }

            if (CombineDIReservoirs(restirReservoir, prevReservoir, rand(sysParam, randIdx), neighborWeight))
            {
                lightSample = candidateLightSample;
                selectedLoopIdx = i;
            }
        }

        // Bias correction
        if (IsValidDIReservoir(restirReservoir))
        {
            float pi = restirReservoir.targetPdf;
            float piSum = restirReservoir.targetPdf * 1;

            unsigned int seletedLightID = GetDIReservoirLightIndex(restirReservoir);

            for (unsigned int i = 0; i < numTemporalSamples; ++i)
            {
                if ((cachedResult & (1u << i)) == 0)
                    continue;

                Int2 offset = temporalOffsets[i];
                Int2 idx = pixelPosition + offset;
                idx = ClampSamplePositionIntoView(idx);

                Surface temporalSurface;
                GetPrevSurface(temporalSurface, idx);

                LightSample selectedSampleAtNeighbor = {};
                GetLightSampleFromReservoir(selectedSampleAtNeighbor, restirReservoir, temporalSurface, numLocalLightSamples > 0);

                float ps = GetLightSampleTargetPdfForSurface(selectedSampleAtNeighbor, temporalSurface);

                if (ps > 0 && !(i == 0 && i == selectedLoopIdx))
                {
                    constexpr float rayLengthEpsilon = 0.01f;
                    const float extraRayOffset = 0.01f + 0.01f * temporalSurface.depth;
                    constexpr bool usePrevBvh = true;

                    Float3 sampleDir = (lightSample.lightType == LightTypeLocalTriangle) ? normalize(lightSample.position - temporalSurface.pos) : lightSample.position;
                    float maxDistance = (lightSample.lightType == LightTypeLocalTriangle) ? length(lightSample.position - temporalSurface.pos) - rayLengthEpsilon - extraRayOffset : RayMax;

                    bool isNeighborSampleVisible = false;
                    UInt2 visibilityRayPayload = splitPointer(&isNeighborSampleVisible);
                    Float3 shadowRayOrig = temporalSurface.pos;

                    optixTraverse(usePrevBvh ? sysParam.prevTopObject : sysParam.topObject,
                                  (float3)shadowRayOrig,
                                  (float3)sampleDir,
                                  extraRayOffset, // tmin
                                  maxDistance,    // tmax
                                  0.0f,
                                  OptixVisibilityMask(0xFE),
                                  OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                                  0, 2, 2,
                                  visibilityRayPayload.x, visibilityRayPayload.y);
                    optixInvoke(visibilityRayPayload.x, visibilityRayPayload.y);

                    if (!isNeighborSampleVisible)
                    {
                        ps = 0.0f;
                    }
                }

                DIReservoir prevReservoir = LoadDIReservoir(idx);
                if (isnan(prevReservoir.weightSum) || isinf(prevReservoir.weightSum))
                {
                    prevReservoir = EmptyDIReservoir();
                }

                if (prevReservoir.M > mCap)
                {
                    prevReservoir.M = mCap;
                }

                if (selectedLoopIdx == i)
                {
                    pi = ps;
                }

                piSum += ps * prevReservoir.M;
            }

            FinalizeResampling(restirReservoir, pi, piSum);

            // FinalizeResampling(restirReservoir, 1.0f, restirReservoir.M);
        }

        // Final visibility
        if (lightSample.lightType != LightTypeInvalid)
        {
            constexpr float rayLengthEpsilon = 0.01f;
            constexpr float extraRayOffset = 0.0f;
            constexpr bool usePrevBvh = false;

            Float3 sampleDir = (lightSample.lightType == LightTypeLocalTriangle) ? normalize(lightSample.position - surface.pos) : lightSample.position;
            float maxDistance = (lightSample.lightType == LightTypeLocalTriangle) ? length(lightSample.position - surface.pos) - rayLengthEpsilon - extraRayOffset : RayMax;

            isLightVisible = false;
            UInt2 visibilityRayPayload = splitPointer(&isLightVisible);
            Float3 shadowRayOrig = surface.isThinfilm ? (dot(sampleDir, surface.state.normal) > 0.0f ? frontPos : backPos) : frontPos;

            optixTraverse(usePrevBvh ? sysParam.prevTopObject : sysParam.topObject,
                          (float3)shadowRayOrig,
                          (float3)sampleDir,
                          extraRayOffset, // tmin
                          maxDistance,    // tmax
                          0.0f,
                          OptixVisibilityMask(0xFE),
                          OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                          0, 2, 2,
                          visibilityRayPayload.x, visibilityRayPayload.y);
            optixInvoke(visibilityRayPayload.x, visibilityRayPayload.y);

            if (!isLightVisible)
            {
                // Keep M for correct resampling, remove the actual sample
                restirReservoir.lightData = 0;
                restirReservoir.weightSum = 0;
            }
        }
    }

    // Shading
    DIReservoir shadingReservoir = enableReSTIR ? restirReservoir : risReservoir;

    if (lightSample.lightType != LightTypeInvalid && IsValidDIReservoir(shadingReservoir))
    {
        if (isLightVisible)
        {
            Float3 sampleDir = (lightSample.lightType == LightTypeLocalTriangle) ? normalize(lightSample.position - surface.pos) : lightSample.position;

            const Float3 albedo = skipAlbedoInShadowRayContribution ? Float3(1.0f) : state.albedo;
            Float3 bsdf;
            float pdf;

            DisneyBSDFEvaluate(state.normal, state.geoNormal, sampleDir, state.wo, albedo, state.metallic, state.translucency, state.roughness, bsdf, pdf);

            float cosTheta = fmaxf(0.0f, dot(sampleDir, state.normal));

            Float3 shadowRayRadiance = bsdf * cosTheta * lightSample.radiance * GetDIReservoirInvPdf(shadingReservoir) / lightSample.solidAnglePdf;

            rayData->radiance += shadowRayRadiance;
        }
    }

    // Store the reservoir
    if (enableReSTIR)
    {
        StoreDIReservoir(restirReservoir, pixelPosition);
    }
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

    if (parameters.isEmissive)
    {
        int idx = -1;
        {
            int left = 0;
            int right = sysParam.instanceLightMappingSize - 1;

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