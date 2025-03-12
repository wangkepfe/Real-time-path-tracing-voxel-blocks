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
    state.normal = lerp3f(state.geometricNormal, state.normal, 0.5f);

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
    rayData->geoNormal = state.geometricNormal;
    rayData->roughness = state.roughness;
    rayData->isCurrentBounceDiffuse = isDiffuse;
    rayData->hitThinfilm = isThinfilm;

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
    bool skipAlbedoInShadowRayContribution = false;
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
        skipAlbedoInShadowRayContribution = true;
    }

    if (isDiffuse)
    {
        Int2 pixelPosition = Int2(optixGetLaunchIndex());
        int &randIdx = rayData->randIdx;

        const bool enableRIS = true;
        const bool enableReSTIR = true && rayData->depth == 0;

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

        LightSample lightSample = {};
        DIReservoir risReservoir = EmptyDIReservoir();

        if (enableRIS)
        {
            const int numLocalLightSamples = 8;
            const int numSunLightSamples = 1;
            const int numSkyLightSamples = 1;
            const int numBrdfSamples = 1;

            const int numMisSamples = numLocalLightSamples + numSunLightSamples + numSkyLightSamples + numBrdfSamples;

            const float localLightMisWeight = float(numLocalLightSamples) / numMisSamples;
            const float sunLightMisWeight = float(numSunLightSamples) / numMisSamples;
            const float skyLightMisWeight = float(numSkyLightSamples) / numMisSamples;
            const float brdfMisWeight = float(numBrdfSamples) / numMisSamples;

            const float brdfCutoff = 0.0001f;

            // Local light
            DIReservoir localReservoir = EmptyDIReservoir();
            LightSample localSample = LightSample{};
            for (unsigned int i = 0; i < numLocalLightSamples; i++)
            {
                float sourcePdf;
                int lightIndex = sysParam.lightAliasTable->sample(rand(sysParam, randIdx), sourcePdf);
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
                if (GetSurfaceBrdfSample(surface, rand3(sysParam, randIdx), sampleDir, brdfPdf))
                {
                    float maxDistance = BrdfMaxDistanceFromPdf(brdfCutoff, brdfPdf);

                    ShadowRayData rayData;
                    rayData.lightIdx = InvalidLightIndex;
                    UInt2 payload = splitPointer(&rayData);

                    Float3 shadowRayOrig = surface.thinfilm ? (dot(sampleDir, surface.normal) > 0.0f ? surface.pos : surface.backfacePos) : surface.pos;
                    optixTrace(sysParam.topObject,
                               (float3)shadowRayOrig, (float3)sampleDir,
                               0.0f, maxDistance, 0.0f, // tmin, tmax, time
                               OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                               1, 2, 1,
                               payload.x, payload.y);

                    if (rayData.lightIdx == InvalidLightIndex)
                    {
                        lightIndex = InvalidLightIndex;
                    }
                    else if (rayData.lightIdx == SkyLightIndex)
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
                            uv = EqualAreaMap(sampleDir);

                            Int2 skyIdx((int)(uv.x * sysParam.skyRes.x - 0.5f), (int)(uv.y * sysParam.skyRes.y - 0.5f));
                            clamp2i(skyIdx, Int2(0), sysParam.skyRes - 1);

                            int sampledSkyIdx = skyIdx.y * sysParam.skyRes.x + skyIdx.x;

                            candidateSample = createSkyLightSample(sampledSkyIdx);
                            candidateSample.position = sampleDir; // override with the BSDF sample direction

                            lightSourcePdf = sysParam.skyAliasTable->PMF(sampledSkyIdx);
                        }
                    }
                    else
                    {
                        lightIndex = rayData.lightIdx;

                        LightInfo lightInfo = sysParam.lights[lightIndex];

                        TriangleLight triLight = TriangleLight::Create(lightInfo);

                        uv = InverseTriangleSample(rayData.bary);
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
        }

        // Initial visibility
        bool isLightVisible = false;
        if (lightSample.lightType != LightTypeInvalid && IsValidDIReservoir(risReservoir))
        {
            isLightVisible = TraceVisibilityShadowRay(lightSample, surface);

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

            // prevUV = Float2(pixelPosition.x / sysParam.camera.resolution.x, pixelPosition.y / sysParam.camera.resolution.y);
            Int2 prevPixelPos = Int2(prevUV.x * sysParam.prevCamera.resolution.x, prevUV.y * sysParam.prevCamera.resolution.y);
            // rayData->radiance += Float3(max(prevPixelPos.x - pixelPosition.x, 0), 0, 0);
            float expectedPrevLinearDepth = distance(prevWorldPos, sysParam.prevCamera.pos);

            constexpr unsigned int numTemporalSamples = 2;
            constexpr unsigned int numSpatialSamples = 2;
            constexpr unsigned int numSamples = numTemporalSamples + numSpatialSamples;
            constexpr float mCap = 20.0f;
            constexpr float spatialRadius = 16.0f;

            Int2 temporalOffsets[numTemporalSamples];
            temporalOffsets[0] = Int2(0);
            temporalOffsets[1] = Int2((int)(rand(sysParam, randIdx) * 3.0f) - 1, (int)(rand(sysParam, randIdx) * 3.0f) - 1);

            Int2 spatialOffsets[numSpatialSamples == 0 ? 1 : numSpatialSamples];
            for (int i = 0; i < numSpatialSamples; ++i)
            {
                spatialOffsets[i] = Int2((int)((rand(sysParam, randIdx) - 0.5f) * spatialRadius), (int)((rand(sysParam, randIdx) - 0.5f) * spatialRadius));
            }

            unsigned int cachedResult = 0;
            int selectedLoopIdx = -1;

            for (int i = 0; i < numSamples; ++i)
            {
                Int2 offset;
                if (i < numTemporalSamples)
                {
                    offset = temporalOffsets[i];
                }
                else
                {
                    offset = spatialOffsets[i - numTemporalSamples];
                }
                Int2 idx = prevPixelPos + offset;
                idx = ClampSamplePositionIntoView(idx);

                Surface temporalSurface;
                if (!GetPrevSurface(temporalSurface, idx))
                    continue;

                bool isNormalValid = dot(surface.normal, temporalSurface.normal) >= 0.5f;
                bool isGeoNormalValid = dot(surface.geoNormal, temporalSurface.geoNormal) >= 0.9f;
                bool isDepthValid = abs(expectedPrevLinearDepth - temporalSurface.depth) <= 0.1f * max(expectedPrevLinearDepth, temporalSurface.depth);
                bool isRoughValid = abs(surface.roughness - temporalSurface.roughness) <= 0.5f * max(surface.roughness, temporalSurface.roughness);
                bool isAlbedoValid = abs(luminance(surface.albedo) - luminance(temporalSurface.albedo)) < 0.25f;

                if (!(isNormalValid && isGeoNormalValid && isDepthValid && isRoughValid && isAlbedoValid))
                    continue;

                cachedResult |= (1u << unsigned int(i));

                DIReservoir prevReservoir = LoadDIReservoir(idx);
                if (isnan(prevReservoir.weightSum) || isinf(prevReservoir.weightSum))
                {
                    prevReservoir = EmptyDIReservoir();
                }

                if (prevReservoir.M > mCap)
                {
                    prevReservoir.M = mCap;
                }

                float neighborWeight = 0;
                LightSample candidateLightSample = {};
                if (IsValidDIReservoir(prevReservoir))
                {
                    candidateLightSample = GetLightSampleFromReservoir(prevReservoir, surface);
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
                // float pi = restirReservoir.targetPdf;
                // float piSum = restirReservoir.targetPdf;

                // unsigned int seletedLightID = GetDIReservoirLightIndex(restirReservoir);

                // for (int i = 0; i < numSamples; ++i)
                // {
                //     if ((cachedResult & (1u << unsigned int(i))) == 0)
                //         continue;

                //     Int2 offset;
                //     if (i < numTemporalSamples)
                //     {
                //         offset = temporalOffsets[i];
                //     }
                //     else
                //     {
                //         offset = spatialOffsets[i - numTemporalSamples];
                //     }
                //     Int2 idx = prevPixelPos + offset;
                //     idx = ClampSamplePositionIntoView(idx);

                //     Surface temporalSurface;
                //     GetPrevSurface(temporalSurface, idx);

                //     LightSample selectedSampleAtNeighbor = GetLightSampleFromReservoir(restirReservoir, temporalSurface);

                //     float ps = GetLightSampleTargetPdfForSurface(selectedSampleAtNeighbor, temporalSurface);

                //     // if (selectedLoopIdx != i || i >= numTemporalSamples)
                //     // {
                //     //     if (!TraceVisibilityShadowRay(selectedSampleAtNeighbor, temporalSurface, 0.01f))
                //     //     {
                //     //         ps = 0.0f;
                //     //     }
                //     // }

                //     DIReservoir prevReservoir = LoadDIReservoir(idx);
                //     if (isnan(prevReservoir.weightSum) || isinf(prevReservoir.weightSum))
                //     {
                //         prevReservoir = EmptyDIReservoir();
                //     }

                //     if (prevReservoir.M > mCap)
                //     {
                //         prevReservoir.M = mCap;
                //     }

                //     if (selectedLoopIdx == i)
                //     {
                //         pi = ps;
                //     }

                //     piSum += ps * prevReservoir.M;
                // }

                // FinalizeResampling(restirReservoir, pi, piSum);

                FinalizeResampling(restirReservoir, 1.0f, restirReservoir.M);
            }
        }

        // Boiling filter
        // {
        //     float threadGroupWeightSum = restirReservoir.weightSum;
        //     float threadCount = 1.0f;
        //     for (int offset = warpSize / 2; offset > 0; offset /= 2)
        //     {
        //         threadGroupWeightSum += __shfl_down_sync(__activemask(), threadGroupWeightSum, offset);
        //         threadCount += __shfl_down_sync(__activemask(), threadCount, offset);
        //     }

        //     if (restirReservoir.weightSum > (threadGroupWeightSum / threadCount) * 40.0f)
        //     {
        //         restirReservoir = EmptyDIReservoir();
        //     }
        // }

        // Shading
        if (lightSample.lightType != LightTypeInvalid && IsValidDIReservoir(restirReservoir))
        {
            // Final visibility
            if (enableReSTIR)
            {
                isLightVisible = TraceVisibilityShadowRay(lightSample, surface);

                if (!isLightVisible)
                {
                    // Keep M for correct resampling, remove the actual sample
                    restirReservoir.lightData = 0;
                    restirReservoir.weightSum = 0;
                }
            }

            if (isLightVisible)
            {
                Float3 sampleDir = (lightSample.lightType == LightTypeLocalTriangle) ? normalize(lightSample.position - surface.pos) : lightSample.position;

                const int indexBsdfEval = indexBsdfSample + 1;
                const Float4 bsdfPdf = optixDirectCall<Float4, MaterialParameter const &, MaterialState const &, RayData const *, const Float3>(indexBsdfEval, parameters, state, rayData, sampleDir);
                Float3 bsdf = bsdfPdf.xyz;

                if (skipAlbedoInShadowRayContribution)
                {
                    bsdf /= state.albedo;
                }

                float cosTheta = fmaxf(0.0f, dot(sampleDir, state.normal));

                Float3 shadowRayRadiance = bsdf * cosTheta * lightSample.radiance * GetDIReservoirInvPdf(restirReservoir) / lightSample.solidAnglePdf;

                rayData->radiance += shadowRayRadiance;
            }
        }

        if (enableReSTIR)
        {
            StoreDIReservoir(restirReservoir, pixelPosition);
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