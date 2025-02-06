#include "SystemParameter.h"
#include "OptixShaderCommon.h"
#include "ShaderDebugUtils.h"
#include "Sampler.h"
#include "SelfHit.h"

namespace jazzfusion
{

    extern "C" __constant__ SystemParameter sysParam;

    extern "C" __global__ void __closesthit__radiance()
    {
        PerRayData *rayData = mergePointer(optixGetPayload_0(), optixGetPayload_1());

        rayData->distance = optixGetRayTmax();

        // if (OPTIX_CENTER_PIXEL())
        // {
        //     OPTIX_DEBUG_PRINT(rayData->distance);
        // }

        // rayData->pos = rayData->pos + rayData->wi * rayData->distance;

        GeometryInstanceData *instanceData = reinterpret_cast<GeometryInstanceData *>(optixGetSbtDataPointer());

        const unsigned int thePrimtiveIndex = optixGetPrimitiveIndex();

        const Int3 tri = instanceData->indices[thePrimtiveIndex];

        const VertexAttributes &va0 = instanceData->attributes[tri.x];
        const VertexAttributes &va1 = instanceData->attributes[tri.y];
        const VertexAttributes &va2 = instanceData->attributes[tri.z];

        const Float3 v0 = va0.vertex;
        const Float3 v1 = va1.vertex;
        const Float3 v2 = va2.vertex;

        float2 bary = optixGetTriangleBarycentrics();

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

        bool isThinfilm = materialId == INDEX_BSDF_DIFFUSE_REFLECTION_TRANSMISSION_THINFILM;

        if (isThinfilm && !hitFrontFace)
        {
            geometricNormal = -geometricNormal;
            hitFrontFace = true;

            Float3 tmp = frontPos;
            frontPos = backPos;
            backPos = tmp;
        }

        if (rayData->isShadowRay)
        {
            rayData->hasShadowRayHitAnything = true;

            if (materialId == INDEX_BSDF_SPECULAR_REFLECTION_TRANSMISSION)
            {
                rayData->hasShadowRayHitTransmissiveSurface = true;
                rayData->absorption_ior.xyz = parameters.absorption;

                if (rayData->isInsideVolume)
                {
                    rayData->pos = frontPos;
                }
                else
                {
                    rayData->pos = backPos;
                }
            }
            else if (isThinfilm)
            {
                rayData->hasShadowRayHitThinfilmSurface = true;
                rayData->absorption_ior.xyz = parameters.absorption;

                rayData->pos = backPos;
            }

            return;
        }

        if (materialId == INDEX_BSDF_EMISSIVE)
        {
            rayData->radiance = parameters.albedo;
            rayData->shouldTerminate = true;
            return;
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

        if (parameters.flags == 2) // use texture coordinates
        {
            state.texcoord = va0.texcoord * alpha + va1.texcoord * theBarycentrics.x + va2.texcoord * theBarycentrics.y;

            // if (OPTIX_CENTER_PIXEL())
            // {
            //     OPTIX_DEBUG_PRINT(state.texcoord);
            //     OPTIX_DEBUG_PRINT(va0.texcoord);
            //     OPTIX_DEBUG_PRINT(va1.texcoord);
            //     OPTIX_DEBUG_PRINT(va2.texcoord);
            //     OPTIX_DEBUG_PRINT(Float3(theBarycentrics.x, theBarycentrics.y, alpha));
            // }
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

        // Ray cone spread
        rayData->rayConeWidth += rayData->rayConeSpread * rayData->distance; // +surfaceRayConeSpread; // @TODO Based on the local surface curvature

        // if (OPTIX_CENTER_PIXEL())
        // {
        //     OPTIX_DEBUG_PRINT(Float4(rayData->pos, rayData->depth));
        //     OPTIX_DEBUG_PRINT(Float4(frontPos, rayData->depth));
        //     OPTIX_DEBUG_PRINT(Float4(backPos, rayData->depth));
        //     OPTIX_DEBUG_PRINT(Float4(rayData->wo, rayData->depth));
        //     OPTIX_DEBUG_PRINT(Float4(state.geometricNormal, rayData->depth));
        // }

        rayData->hitFrontFace = hitFrontFace;

        Float3 albedo = parameters.albedo;

        state.texcoord /= parameters.uvScale;
        float texMip0Size = parameters.texSize.length();
        float lod = log2f(rayData->rayConeWidth / max(dot(state.geometricNormal, rayData->wo), 0.2f) / parameters.uvScale * 2.0f * texMip0Size) - 3.0f;

        if (parameters.textureAlbedo != 0)
        {
            const Float3 texColor = Float3(tex2DLod<float4>(parameters.textureAlbedo, state.texcoord.x, state.texcoord.y, lod));
            albedo *= texColor;

            // if (OPTIX_CENTER_PIXEL())
            // {
            //     OPTIX_DEBUG_PRINT(state.texcoord);
            //     OPTIX_DEBUG_PRINT(texColor);
            //     OPTIX_DEBUG_PRINT(lod);
            // }
        }

        state.albedo = albedo;

        state.roughness = 0.001f;
        if (parameters.textureRoughness != 0)
        {
            state.roughness = tex2DLod<float1>(parameters.textureRoughness, state.texcoord.x, state.texcoord.y, lod).x;
        }

        if (rayData->hitFirstDiffuseSurface && state.roughness < 0.5f) // Roughness control path regulization: After the first diffuse, all BSDF increase its roughness
        {
            state.roughness *= 2.0f;
        }

        state.metallic = 0.0f;
        if (parameters.textureMetallic != 0)
        {
            state.metallic = tex2DLod<float1>(parameters.textureMetallic, state.texcoord.x, state.texcoord.y, lod).x;
        }

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

        if (parameters.flags == 1) // water
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

        bool isDiffuse = materialId >= NUM_SPECULAR_BSDF;

        rayData->isCurrentBounceDiffuse = isDiffuse;

        const int indexBsdfSample = materialId;

        Float3 surfWi;
        Float3 surfBsdfOverPdf;
        float surfSampleSurfPdf;

        optixDirectCall<void, MaterialParameter const &, MaterialState const &, PerRayData *, Float3 &, Float3 &, float &>(indexBsdfSample, parameters, state, rayData, surfWi, surfBsdfOverPdf, surfSampleSurfPdf);

        // if (OPTIX_CENTER_PIXEL())
        // {
        //     OPTIX_DEBUG_PRINT(surfWi);
        //     OPTIX_DEBUG_PRINT(surfBsdfOverPdf);
        //     OPTIX_DEBUG_PRINT(surfSampleSurfPdf);
        // }

        if (isThinfilm)
        {
            rayData->pos = rayData->isHitThinfilmTransmission ? backPos : frontPos;
        }
        else
        {
            if (rayData->hitFrontFace) // front face
            {
                if (rayData->isInsideVolume) // inside volume
                {
                    if (rayData->isHitTransmission) // trasmission
                    {
                        // wrong!
                    }
                    else // reflection
                    {
                        rayData->pos = frontPos;
                    }
                }
                else // outside volumn
                {
                    if (rayData->isHitTransmission) // trasmission
                    {
                        rayData->pos = backPos;
                    }
                    else // reflection
                    {
                        rayData->pos = frontPos;
                    }
                }
            }
            else // backface
            {
                if (rayData->isInsideVolume) // inside volume
                {
                    if (rayData->isHitTransmission) // trasmission
                    {
                        rayData->pos = frontPos;
                    }
                    else // reflection
                    {
                        rayData->pos = backPos;
                    }
                }
                else // outside volumn
                {
                    // pretty wrong situation: pass through

                    if (rayData->isHitTransmission) // trasmission
                    {
                        rayData->pos = backPos;
                    }
                    else // reflection
                    {
                        rayData->pos = frontPos;
                    }

                    rayData->f_over_pdf = Float3(1.0f);
                    rayData->pdf = 1.0f;
                    return;
                }
            }
        }

        if (!rayData->hitFirstDiffuseSurface && isDiffuse)
        {
            rayData->hitFirstDiffuseSurface = true;
            if (rayData->sampleIdx == 0)
            {
                rayData->albedo = albedo;
            }
            else
            {
                rayData->albedo = lerp3f(rayData->albedo, albedo, 1.0f / (float)(rayData->sampleIdx + 1));
            }
            albedo = Float3(1.0f);
        }
        else
        {
            surfBsdfOverPdf *= albedo;
        }

        constexpr bool enableDiffuseOptimization = true;

        if (isDiffuse)
        {
            // Diffuse after diffuse, shadow ray only
            bool shadowRayOnly = false;
            if (enableDiffuseOptimization)
            {
                if (rayData->isLastBounceDiffuse)
                {
                    shadowRayOnly = true;
                }
            }

            // Env light sample
            LightSample lightSample;
            {
                const Int2 skyRes(512, 256);
                const Int2 sunRes(32, 32);

                const AliasTable &skyAliasTable = *sysParam.skyAliasTable;
                const AliasTable &sunAliasTable = *sysParam.sunAliasTable;
                const float &accumulatedSkyLuminance = sysParam.accumulatedSkyLuminance;
                const float &accumulatedSunLuminance = sysParam.accumulatedSunLuminance;
                const int skySize = skyRes.x * skyRes.y;
                const int sunSize = sunRes.x * sunRes.y;
                const float sunAngle = 0.51f; // angular diagram in degrees
                const float sunAngleCosThetaMax = cosf(sunAngle * M_PI / 180.0f / 2.0f);

                const float totalSkyLum = accumulatedSkyLuminance * TWO_PI / skySize; // Jacobian of the hemisphere mapping
                const float totalSunLum = accumulatedSunLuminance * TWO_PI * (1.0f - sunAngleCosThetaMax) / sunSize;

                // Sample sky or sun pdf
                const float sampleSkyVsSun = totalSkyLum / (totalSkyLum + totalSunLum);

                if (sampleSkyVsSun > rayData->rand(sysParam))
                {
                    float sampledSkyPdf;
                    const int sampledSkyIdx = skyAliasTable.sample(rayData->rand(sysParam), sampledSkyPdf);

                    // Each tile has area 2Pi / resolution, pdf = 1/area = resolution / 2Pi
                    sampledSkyPdf = sampledSkyPdf * skySize / TWO_PI;

                    // Index to 2D coordinates
                    Int2 skyIdx(sampledSkyIdx % skyRes.x, sampledSkyIdx / skyRes.x);
                    float u = (skyIdx.x + 0.5f) / skyRes.x;
                    float v = (skyIdx.y + 0.5f) / skyRes.y;

                    // Hemisphere projection
                    Float3 rayDir = EqualAreaMap(u, v);

                    // Load sky buffer
                    Float3 skyEmission = Load2DFloat4(sysParam.skyBuffer, skyIdx).xyz;

                    // Set light sample direction and PDF
                    lightSample.direction = rayDir;
                    lightSample.pdf = sampledSkyPdf * sampleSkyVsSun;
                    lightSample.emission = skyEmission;
                }
                else // Choose to sample sun
                {
                    float sampledSunPdf;
                    const int sampledSunIdx = sunAliasTable.sample(rayData->rand(sysParam), sampledSunPdf);

                    // Each tile has area = coneAnglularArea / resolution, pdf = 1/area = resolution / (TWO_PI * (1.0f - cosThetaMax))
                    sampledSunPdf = sampledSunPdf * sunSize / (TWO_PI * (1.0f - sunAngleCosThetaMax));

                    // Index to 2D coordinates
                    Int2 sunIdx(sampledSunIdx % sunRes.x, sampledSunIdx / sunRes.x);
                    float u = (sunIdx.x + 0.5f) / sunRes.x;
                    float v = (sunIdx.y + 0.5f) / sunRes.y;

                    // Hemisphere projection
                    Float3 rayDir = EqualAreaMapCone(sysParam.sunDir, u, v, sunAngleCosThetaMax);

                    // Load sky buffer
                    Float3 sunEmission = Load2DFloat4(sysParam.sunBuffer, sunIdx).xyz;

                    // Set light sample direction and PDF
                    lightSample.direction = rayDir;
                    lightSample.pdf = sampledSunPdf * (1.0f - sampleSkyVsSun);
                    lightSample.emission = sunEmission;
                }
            }

            float lightSampleLightDistPdf = lightSample.pdf;

            bool isLightGeometricallyVisible;

            if (isThinfilm)
            {
                isLightGeometricallyVisible = true;
            }
            else
            {
                isLightGeometricallyVisible = dot(lightSample.direction, state.geometricNormal) > 0.0f;
            }

            if (0.0f < lightSampleLightDistPdf && isLightGeometricallyVisible) // Valid light sample, verify light distribution
            {
                const int indexBsdfEval = indexBsdfSample + 1;
                const Float4 lightSampleSurfDistBsdfPdf = optixDirectCall<Float4, MaterialParameter const &, MaterialState const &, PerRayData const *, const Float3>(indexBsdfEval, parameters, state, rayData, lightSample.direction);
                Float3 lightSampleSurfDistBsdf = lightSampleSurfDistBsdfPdf.xyz;
                float lightSampleSurfDistPdf = lightSampleSurfDistBsdfPdf.w;

                if (0.0f < lightSampleSurfDistPdf) // Valid light sample, verify surface distribution
                {
                    rayData->isShadowRay = true;

                    Float3 originalPos = rayData->pos;
                    float originalDistance = rayData->distance;
                    Float3 originalAbsorption = rayData->absorption_ior.xyz;
                    bool originalIsInsideVolume = rayData->isInsideVolume;

                    Float3 glassThroughput = Float3(1.0f);

                    if (isThinfilm)
                    {
                        bool visibleFromFrontFace = dot(lightSample.direction, state.geometricNormal) > 0.0f;
                        rayData->pos = visibleFromFrontFace ? frontPos : backPos;

                        const float cosTheta = abs(dot(lightSample.direction, state.normal));
                        glassThroughput = visibleFromFrontFace ? Float3(1.0f) : (rayData->absorption_ior.xyz * cosTheta / M_PI / 0.75f);
                    }

                    rayData->wi = lightSample.direction;

                    // For glass, go *straight* through the surface and volumn to calculate the direct light contribution
                    // Very wrong caustics but cheap enough and look decent
                    for (int shadowRayIter = 0; shadowRayIter < 5; ++shadowRayIter)
                    {
                        UInt2 payload = splitPointer(rayData);

                        rayData->hasShadowRayHitAnything = false;
                        rayData->hasShadowRayHitTransmissiveSurface = false;
                        rayData->hasShadowRayHitThinfilmSurface = false;

                        optixTrace(sysParam.topObject,
                                   (float3)rayData->pos, (float3)rayData->wi,
                                   sysParam.sceneEpsilon, RayMax, 0.0f, // tmin, tmax, time
                                   OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                                   0, 1, 0,
                                   payload.x, payload.y);

                        if (!rayData->hasShadowRayHitAnything)
                        {
                            float misWeightLightSample = powerHeuristic(lightSampleLightDistPdf, lightSampleSurfDistPdf);
                            const float cosTheta = fmaxf(0.0f, dot(lightSample.direction, state.normal));
                            Float3 shadowRayBsdfOverPdf = lightSampleSurfDistBsdf * cosTheta / lightSampleLightDistPdf;
                            Float3 shadowRayRadiance = lightSample.emission * misWeightLightSample * shadowRayBsdfOverPdf * glassThroughput * albedo;
                            rayData->radiance += shadowRayRadiance;
                            break;
                        }
                        else
                        {
                            if (rayData->hasShadowRayHitTransmissiveSurface)
                            {
                                if (rayData->isInsideVolume)
                                {
                                    rayData->isInsideVolume = false;
                                    glassThroughput *= exp3f(-rayData->distance * rayData->absorption_ior.xyz);
                                }
                                else
                                {
                                    rayData->isInsideVolume = true;
                                }
                                continue;
                            }
                            else if (rayData->hasShadowRayHitThinfilmSurface)
                            {
                                const float cosTheta = abs(dot(lightSample.direction, state.normal));
                                glassThroughput *= rayData->absorption_ior.xyz * cosTheta / M_PI / 0.75f;
                                continue;
                            }
                            else
                            {
                                break;
                            }
                        }
                    }

                    rayData->pos = originalPos;
                    rayData->distance = originalDistance;
                    rayData->absorption_ior.xyz = originalAbsorption;
                    rayData->isInsideVolume = originalIsInsideVolume;

                    rayData->isShadowRay = false;

                    if (enableDiffuseOptimization)
                    {
                        if (shadowRayOnly)
                        {
                            rayData->shouldTerminate = true;
                            return;
                        }
                    }
                }
            }
        }

        rayData->wi = surfWi;
        rayData->f_over_pdf = surfBsdfOverPdf;
        rayData->pdf = surfSampleSurfPdf;
    }
}