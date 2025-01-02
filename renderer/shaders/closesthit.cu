#include "SystemParameter.h"
#include "OptixShaderCommon.h"
#include "ShaderDebugUtils.h"
#include "Sampler.h"

namespace jazzfusion
{

    extern "C" __constant__ SystemParameter sysParam;

    // Get the 3x4 object to world transform and its inverse from a two-level hierarchy.
    // Arguments Float4* objectToWorld, Float4* worldToObject shortened for smaller code.
    __forceinline__ __device__ void getTransforms(Float4 *mW, Float4 *mO)
    {
        OptixTraversableHandle handle = optixGetTransformListHandle(0);

        const Float4 *tW = reinterpret_cast<const Float4 *>(optixGetInstanceTransformFromHandle(handle));
        const Float4 *tO = reinterpret_cast<const Float4 *>(optixGetInstanceInverseTransformFromHandle(handle));

        mW[0] = tW[0];
        mW[1] = tW[1];
        mW[2] = tW[2];

        mO[0] = tO[0];
        mO[1] = tO[1];
        mO[2] = tO[2];
    }

    // Functions to get the individual transforms in case only one of them is needed.

    __forceinline__ __device__ void getTransformObjectToWorld(Float4 *mW)
    {
        OptixTraversableHandle handle = optixGetTransformListHandle(0);

        const Float4 *tW = reinterpret_cast<const Float4 *>(optixGetInstanceTransformFromHandle(handle));

        mW[0] = tW[0];
        mW[1] = tW[1];
        mW[2] = tW[2];
    }

    __forceinline__ __device__ void getTransformWorldToObject(Float4 *mO)
    {
        OptixTraversableHandle handle = optixGetTransformListHandle(0);

        const Float4 *tO = reinterpret_cast<const Float4 *>(optixGetInstanceInverseTransformFromHandle(handle));

        mO[0] = tO[0];
        mO[1] = tO[1];
        mO[2] = tO[2];
    }

    // Matrix3x4 * point. v.w == 1.0f
    __forceinline__ __device__ Float3 transformPoint(const Float4 *m, Float3 const &v)
    {
        Float3 r;

        r.x = m[0].x * v.x + m[0].y * v.y + m[0].z * v.z + m[0].w;
        r.y = m[1].x * v.x + m[1].y * v.y + m[1].z * v.z + m[1].w;
        r.z = m[2].x * v.x + m[2].y * v.y + m[2].z * v.z + m[2].w;

        return r;
    }

    // Matrix3x4 * vector. v.w == 0.0f
    __forceinline__ __device__ Float3 transformVector(const Float4 *m, Float3 const &v)
    {
        Float3 r;

        r.x = m[0].x * v.x + m[0].y * v.y + m[0].z * v.z;
        r.y = m[1].x * v.x + m[1].y * v.y + m[1].z * v.z;
        r.z = m[2].x * v.x + m[2].y * v.y + m[2].z * v.z;

        return r;
    }

    // InverseMatrix3x4^T * normal. v.w == 0.0f
    // Get the inverse matrix as input and applies it as inverse transpose.
    __forceinline__ __device__ Float3 transformNormal(const Float4 *m, Float3 const &v)
    {
        Float3 r;

        r.x = m[0].x * v.x + m[1].x * v.y + m[2].x * v.z;
        r.y = m[0].y * v.x + m[1].y * v.y + m[2].y * v.z;
        r.z = m[0].z * v.x + m[1].z * v.y + m[2].z * v.z;

        return r;
    }

    extern "C" __global__ void __closesthit__radiance()
    {
        PerRayData *rayData = mergePointer(optixGetPayload_0(), optixGetPayload_1());

        rayData->distance = optixGetRayTmax();
        rayData->pos = rayData->pos + rayData->wi * rayData->distance;

        // if rayData->pos is near the line segment highlightPoint[0] to highlightPoint[1], 1 to 2, 2 to 3 and 3 to 0, then set rayData->radiance to 0 and return

        GeometryInstanceData *instanceData = reinterpret_cast<GeometryInstanceData *>(optixGetSbtDataPointer());
        const MaterialParameter &parameters = sysParam.materialParameters[instanceData->materialIndex]; // Use a const reference, not all BSDFs need all values.
        int materialId = parameters.indexBSDF;

        if (rayData->flags & FLAG_SHADOW)
        {
            rayData->flags |= FLAG_SHADOW_HIT;

            if (materialId == INDEX_BSDF_SPECULAR_REFLECTION_TRANSMISSION)
            {
                rayData->flags |= FLAG_SHADOW_GLASS_HIT;
                rayData->absorption_ior.xyz = parameters.absorption;
            }

            return;
        }

        // UI Box
        if (rayData->depth == 0)
        {
            Float3 highlightPoint[4];
            highlightPoint[0] = sysParam.edgeToHighlight[0];
            highlightPoint[1] = sysParam.edgeToHighlight[1];
            highlightPoint[2] = sysParam.edgeToHighlight[2];
            highlightPoint[3] = sysParam.edgeToHighlight[3];

            // Check each line segment: (0->1), (1->2), (2->3), (3->0)
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

        const unsigned int thePrimtiveIndex = optixGetPrimitiveIndex();

        const Int3 tri = instanceData->indices[thePrimtiveIndex];

        const VertexAttributes &va0 = instanceData->attributes[tri.x];
        const VertexAttributes &va1 = instanceData->attributes[tri.y];
        const VertexAttributes &va2 = instanceData->attributes[tri.z];

        const Float2 theBarycentrics = Float2(optixGetTriangleBarycentrics()); // beta and gamma
        const float alpha = 1.0f - theBarycentrics.x - theBarycentrics.y;

        const Float3 ng = cross(va1.vertex - va0.vertex, va2.vertex - va0.vertex);
        // const Float3 ns = va0.normal * alpha + va1.normal * theBarycentrics.x + va2.normal * theBarycentrics.y;

        MaterialState state;
        // state.texcoord = va0.texcoord * alpha + va1.texcoord * theBarycentrics.x + va2.texcoord * theBarycentrics.y;

        Float4 worldToObject[3];
        getTransformWorldToObject(worldToObject);
        // state.normalGeo = normalize(transformNormal(worldToObject, ng));
        // state.normal = normalize(transformNormal(worldToObject, ns));
        state.geometricNormal = normalize(transformNormal(worldToObject, ng));

        state.wo = rayData->wo;

        // rayData->totalDistance += rayData->distance;

        if (abs(state.geometricNormal.x) > 0.9f)
        {
            state.texcoord.x = fract(rayData->pos.y);
            state.texcoord.y = fract(rayData->pos.z);
        }
        else if (abs(state.geometricNormal.y) > 0.9f)
        {
            state.texcoord.x = fract(rayData->pos.x);
            state.texcoord.y = fract(rayData->pos.z);
        }
        else if (abs(state.geometricNormal.z) > 0.9f)
        {
            state.texcoord.x = fract(rayData->pos.y);
            state.texcoord.y = fract(rayData->pos.x);
        }

        // Ray cone spread
        rayData->rayConeWidth += rayData->rayConeSpread * rayData->distance;
        // rayState.rayConeSpread += surfaceRayConeSpread; // @TODO Based on the local surface curvature

        // Face forward
        rayData->flags |= (dot(rayData->wo, state.geometricNormal) >= 0.0f) ? FLAG_FRONTFACE : 0;
        if ((rayData->flags & FLAG_FRONTFACE) == 0)
        {
            state.geometricNormal = -state.geometricNormal;
        }

        rayData->radiance = Float3(0.0f);
        rayData->f_over_pdf = Float3(0.0f);
        rayData->pdf = 0.0f;

        Float3 albedo = parameters.albedo; // PERF Copy only this locally to be able to modulate it with the optional texture.

        float texMip0Size = parameters.texSize.length();
        float lod = log2f(rayData->rayConeWidth / max(dot(state.geometricNormal, rayData->wo), 0.01f) * parameters.uvScale * 2.0f * texMip0Size) - 3.0f;
        // lod = 0.0f;

        // state.texcoord *= 2.0f;

        if (parameters.textureAlbedo != 0)
        {
            const Float3 texColor = Float3(tex2DLod<float4>(parameters.textureAlbedo, state.texcoord.x, state.texcoord.y, lod));
            albedo *= texColor;
        }

        state.roughness = 0.0f;
        if (parameters.textureRoughness != 0)
        {
            state.roughness = tex2DLod<float1>(parameters.textureRoughness, state.texcoord.x, state.texcoord.y, lod).x;
        }

        if (parameters.textureNormal != 0)
        {
            state.normal = Float3(tex2DLod<float4>(parameters.textureNormal, state.texcoord.x, state.texcoord.y, lod));
            alignVector(state.geometricNormal, state.normal);
            state.normal = normalize(state.normal);
        }

        rayData->normal = state.normal;
        rayData->roughness = state.roughness;

        rayData->flags = rayData->flags | parameters.flags; // FLAG_THINWALLED can be set directly from the material parameters.

        const bool isDiffuse = materialId >= NUM_SPECULAR_BSDF;

        const int indexBsdfSample = NUM_LIGHT_TYPES + materialId;

        Float3 surfWi;
        Float3 surfBsdfOverPdf;
        float surfSampleSurfPdf;

        optixDirectCall<void, MaterialParameter const &, MaterialState const &, PerRayData *, Float3 &, Float3 &, float &>(indexBsdfSample, parameters, state, rayData, surfWi, surfBsdfOverPdf, surfSampleSurfPdf);

        if (rayData->depth == 0)
        {
            rayData->albedo = albedo;
            albedo = Float3(1.0f);
        }
        else
        {
            surfBsdfOverPdf *= albedo;
        }

        if (isDiffuse)
        {
            bool shadowRayOnly = false;
            if (rayData->flags & FLAG_DIFFUSED)
            {
                shadowRayOnly = true;
            }
            rayData->flags |= FLAG_DIFFUSED;

            // Env light sample
            LightSample lightSample;
            {
                // PERFORMANCE OPTIMIZATION: avoiding integer division
                // const Int2& skyRes = sysParam.skyRes;
                // const Int2& sunRes = sysParam.sunRes;
                const Int2 skyRes(512, 256);
                const Int2 sunRes(32, 32);

                const float *skyCdf = sysParam.skyCdf;
                const float *sunCdf = sysParam.sunCdf;

                const int skySize = skyRes.x * skyRes.y;
                const int sunSize = sunRes.x * sunRes.y;
                const float sunAngle = 0.51f; // angular diagram in degrees
                const float sunAngleCosThetaMax = cosf(sunAngle * M_PI / 180.0f / 2.0f);

                // The accumulated all sky luminance
                const float maxSkyCdf = skyCdf[skySize - 1];

                // The accumulated all sun luminance
                const float maxSunCdf = sunCdf[sunSize - 1];

                const float totalSkyLum = maxSkyCdf * TWO_PI / skySize; // Jacobian of the hemisphere mapping
                const float totalSunLum = maxSunCdf * TWO_PI * (1.0f - sunAngleCosThetaMax) / sunSize;

                // Sample sky or sun pdf
                const float sampleSkyVsSun = totalSkyLum / (totalSkyLum + totalSunLum);

                if (sampleSkyVsSun > rayData->rand(sysParam))
                {
                    // Binary search in range 0 to size-2, since we want result+1 to be the index, we'll need to subtract result for calculating PDF
                    const int sampledSkyIdx = BinarySearch(skyCdf, 0, skySize - 2, rayData->rand(sysParam) * maxSkyCdf) + 1;

                    // Subtract neighbor CDF to get PDF, divided by max CDF to get the probability
                    float sampledSkyPdf = (skyCdf[sampledSkyIdx] - skyCdf[sampledSkyIdx - 1]) / maxSkyCdf;

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
                    lightSample.pdf = sampledSkyPdf;
                    lightSample.emission = skyEmission;

                    // Set light index for shadow ray rejection
                    // lightIdx = ENV_LIGHT_ID;
                }
                else // Choose to sample sun
                {
                    // Binary search in range 0 to size-2, since we want result+1 to be the index, we'll need to subtract result for calculating PDF
                    const int sampledSunIdx = BinarySearch(sunCdf, 0, sunSize - 2, rayData->rand(sysParam) * maxSunCdf) + 1;

                    // Subtract neighbor CDF to get PDF, divided by max CDF to get the probability
                    float sampledSunPdf = (sunCdf[sampledSunIdx] - sunCdf[sampledSunIdx - 1]) / maxSunCdf;

                    // Each tile has area = coneAnglularArea / resolution, pdf = 1/area = resolution / (TWO_PI * (1.0f - cosThetaMax))
                    sampledSunPdf = sampledSunPdf * sunSize / (TWO_PI * (1.0f - sunAngleCosThetaMax));

                    // Index to 2D coordinates
                    Int2 sunIdx(sampledSunIdx % sunRes.x, sampledSunIdx / sunRes.x);
                    float u = (sunIdx.x + 0.5f) / sunRes.x;
                    float v = (sunIdx.y + 0.5f) / sunRes.y;

                    // Hemisphere projection
                    Float3 rayDir = EqualAreaMapCone(sysParam.sunDir, u, v, sunAngleCosThetaMax);

                    // Load sky buffer
                    Float3 sunEmission = Load2DHalf4(sysParam.sunBuffer, sunIdx).xyz;

                    // Set light sample direction and PDF
                    lightSample.direction = rayDir;
                    lightSample.pdf = sampledSunPdf; // * (1.0f - chooseSampleSkyVsSun);
                    lightSample.emission = sunEmission;

                    // Set light index for shadow ray rejection
                    // lightIdx = ENV_LIGHT_ID;

                    // Debug
                    // DEBUG_PRINT(maxSkyCdf);
                    // DEBUG_PRINT(maxSunCdf);
                    // DEBUG_PRINT(sampleSkyVsSunPdf);
                    // DEBUG_PRINT(sampledSunIdx);
                    // DEBUG_PRINT(sampledSunPdf);
                    // DEBUG_PRINT(u);
                    // DEBUG_PRINT(v);
                    // DEBUG_PRINT(rayDir);
                }
            }

            // const int numLights = sysParam.numLights;
            // const Float2 randNum = rayData->rand2(sysParam);
            // const int indexLight = 0; //(1 < numLights) ? clampi(static_cast<int>(floorf(rayData->rand(sysParam) * numLights)), 0, numLights - 1) : 0;
            // LightDefinition const& light = sysParam.lightDefinitions[indexLight];
            // const int indexCallable = light.type;
            // LightSample lightSample = optixDirectCall<LightSample, LightDefinition const&, const Float3, const Float2>(indexCallable, light, rayData->pos, randNum);
            // LightSample lightSample = LightEnvSphereSample(light, rayData->pos, randNum);

            /// Directional light test
            // if (0)
            // {
            //     lightSample.pdf = 1.0f;
            //     lightSample.direction = normalize(Float3(1, 1, 0));
            //     lightSample.emission = Float3(1.0f);
            // }

            // Float4 surfSampleEmissionPdf = LightEnvSphereEval(light, rayData->pos, surfWi);
            // Float3 surfSampleEmission = surfSampleEmissionPdf.xyz;
            // float surfSampleLightDistPdf = surfSampleEmissionPdf.w;

            // float misWeightSurfSample = powerHeuristic(surfSampleSurfPdf, surfSampleLightDistPdf);
            // shadowRayOnly = shadowRayOnly || (misWeightSurfSample < 0.1f);
            // rayData->lightEmission = surfSampleEmission * misWeightSurfSample;

            float lightSampleLightDistPdf = lightSample.pdf;

            if (0.0f < lightSampleLightDistPdf) // Valid light sample, verify light distribution
            {
                const int indexBsdfEval = indexBsdfSample + 1;
                const Float4 lightSampleSurfDistBsdfPdf = optixDirectCall<Float4, MaterialParameter const &, MaterialState const &, PerRayData const *, const Float3>(indexBsdfEval, parameters, state, rayData, lightSample.direction);
                Float3 lightSampleSurfDistBsdf = lightSampleSurfDistBsdfPdf.xyz;
                float lightSampleSurfDistPdf = lightSampleSurfDistBsdfPdf.w;

                if (0.0f < lightSampleSurfDistPdf) // Valid light sample, verify surface distribution
                {
                    // float chooseLightSampleWeight = misWeightLightSample / (misWeightLightSample + misWeightSurfSample);
                    // if ((rayData->rand(sysParam) < chooseLightSampleWeight) || shadowRayOnly)

                    rayData->flags |= FLAG_SHADOW;

                    Float3 originalPos = rayData->pos;
                    float originalDistance = rayData->distance;
                    Float3 originalAbsorption = rayData->absorption_ior.xyz;

                    rayData->wi = lightSample.direction;
                    bool inVolume = rayData->flags & FLAG_VOLUME;
                    Float3 glassThroughput = Float3(1.0f);

                    // For glass, go *straight* through the surface and volumn to calculate the direct light contribution
                    // Very wrong caustics but cheap enough and look decent
                    for (int shadowRayIter = 0; shadowRayIter < 5; ++shadowRayIter)
                    {
                        UInt2 payload = splitPointer(rayData);

                        // if (OPTIX_CENTER_PIXEL())
                        // {
                        //     OPTIX_DEBUG_PRINT(Float4(glassThroughput, shadowRayIter));
                        // }

                        rayData->flags &= ~FLAG_SHADOW_HIT;
                        rayData->flags &= ~FLAG_SHADOW_GLASS_HIT;

                        optixTrace(sysParam.topObject,
                                   (float3)rayData->pos, (float3)rayData->wi,
                                   sysParam.sceneEpsilon, RayMax, 0.0f, // tmin, tmax, time
                                   OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                                   0, 1, 0,
                                   payload.x, payload.y);

                        if (!(rayData->flags & FLAG_SHADOW_HIT))
                        {
                            float misWeightLightSample = powerHeuristic(lightSampleLightDistPdf, lightSampleSurfDistPdf);
                            const float cosTheta = fmaxf(0.0f, dot(lightSample.direction, state.normal));
                            Float3 shadowRayBsdfOverPdf = lightSampleSurfDistBsdf * cosTheta / lightSampleLightDistPdf;
                            Float3 shadowRayRadiance = lightSample.emission * misWeightLightSample * shadowRayBsdfOverPdf * glassThroughput * albedo;
                            rayData->radiance += shadowRayRadiance;

                            // if (OPTIX_CENTER_PIXEL() && shadowRayIter == 2)
                            // {
                            //     OPTIX_DEBUG_PRINT(Float4(shadowRayRadiance, 0));
                            // }

                            break;
                        }
                        else
                        {
                            if (rayData->flags & FLAG_SHADOW_GLASS_HIT)
                            {
                                if (inVolume)
                                {
                                    inVolume = false;
                                    glassThroughput *= exp3f(-rayData->distance * rayData->absorption_ior.xyz);
                                }
                                else
                                {
                                    inVolume = true;
                                }
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

                    if (shadowRayOnly)
                    {
                        rayData->flags |= FLAG_TERMINATE;
                        return;
                    }
                }
            }
        }

        rayData->wi = surfWi;
        rayData->f_over_pdf = surfBsdfOverPdf;
        rayData->pdf = surfSampleSurfPdf;
    }
}