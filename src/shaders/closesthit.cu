#include "SystemParameter.h"
#include "OptixShaderCommon.h"
#include "ShaderDebugUtils.h"

namespace jazzfusion
{

extern "C" __constant__ SystemParameter sysParam;


// Get the 3x4 object to world transform and its inverse from a two-level hierarchy.
// Arguments Float4* objectToWorld, Float4* worldToObject shortened for smaller code.
__forceinline__ __device__ void getTransforms(Float4* mW, Float4* mO)
{
    OptixTraversableHandle handle = optixGetTransformListHandle(0);

    const Float4* tW = reinterpret_cast<const Float4*>(optixGetInstanceTransformFromHandle(handle));
    const Float4* tO = reinterpret_cast<const Float4*>(optixGetInstanceInverseTransformFromHandle(handle));

    mW[0] = tW[0];
    mW[1] = tW[1];
    mW[2] = tW[2];

    mO[0] = tO[0];
    mO[1] = tO[1];
    mO[2] = tO[2];
}

// Functions to get the individual transforms in case only one of them is needed.

__forceinline__ __device__ void getTransformObjectToWorld(Float4* mW)
{
    OptixTraversableHandle handle = optixGetTransformListHandle(0);

    const Float4* tW = reinterpret_cast<const Float4*>(optixGetInstanceTransformFromHandle(handle));

    mW[0] = tW[0];
    mW[1] = tW[1];
    mW[2] = tW[2];
}

__forceinline__ __device__ void getTransformWorldToObject(Float4* mO)
{
    OptixTraversableHandle handle = optixGetTransformListHandle(0);

    const Float4* tO = reinterpret_cast<const Float4*>(optixGetInstanceInverseTransformFromHandle(handle));

    mO[0] = tO[0];
    mO[1] = tO[1];
    mO[2] = tO[2];
}

// Matrix3x4 * point. v.w == 1.0f
__forceinline__ __device__ Float3 transformPoint(const Float4* m, Float3 const& v)
{
    Float3 r;

    r.x = m[0].x * v.x + m[0].y * v.y + m[0].z * v.z + m[0].w;
    r.y = m[1].x * v.x + m[1].y * v.y + m[1].z * v.z + m[1].w;
    r.z = m[2].x * v.x + m[2].y * v.y + m[2].z * v.z + m[2].w;

    return r;
}

// Matrix3x4 * vector. v.w == 0.0f
__forceinline__ __device__ Float3 transformVector(const Float4* m, Float3 const& v)
{
    Float3 r;

    r.x = m[0].x * v.x + m[0].y * v.y + m[0].z * v.z;
    r.y = m[1].x * v.x + m[1].y * v.y + m[1].z * v.z;
    r.z = m[2].x * v.x + m[2].y * v.y + m[2].z * v.z;

    return r;
}

// InverseMatrix3x4^T * normal. v.w == 0.0f
// Get the inverse matrix as input and applies it as inverse transpose.
__forceinline__ __device__ Float3 transformNormal(const Float4* m, Float3 const& v)
{
    Float3 r;

    r.x = m[0].x * v.x + m[1].x * v.y + m[2].x * v.z;
    r.y = m[0].y * v.x + m[1].y * v.y + m[2].y * v.z;
    r.z = m[0].z * v.x + m[1].z * v.y + m[2].z * v.z;

    return r;
}


__forceinline__ __device__ LightSample LightEnvSphereSample(LightDefinition const& light, const Float3 point, const Float2 sample)
{
    LightSample lightSample;

    // Importance-sample the spherical environment light direction.

    // Note that the marginal CDF is one bigger than the texture height. As index this is the 1.0f at the end of the CDF.
    const unsigned int sizeV = sysParam.envHeight;

    unsigned int ilo = 0;     // Use this for full spherical lighting. (This matches the result of indirect environment lighting.)
    unsigned int ihi = sizeV; // Index on the last entry containing 1.0f. Can never be reached with the sample in the range [0.0f, 1.0f).

    const float* cdfV = sysParam.envCDF_V;

    // Binary search the row index to look up.
    while (ilo != ihi - 1) // When a pair of limits have been found, the lower index indicates the cell to use.
    {
        const unsigned int i = (ilo + ihi) >> 1;
        if (sample.y < cdfV[i]) // If the cdf is greater than the sample, use that as new higher limit.
        {
            ihi = i;
        }
        else // If the sample is greater than or equal to the CDF value, use that as new lower limit.
        {
            ilo = i;
        }
    }

    const unsigned int vIdx = ilo; // This is the row we found.

    // Note that the horizontal CDF is one bigger than the texture width. As index this is the 1.0f at the end of the CDF.
    const unsigned int sizeU = sysParam.envWidth; // Note that the horizontal CDFs are one bigger than the texture width.

    // Binary search the column index to look up.
    ilo = 0;
    ihi = sizeU; // Index on the last entry containing 1.0f. Can never be reached with the sample in the range [0.0f, 1.0f).

    // Pointer to the indexY row!
    const float* cdfU = &sysParam.envCDF_U[vIdx * (sizeU + 1)]; // Horizontal CDF is one bigger then the texture width!

    while (ilo != ihi - 1) // When a pair of limits have been found, the lower index indicates the cell to use.
    {
        const unsigned int i = (ilo + ihi) >> 1;
        if (sample.x < cdfU[i]) // If the CDF value is greater than the sample, use that as new higher limit.
        {
            ihi = i;
        }
        else // If the sample is greater than or equal to the CDF value, use that as new lower limit.
        {
            ilo = i;
        }
    }

    const unsigned int uIdx = ilo; // The column result.

    // Continuous sampling of the CDF.
    const float cdfLowerU = cdfU[uIdx];
    const float cdfUpperU = cdfU[uIdx + 1];
    const float du = (sample.x - cdfLowerU) / (cdfUpperU - cdfLowerU);

    const float cdfLowerV = cdfV[vIdx];
    const float cdfUpperV = cdfV[vIdx + 1];
    const float dv = (sample.y - cdfLowerV) / (cdfUpperV - cdfLowerV);

    // Texture lookup coordinates.
    const float u = (float(uIdx) + du) / float(sizeU);
    const float v = (float(vIdx) + dv) / float(sizeV);

    // Light sample direction vector polar coordinates. This is where the environment rotation happens!
    // FIXME Use a light.matrix to rotate the resulting vector instead.
    const float phi = (u - sysParam.envRotation) * 2.0f * M_PIf;
    const float theta = v * M_PIf; // theta == 0.0f is south pole, theta == M_PIf is north pole.

    const float sinTheta = sinf(theta);
    // The miss program places the 1->0 seam at the positive z-axis and looks from the inside.
    lightSample.direction = Float3(-sinf(phi) * sinTheta, // Starting on positive z-axis going around clockwise (to negative x-axis).
        -cosf(theta),          // From south pole to north pole.
        cosf(phi) * sinTheta); // Starting on positive z-axis.

    // Note that environment lights do not set the light sample position!
    lightSample.distance = RayMax; // Environment light.

    const Float3 emission = Float3(tex2D<float4>(sysParam.envTexture, u, v));
    // Explicit light sample. The returned emission must be scaled by the inverse probability to select this light.
    lightSample.emission = emission * sysParam.numLights;
    // For simplicity we pretend that we perfectly importance-sampled the actual texture-filtered environment map
    // and not the Gaussian-smoothed one used to actually generate the CDFs and uniform sampling in the texel.
    lightSample.pdf = intensity(emission) / sysParam.envIntegral;

    return lightSample;
}

__forceinline__ __device__ Float4 LightEnvSphereEval(LightDefinition const& light, const Float3 point, const Float3 R)
{
    // const Float3 R = rayData->wi; // theRay.direction;
    // The seam u == 0.0 == 1.0 is in positive z-axis direction.
    // Compensate for the environment rotation done inside the direct lighting.
    // FIXME Use a light.matrix to rotate the environment.
    const float u = (atan2f(R.x, -R.z) + M_PIf) * 0.5f * M_1_PIf + sysParam.envRotation;
    const float theta = acosf(-R.y);     // theta == 0.0f is south pole, theta == M_PIf is north pole.
    const float v = theta * M_1_PIf; // Texture is with origin at lower left, v == 0.0f is south pole.

    const Float3 emission = Float3(tex2D<float4>(sysParam.envTexture, u, v));
    const float pdfLight = intensity(emission) / sysParam.envIntegral;

    return Float4(emission, pdfLight);
}


extern "C" __global__ void __closesthit__radiance()
{
    PerRayData* rayData = mergePointer(optixGetPayload_0(), optixGetPayload_1());

    if (rayData->flags & FLAG_SHADOW)
    {
        rayData->flags |= FLAG_SHADOW_HIT;
        return;
    }

    GeometryInstanceData* theData = reinterpret_cast<GeometryInstanceData*>(optixGetSbtDataPointer());

    const unsigned int thePrimtiveIndex = optixGetPrimitiveIndex();

    const Int3 tri = theData->indices[thePrimtiveIndex];

    const VertexAttributes& va0 = theData->attributes[tri.x];
    const VertexAttributes& va1 = theData->attributes[tri.y];
    const VertexAttributes& va2 = theData->attributes[tri.z];

    const Float2 theBarycentrics = Float2(optixGetTriangleBarycentrics()); // beta and gamma
    const float alpha = 1.0f - theBarycentrics.x - theBarycentrics.y;

    const Float3 ng = cross(va1.vertex - va0.vertex, va2.vertex - va0.vertex);
    const Float3 ns = va0.normal * alpha + va1.normal * theBarycentrics.x + va2.normal * theBarycentrics.y;

    State state;
    state.texcoord = va0.texcoord * alpha + va1.texcoord * theBarycentrics.x + va2.texcoord * theBarycentrics.y;
    Float4 worldToObject[3];
    getTransformWorldToObject(worldToObject);
    state.normalGeo = normalize(transformNormal(worldToObject, ng));
    state.normal = normalize(transformNormal(worldToObject, ns));

    rayData->distance = optixGetRayTmax();
    rayData->pos = rayData->pos + rayData->wi * rayData->distance;
    // rayData->totalDistance += rayData->distance;

    rayData->rayConeWidth += rayData->rayConeSpread * rayData->distance;
    // rayState.rayConeSpread += surfaceRayConeSpread; // @TODO Based on the local surface curvature

    // Explicitly include edge-on cases as frontface condition! Keeps the material stack from overflowing at silhouettes.
    // Prevents that silhouettes of thin-walled materials use the backface material. Using the true geometry normal attribute as originally defined on the frontface!
    rayData->flags |= (0.0f <= dot(rayData->wo, state.normalGeo)) ? FLAG_FRONTFACE : 0;

    if ((rayData->flags & FLAG_FRONTFACE) == 0) // Looking at the backface?
    {
        // Means geometric normal and shading normal are always defined on the side currently looked at. This gives the backfaces of opaque BSDFs a defined result.
        state.normalGeo = -state.normalGeo;
        state.normal = -state.normal;
    }

    rayData->radiance = Float3(0.0f);
    rayData->f_over_pdf = Float3(0.0f);
    rayData->pdf = 0.0f;

    MaterialParameter parameters = sysParam.materialParameters[theData->materialIndex]; // Use a const reference, not all BSDFs need all values.

    Float3 albedo = parameters.albedo; // PERF Copy only this locally to be able to modulate it with the optional texture.

    float texMip0Size = parameters.texSize.length();
    float lod = log2f(rayData->rayConeWidth / max(dot(state.normal, rayData->wo), 0.01f) * parameters.uvScale * 2.0f * texMip0Size) - 3.0f;
    // lod = 0.0f;

    state.texcoord *= 2.0f;

    if (parameters.textureAlbedo != 0)
    {
        const Float3 texColor = Float3(tex2DLod<float4>(parameters.textureAlbedo, state.texcoord.x, state.texcoord.y, lod));
        albedo *= texColor;
    }

    if (parameters.textureNormal != 0)
    {
        Float3 texNormal = Float3(tex2DLod<float4>(parameters.textureNormal, state.texcoord.x, state.texcoord.y, lod));
        alignVector(state.normal, texNormal);
        state.normal = texNormal;
    }

    // if (parameters.textureRoughness != 0)
    // {
    //     float texRoughness = tex2DLod<float1>(parameters.textureNormal, state.texcoord.x, state.texcoord.y, lod).x;
    //     if (texRoughness < 0.1f)
    //     {
    //         parameters.indexBSDF = INDEX_BSDF_SPECULAR_REFLECTION;
    //     }
    //     else
    //     {
    //         parameters.indexBSDF = INDEX_BSDF_DIFFUSE_REFLECTION;
    //     }
    // }

    rayData->normal = state.normal;

    rayData->flags = rayData->flags | parameters.flags; // FLAG_THINWALLED can be set directly from the material parameters.

    const bool isDiffuse = parameters.indexBSDF >= NUM_SPECULAR_BSDF;

    rayData->albedo *= albedo;

    const int indexBsdfSample = NUM_LIGHT_TYPES + parameters.indexBSDF;

    Float3 surfWi;
    Float3 surfBsdfOverPdf;
    float surfSampleSurfPdf;

    optixDirectCall<void, MaterialParameter const&, State const&, PerRayData*, Float3&, Float3&, float&>(indexBsdfSample, parameters, state, rayData, surfWi, surfBsdfOverPdf, surfSampleSurfPdf);

    if (isDiffuse)
    {
        // rayData->albedo *= (1.0f + abs(dot(state.normal, rayData->centerRayDir)));
        bool shadowRayOnly = false;
        if (rayData->material & RAY_MAT_FLAG_DIFFUSE)
        {
            shadowRayOnly = true;
        }
        rayData->flags |= FLAG_DIFFUSED;
        rayData->material |= RAY_MAT_FLAG_DIFFUSE;

        // const int numLights = sysParam.numLights;
        const Float2 randNum = rayData->rand2();
        const int indexLight = 0; //(1 < numLights) ? clampi(static_cast<int>(floorf(rayData->rand() * numLights)), 0, numLights - 1) : 0;
        LightDefinition const& light = sysParam.lightDefinitions[indexLight];
        //const int indexCallable = light.type;
        //LightSample lightSample = optixDirectCall<LightSample, LightDefinition const&, const Float3, const Float2>(indexCallable, light, rayData->pos, randNum);
        LightSample lightSample = LightEnvSphereSample(light, rayData->pos, randNum);

        /// Directional light test
        // lightSample.pdf = 1.0f;
        // lightSample.direction = normalize(Float3(1, 1, 0));
        // lightSample.emission = Float3(1.0f);

        float lightSampleLightDistPdf = lightSample.pdf;

        // Float4 surfSampleEmissionPdf = LightEnvSphereEval(light, rayData->pos, surfWi);
        // Float3 surfSampleEmission = surfSampleEmissionPdf.xyz;
        // float surfSampleLightDistPdf = surfSampleEmissionPdf.w;

        /// Directional light test
        // if (dot(normalize(Float3(1, 1, 0)), surfWi) > cosf(Pi_over_180))
        // {
        //     surfSampleEmission = Float3(1.0f);
        // }
        // else
        // {
        //     surfSampleEmission = Float3(0.0f);
        // }
        // surfSampleLightDistPdf = 1.0f;

        // float misWeightSurfSample = powerHeuristic(surfSampleSurfPdf, surfSampleLightDistPdf);

        // shadowRayOnly = shadowRayOnly || (misWeightSurfSample < 0.1f);

        // rayData->lightEmission = surfSampleEmission * misWeightSurfSample;


        if (0.0f < lightSampleLightDistPdf) // Valid light sample, verify light distribution
        {
            const int indexBsdfEval = indexBsdfSample + 1;
            const Float4 lightSampleSurfDistBsdfPdf = optixDirectCall<Float4, MaterialParameter const&, State const&, PerRayData const*, const Float3>(indexBsdfEval, parameters, state, rayData, lightSample.direction);
            Float3 lightSampleSurfDistBsdf = lightSampleSurfDistBsdfPdf.xyz;
            float lightSampleSurfDistPdf = lightSampleSurfDistBsdfPdf.w;

            if (0.0f < lightSampleSurfDistPdf) // Valid light sample, verify surface distribution
            {
                // float chooseLightSampleWeight = misWeightLightSample / (misWeightLightSample + misWeightSurfSample);
                // if ((rayData->rand() < chooseLightSampleWeight) || shadowRayOnly)

                rayData->flags |= FLAG_SHADOW;

                UInt2 payload = UInt2(optixGetPayload_0(), optixGetPayload_1());
                optixTrace(sysParam.topObject,
                    (float3)rayData->pos, (float3)lightSample.direction,
                    sysParam.sceneEpsilon, RayMax, 0.0f, // tmin, tmax, time
                    OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                    0, 1, 0,
                    payload.x, payload.y);

                if (!(rayData->flags & FLAG_SHADOW_HIT))
                {
                    float misWeightLightSample = powerHeuristic(lightSampleLightDistPdf, lightSampleSurfDistPdf);
                    const float cosTheta = fmaxf(0.0f, dot(lightSample.direction, state.normal));
                    Float3 shadowRayBsdfOverPdf = lightSampleSurfDistBsdf * cosTheta / lightSampleLightDistPdf;
                    rayData->radiance += lightSample.emission * misWeightLightSample * shadowRayBsdfOverPdf;
                }

                if (shadowRayOnly)
                {
                    rayData->flags |= FLAG_TERMINATE;
                    return;
                }
            }
        }
    }

    if (parameters.indexBSDF == INDEX_BSDF_SPECULAR_REFLECTION)
    {
        rayData->material |= RAY_MAT_FLAG_REFL << (2 * rayData->depth);
    }
    else if (parameters.indexBSDF == INDEX_BSDF_SPECULAR_REFLECTION_TRANSMISSION)
    {
        if (rayData->flags & FLAG_VOLUME)
        {
            rayData->material |= RAY_MAT_FLAG_REFR << (2 * rayData->depth);
        }
        else
        {
            rayData->material |= RAY_MAT_FLAG_REFL_REFR << (2 * rayData->depth);
        }
    }

    rayData->wi = surfWi;
    rayData->f_over_pdf = surfBsdfOverPdf;
    rayData->pdf = surfSampleSurfPdf;
}

}