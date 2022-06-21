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

extern "C" __global__ void __closesthit__radiance()
{
    PerRayData* rayData = mergePointer(optixGetPayload_0(), optixGetPayload_1());

    if (rayData->flags & FLAG_SHADOW)
    {
        rayData->flags |= FLAG_TERMINATE;
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

    // Explicitly include edge-on cases as frontface condition! Keeps the material stack from overflowing at silhouettes.
    // Prevents that silhouettes of thin-walled materials use the backface material. Using the true geometry normal attribute as originally defined on the frontface!
    rayData->flags |= (0.0f <= dot(rayData->wo, state.normalGeo)) ? FLAG_FRONTFACE : 0;

    if ((rayData->flags & FLAG_FRONTFACE) == 0) // Looking at the backface?
    {
        // Means geometric normal and shading normal are always defined on the side currently looked at. This gives the backfaces of opaque BSDFs a defined result.
        state.normalGeo = -state.normalGeo;
        state.normal = -state.normal;
    }

    rayData->normal = state.normal;

    rayData->radiance = Float3(0.0f);
    rayData->f_over_pdf = Float3(0.0f);
    rayData->pdf = 0.0f;

    MaterialParameter parameters = sysParam.materialParameters[theData->materialIndex]; // Use a const reference, not all BSDFs need all values.

    state.albedo = parameters.albedo; // PERF Copy only this locally to be able to modulate it with the optional texture.

    if (parameters.textureAlbedo != 0)
    {
        const Float3 texColor = Float3(tex2D<float4>(parameters.textureAlbedo, state.texcoord.x, state.texcoord.y));
        state.albedo *= texColor;
    }

    rayData->flags = rayData->flags | parameters.flags; // FLAG_THINWALLED can be set directly from the material parameters.

    rayData->material = parameters.indexBSDF;
    rayData->albedo *= state.albedo;

    const int indexBsdfSample = NUM_LIGHT_TYPES + parameters.indexBSDF;

    const bool isDiffuse = parameters.indexBSDF >= NUM_SPECULAR_BSDF;

    Float3 surfWi;
    Float3 surfBsdfOverPdf;
    float surfPdf;

    optixDirectCall<void, MaterialParameter const&, State const&, PerRayData*, Float3&, Float3&, float&>(indexBsdfSample, parameters, state, rayData, surfWi, surfBsdfOverPdf, surfPdf);

    if (isDiffuse)
    {
        const int numLights = sysParam.numLights;
        const Float2 randNum = rng2(rayData->seed);
        const int indexLight = (1 < numLights) ? clampi(static_cast<int>(floorf(rng(rayData->seed) * numLights)), 0, numLights - 1) : 0;
        LightDefinition const& light = sysParam.lightDefinitions[indexLight];
        const int indexCallable = light.type;
        LightSample lightSample = optixDirectCall<LightSample, LightDefinition const&, const Float3, const Float2>(indexCallable, light, rayData->pos, randNum);

        float misLightSurf = powerHeuristic(lightSample.pdf, surfPdf);
        if (0.0f < lightSample.pdf && rng(rayData->seed) < misLightSurf) // Useful light sample?
        {
            // Evaluate the BSDF in the light sample direction. Normally cheaper than shooting rays.
            // Returns BSDF f in .xyz and the BSDF pdf in .w
            // BSDF eval function is one index after the sample fucntion.
            const int indexBsdfEval = indexBsdfSample + 1;
            const Float4 lightBsdfAndPdf = optixDirectCall<Float4, MaterialParameter const&, State const&, PerRayData const*, const Float3>(indexBsdfEval, parameters, state, rayData, lightSample.direction);
            Float3 lightBsdf = lightBsdfAndPdf.xyz;
            float lightPdf = lightBsdfAndPdf.w;

            if (0.0f < lightPdf && isNotNull(lightBsdf))
            {
                rayData->flags |= FLAG_SHADOW;

                const float misWeight = powerHeuristic(lightSample.pdf, lightPdf);

                rayData->wi = lightSample.direction;
                rayData->f_over_pdf = misWeight * lightBsdf / lightSample.pdf;
                rayData->pdf = lightSample.pdf;
            }
        }
        else
        {
            rayData->wi = surfWi;
            rayData->f_over_pdf = surfBsdfOverPdf;
            rayData->pdf = surfPdf;
        }
    }
    else
    {
        rayData->wi = surfWi;
        rayData->f_over_pdf = surfBsdfOverPdf;
        rayData->pdf = surfPdf;
    }

}

}