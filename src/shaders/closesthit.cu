/*
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "app_config.h"

#include <optix.h>

#include "system_parameter.h"
#include "per_ray_data.h"
#include "vertex_attributes.h"
#include "material_parameter.h"
#include "function_indices.h"
#include "light_definition.h"
#include "shader_common.h"
#include "random_number_generators.h"

#include "ShaderDebugUtils.h"

extern "C" __constant__ SystemParameter sysParameter;

// Get the 3x4 object to world transform and its inverse from a two-level hierarchy.
// Arguments float4* objectToWorld, float4* worldToObject shortened for smaller code.
__forceinline__ __device__ void getTransforms(float4 *mW, float4 *mO)
{
    OptixTraversableHandle handle = optixGetTransformListHandle(0);

    const float4 *tW = optixGetInstanceTransformFromHandle(handle);
    const float4 *tO = optixGetInstanceInverseTransformFromHandle(handle);

    mW[0] = tW[0];
    mW[1] = tW[1];
    mW[2] = tW[2];

    mO[0] = tO[0];
    mO[1] = tO[1];
    mO[2] = tO[2];
}

// Functions to get the individual transforms in case only one of them is needed.

__forceinline__ __device__ void getTransformObjectToWorld(float4 *mW)
{
    OptixTraversableHandle handle = optixGetTransformListHandle(0);

    const float4 *tW = optixGetInstanceTransformFromHandle(handle);

    mW[0] = tW[0];
    mW[1] = tW[1];
    mW[2] = tW[2];
}

__forceinline__ __device__ void getTransformWorldToObject(float4 *mO)
{
    OptixTraversableHandle handle = optixGetTransformListHandle(0);

    const float4 *tO = optixGetInstanceInverseTransformFromHandle(handle);

    mO[0] = tO[0];
    mO[1] = tO[1];
    mO[2] = tO[2];
}

// Matrix3x4 * point. v.w == 1.0f
__forceinline__ __device__ float3 transformPoint(const float4 *m, float3 const &v)
{
    float3 r;

    r.x = m[0].x * v.x + m[0].y * v.y + m[0].z * v.z + m[0].w;
    r.y = m[1].x * v.x + m[1].y * v.y + m[1].z * v.z + m[1].w;
    r.z = m[2].x * v.x + m[2].y * v.y + m[2].z * v.z + m[2].w;

    return r;
}

// Matrix3x4 * vector. v.w == 0.0f
__forceinline__ __device__ float3 transformVector(const float4 *m, float3 const &v)
{
    float3 r;

    r.x = m[0].x * v.x + m[0].y * v.y + m[0].z * v.z;
    r.y = m[1].x * v.x + m[1].y * v.y + m[1].z * v.z;
    r.z = m[2].x * v.x + m[2].y * v.y + m[2].z * v.z;

    return r;
}

// InverseMatrix3x4^T * normal. v.w == 0.0f
// Get the inverse matrix as input and applies it as inverse transpose.
__forceinline__ __device__ float3 transformNormal(const float4 *m, float3 const &v)
{
    float3 r;

    r.x = m[0].x * v.x + m[1].x * v.y + m[2].x * v.z;
    r.y = m[0].y * v.x + m[1].y * v.y + m[2].y * v.z;
    r.z = m[0].z * v.x + m[1].z * v.y + m[2].z * v.z;

    return r;
}

extern "C" __global__ void __closesthit__radiance()
{
    PerRayData *thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

    thePrd->f_over_pdf = make_float3(1.0f);

    if (thePrd->flags & FLAG_SHADOW)
    {
        thePrd->flags |= FLAG_TERMINATE;
        return;
    }

    GeometryInstanceData *theData = reinterpret_cast<GeometryInstanceData *>(optixGetSbtDataPointer());

    const unsigned int thePrimtiveIndex = optixGetPrimitiveIndex();

    const int3 tri = theData->indices[thePrimtiveIndex];

    const VertexAttributes &va0 = theData->attributes[tri.x];
    const VertexAttributes &va1 = theData->attributes[tri.y];
    const VertexAttributes &va2 = theData->attributes[tri.z];

    const float2 theBarycentrics = optixGetTriangleBarycentrics(); // beta and gamma
    const float alpha = 1.0f - theBarycentrics.x - theBarycentrics.y;

    const float3 ng = cross(va1.vertex - va0.vertex, va2.vertex - va0.vertex);
    const float3 ns = va0.normal * alpha + va1.normal * theBarycentrics.x + va2.normal * theBarycentrics.y;

    State state;
    state.texcoord = va0.texcoord * alpha + va1.texcoord * theBarycentrics.x + va2.texcoord * theBarycentrics.y;
    float4 worldToObject[3];
    getTransformWorldToObject(worldToObject);
    state.normalGeo = normalize(transformNormal(worldToObject, ng));
    state.normal = normalize(transformNormal(worldToObject, ns));

    thePrd->distance = optixGetRayTmax();
    thePrd->pos = thePrd->pos + thePrd->wi * thePrd->distance;

    // Explicitly include edge-on cases as frontface condition! Keeps the material stack from overflowing at silhouettes.
    // Prevents that silhouettes of thin-walled materials use the backface material. Using the true geometry normal attribute as originally defined on the frontface!
    thePrd->flags |= (0.0f <= dot(thePrd->wo, state.normalGeo)) ? FLAG_FRONTFACE : 0;

    if ((thePrd->flags & FLAG_FRONTFACE) == 0) // Looking at the backface?
    {
        // Means geometric normal and shading normal are always defined on the side currently looked at. This gives the backfaces of opaque BSDFs a defined result.
        state.normalGeo = -state.normalGeo;
        state.normal = -state.normal;
    }

    thePrd->radiance = make_float3(0.0f);
    thePrd->f_over_pdf = make_float3(0.0f);
    thePrd->pdf = 0.0f;

    MaterialParameter parameters = sysParameter.materialParameters[theData->materialIndex]; // Use a const reference, not all BSDFs need all values.

    state.albedo = parameters.albedo; // PERF Copy only this locally to be able to modulate it with the optional texture.

    if (parameters.textureAlbedo != 0)
    {
        const float3 texColor = make_float3(tex2D<float4>(parameters.textureAlbedo, state.texcoord.x, state.texcoord.y));
        state.albedo *= texColor;
    }

    thePrd->flags = (thePrd->flags & ~FLAG_DIFFUSE); // Only the last diffuse hit is tracked for multiple importance sampling of implicit light hits.
    thePrd->flags = thePrd->flags | parameters.flags; // FLAG_THINWALLED can be set directly from the material parameters.

    const int indexBsdfSample = NUM_LIGHT_TYPES + parameters.indexBSDF;

    const bool isDiffuse = parameters.indexBSDF >= NUM_SPECULAR_BSDF;

    float3 surfWi;
    float3 surfBsdfOverPdf;
    float surfPdf;

    optixDirectCall<void, MaterialParameter const &, State const &, PerRayData *, float3&, float3&, float&>(indexBsdfSample, parameters, state, thePrd, surfWi, surfBsdfOverPdf, surfPdf);

    if (isDiffuse)
    {
        thePrd->flags |= FLAG_DIFFUSE;

        const int numLights = sysParameter.numLights;
        const float2 randNum = rng2(thePrd->seed);
        const int indexLight = (1 < numLights) ? clamp(static_cast<int>(floorf(rng(thePrd->seed) * numLights)), 0, numLights - 1) : 0;
        LightDefinition const &light = sysParameter.lightDefinitions[indexLight];
        const int indexCallable = light.type;
        LightSample lightSample = optixDirectCall<LightSample, LightDefinition const &, const float3, const float2>(indexCallable, light, thePrd->pos, randNum);

        float misLightSurf = powerHeuristic(lightSample.pdf, surfPdf);
        if (0.0f < lightSample.pdf && rng(thePrd->seed) < misLightSurf) // Useful light sample?
        {
            // Evaluate the BSDF in the light sample direction. Normally cheaper than shooting rays.
            // Returns BSDF f in .xyz and the BSDF pdf in .w
            // BSDF eval function is one index after the sample fucntion.
            const int indexBsdfEval = indexBsdfSample + 1;
            const float4 lightBsdfAndPdf = optixDirectCall<float4, MaterialParameter const &, State const &, PerRayData const *, const float3>(indexBsdfEval, parameters, state, thePrd, lightSample.direction);
            float3 lightBsdf = make_float3(lightBsdfAndPdf);
            float lightPdf = lightBsdfAndPdf.w;

            if (0.0f < lightPdf && isNotNull(lightBsdf))
            {
                thePrd->flags |= FLAG_SHADOW;

                const float misWeight = powerHeuristic(lightSample.pdf, lightPdf);

                thePrd->wi = lightSample.direction;
                thePrd->f_over_pdf = misWeight * lightBsdf / lightSample.pdf;
                thePrd->pdf = lightSample.pdf;
            }
        }
        else
        {
            thePrd->wi = surfWi;
            thePrd->f_over_pdf = surfBsdfOverPdf;
            thePrd->pdf = surfPdf;
        }
    }
    else
    {
        thePrd->wi = surfWi;
        thePrd->f_over_pdf = surfBsdfOverPdf;
        thePrd->pdf = surfPdf;
    }

}
