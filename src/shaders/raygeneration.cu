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
#include "shader_common.h"
#include "random_number_generators.h"

#include "ShaderDebugUtils.h"

extern "C" __constant__ SystemParameter sysParameter;

__device__ __inline__ void PinholeCamera(const float2 screen, const float2 pixel, const float2 sample, float3 &origin, float3 &direction)
{
    const float2 fragment = pixel + sample;               // Jitter the sub-pixel location
    const float2 ndc = (fragment / screen) * 2.0f - 1.0f; // Normalized device coordinates in range [-1, 1].

    origin = sysParameter.cameraPosition;
    direction = normalize(sysParameter.cameraU * ndc.x + sysParameter.cameraV * ndc.y + sysParameter.cameraW);
}

__device__ __inline__ bool TraceNextPath(PerRayData& prd, float4* absorptionStack, float3& radiance, float3& throughput, int& stackIdx, int& depth)
{
    prd.wo = -prd.wi;              // Direction to observer.
    prd.ior = make_float2(1.0f);   // Reset the volume IORs.
    prd.distance = RT_DEFAULT_MAX; // Shoot the next ray with maximum length.
    prd.flags &= FLAG_CLEAR_MASK;  // Clear all non-persistent flags. In this demo only the last diffuse surface interaction stays.

    // Handle volume absorption of nested materials.
    if (MATERIAL_STACK_FIRST <= stackIdx) // Inside a volume?
    {
        prd.flags |= FLAG_VOLUME;                                // Indicate that we're inside a volume. => At least absorption calculation needs to happen.
        prd.extinction = make_float3(absorptionStack[stackIdx]); // There is only volume absorption in this demo, no volume scattering.
        prd.ior.x = absorptionStack[stackIdx].w;                 // The IOR of the volume we're inside. Needed for eta calculations in transparent materials.
        if (MATERIAL_STACK_FIRST <= stackIdx - 1)
        {
            prd.ior.y = absorptionStack[stackIdx - 1].w; // The IOR of the surrounding volume. Needed when potentially leaving a volume to calculate eta in transparent materials.
        }
    }

    // Note that the primary rays (or volume scattering miss cases) wouldn't normally offset the ray t_min by sysSceneEpsilon. Keep it simple here.

    // Put radiance payload pointer into two unsigned integers.
    uint2 payload = splitPointer(&prd);

    optixTrace(sysParameter.topObject,
                prd.pos, prd.wi,                               // origin, direction
                sysParameter.sceneEpsilon, prd.distance, 0.0f, // tmin, tmax, time
                OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                0, 1, 0,
                payload.x, payload.y);

    // This renderer supports nested volumes.
    if (prd.flags & FLAG_VOLUME)
    {
        // We're inside a volume. Calculate the extinction along the current path segment in any case.
        // The transmittance along the current path segment inside a volume needs to attenuate the ray throughput with the extinction
        // before it modulates the radiance of the hitpoint.
        throughput *= expf(-prd.distance * prd.extinction);
    }

    radiance += throughput * prd.radiance;

    // Path termination by miss shader or sample() routines.
    // If terminate is true, f_over_pdf and pdf might be undefined.
    if ((prd.flags & FLAG_TERMINATE) || prd.pdf <= 0.0f || isNull(prd.f_over_pdf))
    {
        return false;
    }

    // PERF f_over_pdf already contains the proper throughput adjustment for diffuse materials: f * (fabsf(optix::dot(prd.wi, state.normal)) / prd.pdf);
    throughput *= prd.f_over_pdf;

    // Adjust the material volume stack if the geometry is not thin-walled but a border between two volumes
    // and the outgoing ray direction was a transmission.
    if ((prd.flags & (FLAG_THINWALLED | FLAG_TRANSMISSION)) == FLAG_TRANSMISSION)
    {
        // Transmission.
        if (prd.flags & FLAG_FRONTFACE) // Entered a new volume?
        {
            // Push the entered material's volume properties onto the volume stack.
            // rtAssert((stackIdx < MATERIAL_STACK_LAST), 1); // Overflow?
            stackIdx = min(stackIdx + 1, MATERIAL_STACK_LAST);
            absorptionStack[stackIdx] = prd.absorption_ior;
        }
        else // Exited the current volume?
        {
            // Pop the top of stack material volume.
            // This assert fires and is intended because I tuned the frontface checks so that there are more exits than enters at silhouettes.
            // rtAssert((MATERIAL_STACK_EMPTY < stackIdx), 0); // Underflow?
            stackIdx = max(stackIdx - 1, MATERIAL_STACK_EMPTY);
        }
    }

    ++depth; // Next path segment.

    return true;
}

extern "C" __global__ void __raygen__pathtracer()
{
    PerRayData prd;

    // // This assumes that the launch dimensions are matching the size of the output buffer.
    const uint3 theLaunchDim = optixGetLaunchDimensions();
    const uint3 theLaunchIndex = optixGetLaunchIndex();

    // Initialize the random number generator seed from the linear pixel index and the iteration index.
    prd.seed = tea<4>(theLaunchIndex.y * theLaunchDim.x + theLaunchIndex.x, sysParameter.iterationIndex);

    // Decoupling the pixel coordinates from the screen size will allow for partial rendering algorithms.
    // In this case theLaunchIndex is the pixel coordinate and theLaunchDim is sysOutputBuffer.size().
    const float2 screen = make_float2(theLaunchDim);
    const float2 pixel = make_float2(theLaunchIndex);
    const float2 sample = rng2(prd.seed);

    // Lens shaders
    // optixDirectCall<void, const float2, const float2, const float2, float3 &, float3 &>(sysParameter.cameraType, );
    PinholeCamera(screen, pixel, sample, prd.pos, prd.wi);

    // This renderer supports nested volumes. Four levels is plenty enough for most cases.
    // The absorption coefficient and IOR of the volume the ray is currently inside.
    float4 absorptionStack[MATERIAL_STACK_SIZE]; // .xyz == absorptionCoefficient (sigma_a), .w == index of refraction

    float3 radiance = make_float3(1.0f);

    float3 throughput = make_float3(1.0f); // The throughput for the next radiance, starts with 1.0f.

    int stackIdx = MATERIAL_STACK_EMPTY; // Start with empty nested materials stack.

    // Russian Roulette path termination after a specified number of bounces needs the current depth.
    int depth = 0; // Path segment index. Primary ray is 0.

    prd.absorption_ior = make_float4(0.0f, 0.0f, 0.0f, 1.0f); // Assume primary ray starts in vacuum.
    prd.flags = 0;

    while (depth < BounceLimit)
    {
        if (TraceNextPath(prd, absorptionStack, radiance, throughput, stackIdx, depth) == false)
            break;
    }

    if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z))
    {
        radiance = make_float3(10000.0f, 0.0f, 0.0f);
    }

    const unsigned int index = theLaunchIndex.y * theLaunchDim.x + theLaunchIndex.x;
    sysParameter.outputBuffer[index] = make_float4(radiance, 1.0f);
}
