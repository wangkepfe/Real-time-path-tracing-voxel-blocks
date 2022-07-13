#include "OptixShaderCommon.h"
#include "SystemParameter.h"
#include "ShaderDebugUtils.h"

namespace jazzfusion
{

extern "C" __constant__ SystemParameter sysParam;

extern "C" __global__ void __miss__env_sphere()
{
    // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
    PerRayData* rayData = mergePointer(optixGetPayload_0(), optixGetPayload_1());

    if (rayData->flags & FLAG_SHADOW)
    {
        return;
    }

    Float3 emission;

    // if ((rayData->flags & FLAG_DIFFUSED))
    // {
    //     emission = rayData->lightEmission;
    // }
    // else
    // {
    const Float3 R = rayData->wi; // theRay.direction;
    // The seam u == 0.0 == 1.0 is in positive z-axis direction.
    // Compensate for the environment rotation done inside the direct lighting.
    // FIXME Use a light.matrix to rotate the environment.
    const float u = (atan2f(R.x, -R.z) + M_PIf) * 0.5f * M_1_PIf + sysParam.envRotation;
    const float theta = acosf(-R.y);     // theta == 0.0f is south pole, theta == M_PIf is north pole.
    const float v = theta * M_1_PIf; // Texture is with origin at lower left, v == 0.0f is south pole.

    emission = Float3(tex2D<float4>(sysParam.envTexture, u, v));

    /// Directional light test
    if (0)
    {
        if (dot(normalize(Float3(1, 1, 0)), R) > cosf(Pi_over_180))
        {
            emission = Float3(1.0f);
        }
        else
        {
            emission = Float3(0.0f);
        }
    }

    float weightMIS = 1.0f;
    // // If the last surface intersection was a diffuse event which was directly lit with multiple importance sampling,
    // // then calculate light emission with multiple importance sampling for this implicit light hit as well.
    if (rayData->flags & FLAG_DIFFUSED)
    {
        // For simplicity we pretend that we perfectly importance-sampled the actual texture-filtered environment map
        // and not the Gaussian smoothed one used to actually generate the CDFs.
        const float pdfLight = intensity(emission) / sysParam.envIntegral;
        weightMIS = powerHeuristic(rayData->pdf, pdfLight);

        // if (OPTIX_CENTER_PIXEL())
        // {
        //     OPTIX_DEBUG_PRINT(Float4(pdfLight, weightMIS, 0, 0));
        // }
    }
    rayData->radiance = emission * weightMIS;

    // rayData->totalDistance = RayMax;
    rayData->distance = RayMax;
    rayData->flags |= FLAG_TERMINATE;
    if (!(rayData->flags & FLAG_DIFFUSED))
    {
        rayData->material |= RAY_MAT_FLAG_SKY << (2 * rayData->depth);
    }

}

}