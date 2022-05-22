#include <optix.h>
#include "MathUtils.h"
#include "SystemParameter.h"

extern "C" __constant__ SystemParameter sysParameter;

extern "C" __global__ void __miss__env_sphere()
{
    // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
    PerRayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

    const float3 R = thePrd->wi; // theRay.direction;
    // The seam u == 0.0 == 1.0 is in positive z-axis direction.
    // Compensate for the environment rotation done inside the direct lighting.
    // FIXME Use a light.matrix to rotate the environment.
    const float u = (atan2f(R.x, -R.z) + M_PIf) * 0.5f * M_1_PIf + sysParameter.envRotation;
    const float theta = acosf(-R.y);     // theta == 0.0f is south pole, theta == M_PIf is north pole.
    const float v = theta * M_1_PIf; // Texture is with origin at lower left, v == 0.0f is south pole.

    const float3 emission = make_float3(tex2D<float4>(sysParameter.envTexture, u, v));

    thePrd->radiance = emission;
    thePrd->flags |= FLAG_TERMINATE;
}
