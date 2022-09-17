#include "SystemParameter.h"
#include "OptixShaderCommon.h"

namespace jazzfusion
{

extern "C" __constant__ SystemParameter sysParam;


__forceinline__ __device__ void unitSquareToSphere(const float u, const float v, Float3& p, float& pdf)
{
    p.z = 1.0f - 2.0f * u;
    float r = 1.0f - p.z * p.z;
    r = (0.0f < r) ? sqrtf(r) : 0.0f;

    const float phi = v * 2.0f * M_PIf;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    pdf = 0.25f * M_1_PIf; // == 1.0f / (4.0f * M_PIf)
}

extern "C" __device__ LightSample __direct_callable__light_env_sphere(LightDefinition const& light, const Float3 point, const Float2 sample)
{
    return {};
}

extern "C" __device__ LightSample __direct_callable__light_parallelogram(LightDefinition const& light, const Float3 point, const Float2 sample)
{
    LightSample lightSample;

    lightSample.pdf = 0.0f; // Default return, invalid light sample (backface, edge on, or too near to the surface)

    lightSample.position = light.position + light.vecU * sample.x + light.vecV * sample.y; // The light sample position in world coordinates.
    lightSample.direction = lightSample.position - point;                                  // Sample direction from surface point to light sample position.
    lightSample.distance = lightSample.direction.length();
    if (DenominatorEpsilon < lightSample.distance)
    {
        lightSample.direction /= lightSample.distance; // Normalized direction to light.

        const float cosTheta = dot(-lightSample.direction, light.normal);
        if (DenominatorEpsilon < cosTheta) // Only emit light on the front side.
        {
            // Explicit light sample, must scale the emission by inverse probabilty to hit this light.
            lightSample.emission = light.emission * float(sysParam.numLights);
            lightSample.pdf = (lightSample.distance * lightSample.distance) / (light.area * cosTheta); // Solid angle pdf. Assumes light.area != 0.0f.
        }
    }

    return lightSample;
}

}