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