#include "OptixShaderCommon.h"
#include "SystemParameter.h"
#include "ShaderDebugUtils.h"
#include "Sampler.h"

namespace jazzfusion
{

    extern "C" __constant__ SystemParameter sysParam;

    extern "C" __global__ void __miss__env_sphere()
    {
        // Get the current rtPayload pointer from the unsigned int payload registers p0 and p1.
        PerRayData *rayData = mergePointer(optixGetPayload_0(), optixGetPayload_1());

        if (rayData->flags & FLAG_SHADOW)
        {
            return;
        }

        Float3 emission = Float3(0);
        float misWeight = 1.0f;

        const Int2 &skyRes = sysParam.skyRes;
        const Int2 &sunRes = sysParam.sunRes;
        const int skySize = skyRes.x * skyRes.y;
        const int sunSize = sunRes.x * sunRes.y;
        const float *skyCdf = sysParam.skyCdf;
        const float *sunCdf = sysParam.sunCdf;
        const float sunAngle = 0.51f; // angular diagram in degrees
        const float sunAngleCosThetaMax = cosf(sunAngle * M_PI / 180.0f / 2.0f);

        const Float3 &rayDir = rayData->wi;

        // Map the ray diretcion to uv
        Float2 uv;
        bool hitDisk = EqualAreaMapCone(uv, sysParam.sunDir, rayDir, sunAngleCosThetaMax);

        // Sky
        {
            // Map the ray diretcion to uv
            uv = EqualAreaMap(rayDir);
            // Int2 skyIdx((int)(uv.x * skyRes.x), (int)(uv.y * skyRes.y));
            Float2 skyUv = uv * skyRes;

            // Wrapping around on X dimension
            // if (skyIdx.x >= skyRes.x) { skyIdx.x %= skyRes.x; }
            // if (skyIdx.x < 0) { skyIdx.x = skyRes.x - (-skyIdx.x) % skyRes.x; }

            // Load buffer
            // Float3 skyEmission = Load2DHalf4(sysParam.skyBuffer, skyIdx).xyz;

            Float3 skyEmission = SampleBicubicSmoothStep<Load2DFuncFloat4<Float3>, Float3, BoundaryFuncRepeatXClampY>(sysParam.skyBuffer, skyUv, skyRes);

            // Blend the sky color with mist
            Float3 mistColor = Float3(0.2f);
            float blenderFactor = clampf((rayDir.y + 0.4f) * (1.0f / 0.5f));

            // If the last surface intersection was a diffuse event which was directly lit with multiple importance sampling,
            // then calculate light emission with multiple importance sampling for this implicit light hit as well.
            if (rayData->flags & FLAG_DIFFUSED)
            {
                const float maxSkyCdf = skyCdf[skySize - 1];
                float lightSamplePdf = dot(skyEmission, Float3(0.3f, 0.6f, 0.1f)) / maxSkyCdf;
                lightSamplePdf *= skySize / TWO_PI;
                misWeight = powerHeuristic(rayData->pdf, lightSamplePdf);
            }

            // Add sky emission
            emission += smoothstep3f(mistColor, skyEmission, blenderFactor);
        }

        // Is the ray hitting the solar disk, then compute the sun emission
        if (hitDisk)
        {
            Int2 sunIdx((int)(uv.x * sunRes.x), (int)(uv.y * sunRes.y));
            // Float2 sunUv = uv * sunRes;

            // Wrapping around on X dimension
            if (sunIdx.x >= sunRes.x)
            {
                sunIdx.x %= sunRes.x;
            }
            if (sunIdx.x < 0)
            {
                sunIdx.x = sunRes.x - (-sunIdx.x) % sunRes.x;
            }
            Float3 sunEmission = Load2DHalf4(sysParam.sunBuffer, sunIdx).xyz;

            // Float3 sunEmission = SampleBicubicSmoothStep<Load2DFuncHalf4<Float3>, Float3, BoundaryFuncRepeatXClampY>(sysParam.sunBuffer, sunUv, sunRes);

            // If the last surface intersection was a diffuse event which was directly lit with multiple importance sampling,
            // then calculate light emission with multiple importance sampling for this implicit light hit as well.
            if (rayData->flags & FLAG_DIFFUSED)
            {
                const float maxSunCdf = sunCdf[sunSize - 1];
                float lightSamplePdf = dot(sunEmission, Float3(0.3f, 0.6f, 0.1f)) / maxSunCdf;
                lightSamplePdf *= sunSize / (TWO_PI * (1.0f - sunAngleCosThetaMax));
                misWeight = powerHeuristic(rayData->pdf, lightSamplePdf);
            }

            // Add sun emission
            emission += sunEmission;
        }

        rayData->radiance = emission * misWeight;

        // rayData->totalDistance = RayMax;
        rayData->distance = RayMax;
        rayData->flags |= FLAG_TERMINATE;
        if (!(rayData->flags & FLAG_DIFFUSED))
        {
            rayData->material |= RAY_MAT_FLAG_SKY << (2 * rayData->depth);
        }
    }

}