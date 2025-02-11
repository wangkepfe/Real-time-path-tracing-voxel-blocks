#include "OptixShaderCommon.h"
#include "SystemParameter.h"
#include "ShaderDebugUtils.h"
#include "Sampler.h"

extern "C" __constant__ SystemParameter sysParam;

extern "C" __global__ void __miss__radiance()
{
    RayData *rayData = (RayData *)mergePointer(optixGetPayload_0(), optixGetPayload_1());

    if (rayData->isShadowRay)
    {
        return;
    }

    // Float3 emission = Float3(0);

    // const Int2 &skyRes = sysParam.skyRes;
    // const Int2 &sunRes = sysParam.sunRes;
    // const int skySize = skyRes.x * skyRes.y;
    // const int sunSize = sunRes.x * sunRes.y;
    // const float &accumulatedSkyLuminance = sysParam.accumulatedSkyLuminance;
    // const float &accumulatedSunLuminance = sysParam.accumulatedSunLuminance;
    // const float sunAngle = 0.51f; // angular diagram in degrees
    // const float sunAngleCosThetaMax = cosf(sunAngle * M_PI / 180.0f / 2.0f);

    // const Float3 &rayDir = rayData->wi;

    // const float totalSkyLum = accumulatedSkyLuminance * TWO_PI / skySize; // Jacobian of the hemisphere mapping
    // const float totalSunLum = accumulatedSunLuminance * TWO_PI * (1.0f - sunAngleCosThetaMax) / sunSize;

    // // Sample sky or sun pdf
    // const float sampleSkyVsSun = totalSkyLum / (totalSkyLum + totalSunLum);

    // // Map the ray diretcion to uv
    // Float2 uv;

    // // Sky
    // {
    //     // Map the ray diretcion to uv
    //     uv = EqualAreaMap(rayDir);
    //     Float3 skyEmission = SampleBicubicSmoothStep<Load2DFuncFloat4<Float3>, Float3, BoundaryFuncRepeatXClampY>(sysParam.skyBuffer, uv, skyRes);

    //     // Blend the sky color with mist
    //     Float3 mistColor = Float3(accumulatedSkyLuminance / skySize); // Average color of elevation=0
    //     float blenderFactor = clampf((rayDir.y + 0.4f) * (1.0f / 0.5f));

    //     float misWeight = 1.0f;

    //     if (rayData->isLastBounceDiffuse)
    //     {
    //         float lightSamplePdf = dot(skyEmission, Float3(0.3f, 0.6f, 0.1f)) / accumulatedSkyLuminance;
    //         lightSamplePdf *= skySize / TWO_PI;
    //         lightSamplePdf *= sampleSkyVsSun;
    //         misWeight = powerHeuristic(rayData->pdf, lightSamplePdf);
    //     }

    //     // Add sky emission
    //     emission += smoothstep3f(mistColor, skyEmission, blenderFactor) * misWeight;
    // }

    // // Is the ray hitting the solar disk, then compute the sun emission
    // bool hitDisk = EqualAreaMapCone(uv, sysParam.sunDir, rayDir, sunAngleCosThetaMax);
    // if (hitDisk)
    // {
    //     Int2 sunIdx((int)(uv.x * sunRes.x), (int)(uv.y * sunRes.y));

    //     // Wrapping around on X dimension
    //     if (sunIdx.x >= sunRes.x)
    //     {
    //         sunIdx.x %= sunRes.x;
    //     }
    //     if (sunIdx.x < 0)
    //     {
    //         sunIdx.x = sunRes.x - (-sunIdx.x) % sunRes.x;
    //     }
    //     Float3 sunEmission = Load2DFloat4(sysParam.sunBuffer, sunIdx).xyz;

    //     float misWeight = 1.0f;

    //     if (rayData->isLastBounceDiffuse)
    //     {
    //         float lightSamplePdf = dot(sunEmission, Float3(0.3f, 0.6f, 0.1f)) / accumulatedSunLuminance;
    //         lightSamplePdf *= sunSize / (TWO_PI * (1.0f - sunAngleCosThetaMax));
    //         lightSamplePdf *= (1.0f - sampleSkyVsSun);
    //         misWeight = powerHeuristic(rayData->pdf, lightSamplePdf);
    //     }

    //     // Add sun emission
    //     emission += sunEmission * misWeight;
    // }

    // rayData->radiance = emission;
    rayData->distance = RayMax;
    rayData->shouldTerminate = true;
}

extern "C" __global__ void __miss__shadow()
{
}