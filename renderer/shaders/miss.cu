#include "OptixShaderCommon.h"
#include "SystemParameter.h"
#include "ShaderDebugUtils.h"
#include "Sampler.h"
#include "Restir.h"

extern "C" __constant__ SystemParameter sysParam;

extern "C" __global__ void __miss__radiance()
{
    RayData *rayData = (RayData *)mergePointer(optixGetPayload_0(), optixGetPayload_1());

    Int2 pixelPosition = Int2(optixGetLaunchIndex());

    Float3 emission = Float3(0);

    const bool enableReSTIR = true && rayData->depth == 0;
    if (enableReSTIR)
    {
        StoreDIReservoir(EmptyDIReservoir(), pixelPosition);
    }

    if (rayData->depth == 0)
    {
        Store2DFloat4(Float4(1.0f), sysParam.albedoBuffer, pixelPosition);
        Store2DFloat1((float)(0xFFFF), sysParam.materialBuffer, pixelPosition);
        Store2DFloat4(Float4(0.0f, -1.0f, 0.0f, 0.0f), sysParam.normalRoughnessBuffer, pixelPosition);
        Store2DFloat4(Float4(0.0f, -1.0f, 0.0f, 0.0f), sysParam.geoNormalThinfilmBuffer, pixelPosition);
        Store2DFloat4(Float4(0.0f, 0.0f, 0.0f, 0.0f), sysParam.materialParameterBuffer, pixelPosition);
    }

    // if (rayData->isLastBounceDiffuse)
    // {
    //     rayData->distance = RayMax;
    //     rayData->shouldTerminate = true;
    //     return;
    // }

    const Int2 &skyRes = sysParam.skyRes;
    const Int2 &sunRes = sysParam.sunRes;
    const int skySize = skyRes.x * skyRes.y;
    const float &accumulatedSkyLuminance = sysParam.accumulatedSkyLuminance;
    const float sunAngle = 0.51f; // angular diagram in degrees
    const float sunAngleCosThetaMax = cosf(sunAngle * M_PI / 180.0f / 2.0f);

    const Float3 &rayDir = rayData->wi;

    // Map the ray diretcion to uv
    Float2 uv;

    // Sky
    {
        // Map the ray diretcion to uv
        uv = EqualAreaMap(rayDir);
        Float3 skyEmission = SampleBicubicSmoothStep<Load2DFuncFloat4<Float3>, Float3, BoundaryFuncRepeatXClampY>(sysParam.skyBuffer, uv, skyRes);

        // Blend the sky color with mist
        Float3 mistColor = Float3(accumulatedSkyLuminance / skySize); // Average color of elevation=0
        float blenderFactor = clampf((rayDir.y + 0.4f) * (1.0f / 0.5f));

        float misWeight = 1.0f;

        // Add sky emission
        emission += smoothstep3f(mistColor, skyEmission, blenderFactor) * misWeight;
    }

    // Is the ray hitting the solar disk, then compute the sun emission
    bool hitDisk = EqualAreaMapCone(uv, sysParam.sunDir, rayDir, sunAngleCosThetaMax);
    if (hitDisk)
    {
        Int2 sunIdx((int)(uv.x * sunRes.x), (int)(uv.y * sunRes.y));

        // Wrapping around on X dimension
        if (sunIdx.x >= sunRes.x)
        {
            sunIdx.x %= sunRes.x;
        }
        if (sunIdx.x < 0)
        {
            sunIdx.x = sunRes.x - (-sunIdx.x) % sunRes.x;
        }
        Float3 sunEmission = Load2DFloat4(sysParam.sunBuffer, sunIdx).xyz;

        float misWeight = 1.0f;

        // Add sun emission
        emission += sunEmission * misWeight;
    }

    rayData->radiance = emission;
    rayData->distance = RayMax;
    rayData->shouldTerminate = true;
}

extern "C" __global__ void __miss__bsdf_light()
{
    ShadowRayData *rayData = (ShadowRayData *)mergePointer(optixGetPayload_0(), optixGetPayload_1());
    rayData->lightIdx = SkyLightIndex;
}

extern "C" __global__ void __miss__visibility()
{
    bool *isVisible = (bool *)mergePointer(optixGetPayload_0(), optixGetPayload_1());
    *isVisible = true;
}
