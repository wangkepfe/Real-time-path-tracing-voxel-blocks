#pragma once

#include "RestirCommon.h"
#include "OptixShaderCommon.h"
#include "Bsdf.h"

extern "C" __constant__ SystemParameter sysParam;

static const unsigned int RESERVOIR_BLOCK_SIZE = 16;

INL_DEVICE void StoreDIReservoir(const DIReservoir reservoir, Int2 reservoirPosition)
{
    unsigned int pointer = (reservoirPosition.x + reservoirPosition.y * sysParam.camera.resolution.x) + (sysParam.iterationIndex % 2) * sysParam.camera.resolution.x * sysParam.camera.resolution.y;
    sysParam.reservoirBuffer[pointer] = reservoir;
}

INL_DEVICE DIReservoir LoadDIReservoir(Int2 reservoirPosition)
{
    unsigned int pointer = (reservoirPosition.x + reservoirPosition.y * sysParam.camera.resolution.x) + ((sysParam.iterationIndex + 1) % 2) * sysParam.camera.resolution.x * sysParam.camera.resolution.y;
    return sysParam.reservoirBuffer[pointer];
}

INL_DEVICE DIReservoir EmptyDIReservoir()
{
    DIReservoir s;
    s.lightData = 0;
    s.uvData = 0;
    s.targetPdf = 0;
    s.weightSum = 0;
    s.M = 0;
    return s;
}

INL_DEVICE bool IsValidDIReservoir(const DIReservoir reservoir)
{
    return reservoir.lightData != 0;
}

INL_DEVICE unsigned int GetDIReservoirLightIndex(const DIReservoir reservoir)
{
    return reservoir.lightData & DIReservoir_LightIndexMask;
}

INL_DEVICE Float2 GetDIReservoirSampleUV(const DIReservoir reservoir)
{
    return Float2(reservoir.uvData & 0xffff, reservoir.uvData >> 16) / float(0xffff);
}

INL_DEVICE float GetDIReservoirInvPdf(const DIReservoir reservoir)
{
    return reservoir.weightSum;
}

// Adds a new, non-reservoir light sample into the reservoir, returns true if this sample was selected.
// Algorithm (3) from the ReSTIR paper, Streaming RIS using weighted reservoir sampling.
INL_DEVICE bool StreamSample(
    DIReservoir &reservoir,
    unsigned int lightIndex,
    Float2 uv,
    float random,
    float targetPdf,
    float invSourcePdf)
{
    // What's the current weight
    float risWeight = targetPdf * invSourcePdf;

    // Add one sample to the counter
    reservoir.M += 1;

    // Update the weight sum
    reservoir.weightSum += risWeight;

    // Decide if we will randomly pick this sample
    bool selectSample = (random * reservoir.weightSum < risWeight);

    // If we did select this sample, update the relevant data.
    // New samples don't have visibility or age information, we can skip that.
    if (selectSample)
    {
        reservoir.lightData = lightIndex | DIReservoir_LightValidBit;
        reservoir.uvData = unsigned int(saturate(uv.x) * 0xffff) | (unsigned int(saturate(uv.y) * 0xffff) << 16);
        reservoir.targetPdf = targetPdf;
    }

    return selectSample;
}

// Adds `newReservoir` into `reservoir`, returns true if the new reservoir's sample was selected.
// This is a very general form, allowing input parameters to specfiy normalization and targetPdf
// rather than computing them from `newReservoir`.  Named "internal" since these parameters take
// different meanings (e.g., in CombineDIReservoirs() or StreamNeighborWithPairwiseMIS())
INL_DEVICE bool InternalSimpleResample(
    DIReservoir &reservoir,
    const DIReservoir newReservoir,
    float random,
    float targetPdf = 1.0f,           // Usually closely related to the sample normalization,
    float sampleNormalization = 1.0f, //     typically off by some multiplicative factor
    float sampleM = 1.0f              // In its most basic form, should be newReservoir.M
)
{
    // What's the current weight (times any prior-step RIS normalization factor)
    float risWeight = targetPdf * sampleNormalization;

    // Our *effective* candidate pool is the sum of our candidates plus those of our neighbors
    reservoir.M += sampleM;

    // Update the weight sum
    reservoir.weightSum += risWeight;

    // Decide if we will randomly pick this sample
    bool selectSample = (random * reservoir.weightSum < risWeight);

    // If we did select this sample, update the relevant data
    if (selectSample)
    {
        reservoir.lightData = newReservoir.lightData;
        reservoir.uvData = newReservoir.uvData;
        reservoir.targetPdf = targetPdf;
        // reservoir.visibility = newReservoir.visibility;
        // reservoir.spatialDistance = newReservoir.spatialDistance;
        // reservoir.age = newReservoir.age;
    }

    return selectSample;
}

// Adds `newReservoir` into `reservoir`, returns true if the new reservoir's sample was selected.
// Algorithm (4) from the ReSTIR paper, Combining the streams of multiple reservoirs.
// Normalization - Equation (6) - is postponed until all reservoirs are combined.
INL_DEVICE bool CombineDIReservoirs(
    DIReservoir &reservoir,
    const DIReservoir newReservoir,
    float random,
    float targetPdf)
{
    return InternalSimpleResample(
        reservoir,
        newReservoir,
        random,
        targetPdf,
        newReservoir.weightSum * newReservoir.M,
        newReservoir.M);
}

// Performs normalization of the reservoir after streaming. Equation (6) from the ReSTIR paper.
INL_DEVICE void FinalizeResampling(
    DIReservoir &reservoir,
    float normalizationNumerator,
    float normalizationDenominator)
{
    float denominator = reservoir.targetPdf * normalizationDenominator;

    reservoir.weightSum = (denominator == 0.0) ? 0.0 : (reservoir.weightSum * normalizationNumerator) / denominator;
}

INL_DEVICE float GetSurfaceBrdfPdf(Surface surface, Float3 wi)
{
    Float3 f;
    float pdf;

    UberBSDFEvaluate(surface.normal, surface.geoNormal, wi, surface.wo, surface.albedo, Float3(0.0278f), fmaxf(surface.roughness, 0.01f), f, pdf);

    return pdf;
}

INL_DEVICE bool GetSurfaceBrdfSample(Surface surface, Float3 u, Float3 &wi, float &pdf)
{
    Float3 bsdfOverPdf;

    UberBSDFSample(u, surface.normal, surface.geoNormal, surface.wo, surface.albedo, Float3(0.0278f), fmaxf(surface.roughness, 0.01f), wi, bsdfOverPdf, pdf);

    return pdf > 0.0f;
}

// Computes the weight of the given light samples when the given surface is
// shaded using that light sample. Exact or approximate BRDF evaluation can be
// used to compute the weight. ReSTIR will converge to a correct lighting result
// even if all samples have a fixed weight of 1.0, but that will be very noisy.
// Scaling of the weights can be arbitrary, as long as it's consistent
// between all lights and surfaces.
INL_DEVICE float GetLightSampleTargetPdfForSurface(LightSample lightSample, Surface surface)
{
    if (lightSample.solidAnglePdf <= 0 || lightSample.lightType == LightTypeInvalid)
    {
        return 0.0f;
    }

    Float3 wi = (lightSample.lightType == LightTypeLocalTriangle) ? normalize(lightSample.position - surface.pos) : lightSample.position;

    Float3 f;
    float pdf;

    UberBSDFEvaluate(surface.normal, surface.geoNormal, wi, surface.wo, surface.albedo, Float3(0.0278f), fmaxf(surface.roughness, 0.01f), f, pdf);

    Float3 reflectedRadiance = lightSample.radiance * f * abs(dot(wi, surface.normal)) / lightSample.solidAnglePdf;

    return luminance(reflectedRadiance);
}

// Heuristic to determine a max visibility ray length from a PDF wrt. solid angle.
INL_DEVICE float BrdfMaxDistanceFromPdf(float brdfCutoff, float pdf)
{
    const float kRayTMax = 3.402823466e+38F; // FLT_MAX
    return brdfCutoff > 0.f ? sqrt((1.f / brdfCutoff - 1.f) * pdf) : kRayTMax;
}

// Helper function to create a LightSample for the sun.
INL_DEVICE LightSample createSunLightSample(int sampledSunIdx)
{
    // Convert the 1D index into a 2D coordinate using sysParam.sunRes.
    Int2 sunIdx;
    sunIdx.x = sampledSunIdx % sysParam.sunRes.x;
    sunIdx.y = sampledSunIdx / sysParam.sunRes.x;

    // Compute normalized UV coordinates with a half-pixel offset.
    Float2 uv((sunIdx.x + 0.5f) / float(sysParam.sunRes.x),
              (sunIdx.y + 0.5f) / float(sysParam.sunRes.y));

    // Sun parameters.
    const float sunAngle = 0.51f; // angular diameter in degrees.
    const float sunAngleCosThetaMax = cosf(sunAngle * M_PI / 180.0f / 2.0f);

    // Compute the solid angle PDF.
    float solidAnglePdf = (sysParam.sunRes.x * sysParam.sunRes.y) /
                          (TWO_PI * (1.0f - sunAngleCosThetaMax));

    // Map the UV to a ray direction using an equal-area cone mapping.
    Float3 rayDir = EqualAreaMapCone(sysParam.sunDir, uv.x, uv.y, sunAngleCosThetaMax);

    // Retrieve the sun emission from the sun buffer.
    Float3 sunEmission = Load2DFloat4(sysParam.sunBuffer, sunIdx).xyz;

    // Fill and return the LightSample.
    LightSample sample;
    sample.position = rayDir;
    sample.radiance = sunEmission;
    sample.solidAnglePdf = solidAnglePdf;
    sample.lightType = LightTypeSun;
    return sample;
}

// Helper function to create a LightSample for the sky.
INL_DEVICE LightSample createSkyLightSample(int sampledSkyIdx)
{
    // Convert the 1D index into a 2D coordinate using sysParam.skyRes.
    Int2 skyIdx;
    skyIdx.x = sampledSkyIdx % sysParam.skyRes.x;
    skyIdx.y = sampledSkyIdx / sysParam.skyRes.x;

    // Compute normalized UV coordinates with a half-pixel offset.
    Float2 uv((skyIdx.x + 0.5f) / float(sysParam.skyRes.x),
              (skyIdx.y + 0.5f) / float(sysParam.skyRes.y));

    // Compute the solid angle PDF.
    float solidAnglePdf = (sysParam.skyRes.x * sysParam.skyRes.y) / TWO_PI;

    // Map the UV to a ray direction using an equal-area mapping.
    Float3 rayDir = EqualAreaMap(uv.x, uv.y);

    // Retrieve the sky emission from the sky buffer.
    Float3 skyEmission = Load2DFloat4(sysParam.skyBuffer, skyIdx).xyz;

    // Fill and return the LightSample.
    LightSample sample;
    sample.position = rayDir;
    sample.radiance = skyEmission;
    sample.solidAnglePdf = solidAnglePdf;
    sample.lightType = LightTypeSky;
    return sample;
}

// Computes the multi importance sampling pdf for brdf and light sample. For light and BRDF PDFs wrt solid angle, blend between the two. lightSelectionPdf is a dimensionless selection pdf
INL_DEVICE float LightBrdfMisWeight(Surface surface, LightSample lightSample, float lightSelectionPdf, float lightMisWeight, bool isEnvironmentMap, float brdfMisWeight, float brdfCutoff)
{
    float lightSolidAnglePdf = lightSample.solidAnglePdf;

    if (brdfMisWeight == 0.0f || lightSolidAnglePdf <= 0.0f || isinf(lightSolidAnglePdf) || isnan(lightSolidAnglePdf))
    {
        // BRDF samples disabled or we can't trace BRDF rays MIS with analytical lights
        return lightMisWeight * lightSelectionPdf;
    }

    Float3 lightDir;
    float lightDistance;
    if (lightSample.lightType == LightTypeSky || lightSample.lightType == LightTypeSun)
    {
        lightDir = lightSample.position;
        lightDistance = RayMax;
    }
    else
    {
        Float3 toLight = lightSample.position - surface.pos;
        lightDistance = length(toLight);
        lightDir = toLight / lightDistance;
    }

    // Compensate for ray shortening due to brdf cutoff, does not apply to environment map sampling
    float brdfPdf = GetSurfaceBrdfPdf(surface, lightDir);
    float maxDistance = BrdfMaxDistanceFromPdf(brdfCutoff, brdfPdf);
    if (!isEnvironmentMap && lightDistance > maxDistance)
    {
        brdfPdf = 0.0f;
    }

    // Convert light selection pdf (unitless) to a solid angle measurement
    float sourcePdfWrtSolidAngle = lightSelectionPdf * lightSolidAnglePdf;

    // MIS blending against solid angle pdfs.
    float blendedPdfWrtSolidangle = lightMisWeight * sourcePdfWrtSolidAngle + brdfMisWeight * brdfPdf;

    // Convert back, RTXDI divides shading again by this term later
    float blendedSourcePdf = blendedPdfWrtSolidangle / lightSolidAnglePdf;

    return blendedSourcePdf;
}

INL_DEVICE bool TraceVisibilityShadowRay(LightSample lightSample, Surface surface, float rayOffset = 0.0f)
{
    Float3 sampleDir = (lightSample.lightType == LightTypeLocalTriangle) ? normalize(lightSample.position - surface.pos) : lightSample.position;
    float maxDistance = (lightSample.lightType == LightTypeLocalTriangle) ? length(lightSample.position - surface.pos) - 0.01f - rayOffset : RayMax;

    bool isLightVisible = false;
    UInt2 visibilityRayPayload = splitPointer(&isLightVisible);
    Float3 shadowRayOrig = surface.thinfilm ? (dot(sampleDir, surface.normal) > 0.0f ? surface.pos : surface.backfacePos) : surface.pos;
    optixTrace(sysParam.topObject,
               (float3)shadowRayOrig, (float3)sampleDir,
               rayOffset, maxDistance, 0.0f, // tmin, tmax, time
               OptixVisibilityMask(0xFE), OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               0, 2, 2,
               visibilityRayPayload.x, visibilityRayPayload.y);

    return isLightVisible;
}

INL_DEVICE Int2 ClampSamplePositionIntoView(Int2 pixelPosition)
{
    int width = int(sysParam.camera.resolution.x);
    int height = int(sysParam.camera.resolution.y);

    // Reflect the position across the screen edges. Compared to simple clamping, this prevents the spread of colorful blobs from screen edges.
    if (pixelPosition.x < 0)
        pixelPosition.x = -pixelPosition.x;
    if (pixelPosition.y < 0)
        pixelPosition.y = -pixelPosition.y;
    if (pixelPosition.x >= width)
        pixelPosition.x = 2 * width - pixelPosition.x - 1;
    if (pixelPosition.y >= height)
        pixelPosition.y = 2 * height - pixelPosition.y - 1;

    return pixelPosition;
}

INL_DEVICE bool GetPrevSurface(Surface &surface, Int2 pixelPosition)
{
    if (pixelPosition.x < 0 || pixelPosition.y < 0 || pixelPosition.x >= sysParam.prevCamera.resolution.x || pixelPosition.y >= sysParam.prevCamera.resolution.y)
        return false;

    surface.depth = Load2DFloat1(sysParam.prevDepthBuffer, pixelPosition);

    if (surface.depth == RayMax)
        return false;

    Float4 normalRoughness = Load2DFloat4(sysParam.prevNormalRoughnessBuffer, pixelPosition);
    Float4 geoNormalThinfilm = Load2DFloat4(sysParam.prevGeoNormalThinfilmBuffer, pixelPosition);

    Float3 viewDir = sysParam.prevCamera.uvToViewDirection(Float2(pixelPosition.x, pixelPosition.y) * sysParam.prevCamera.inversedResolution);
    surface.pos = sysParam.prevCamera.pos + viewDir * surface.depth;
    surface.backfacePos = surface.pos;
    surface.wo = normalize(sysParam.prevCamera.pos - surface.pos);

    surface.normal = normalRoughness.xyz;
    surface.geoNormal = geoNormalThinfilm.xyz;
    surface.albedo = Load2DFloat4(sysParam.prevAlbedoBuffer, pixelPosition).xyz;
    surface.roughness = normalRoughness.w;
    surface.thinfilm = (geoNormalThinfilm.w == 1.0f);

    return true;
}

INL_DEVICE LightSample GetLightSampleFromReservoir(const DIReservoir &reservoir, const Surface &surface)
{
    LightSample ls = {};
    unsigned int lightIndex = GetDIReservoirLightIndex(reservoir);
    Float2 uv = GetDIReservoirSampleUV(reservoir);
    if (lightIndex == SkyLightIndex)
    {
        int x = int(uv.x * sysParam.skyRes.x);
        int y = int(uv.y * sysParam.skyRes.y);
        x = clampi(x, 0, sysParam.skyRes.x - 1);
        y = clampi(y, 0, sysParam.skyRes.y - 1);
        int sampledSkyIdx = y * sysParam.skyRes.x + x;
        ls = createSkyLightSample(sampledSkyIdx);
    }
    else if (lightIndex == SunLightIndex)
    {
        int x = int(uv.x * sysParam.sunRes.x);
        int y = int(uv.y * sysParam.sunRes.y);
        x = clampi(x, 0, sysParam.sunRes.x - 1);
        y = clampi(y, 0, sysParam.sunRes.y - 1);
        int sampledSunIdx = y * sysParam.sunRes.x + x;
        ls = createSunLightSample(sampledSunIdx);
    }
    else
    {
        LightInfo lightInfo = sysParam.lights[lightIndex];
        TriangleLight triLight = TriangleLight::Create(lightInfo);
        ls = triLight.calcSample(uv, surface.pos);
    }
    return ls;
}