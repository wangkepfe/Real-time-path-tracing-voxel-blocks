#pragma once

#include "OptixShaderCommon.h"
#include "Bsdf.h"

extern "C" __constant__ SystemParameter sysParam;

// This structure represents a single light reservoir that stores the weights, the sample ref,
// sample count (M), and visibility for reuse. It can be serialized into PackedDIReservoir for storage.
struct DIReservoir
{
    // Light index (bits 0..30) and validity bit (31)
    unsigned int lightData;

    // Sample UV encoded in 16-bit fixed point format
    unsigned int uvData;

    // Overloaded: represents RIS weight sum during streaming,
    // then reservoir weight (inverse PDF) after FinalizeResampling
    float weightSum;

    // Target PDF of the selected sample
    float targetPdf;

    // Number of samples considered for this reservoir (pairwise MIS makes this a float)
    float M;

    // Visibility information stored in the reservoir for reuse
    unsigned int packedVisibility;

    // Screen-space distance between the current location of the reservoir
    // and the location where the visibility information was generated,
    // minus the motion vectors applied in temporal resampling
    Int2 spatialDistance;

    // How many frames ago the visibility information was generated
    unsigned int age;

    // Cannonical weight when using pairwise MIS (ignored except during pairwise MIS computations)
    float canonicalWeight;
};

struct PackedDIReservoir
{
    unsigned int lightData;
    unsigned int uvData;
    unsigned int mVisibility;
    unsigned int distanceAge;
    float targetPdf;
    float weight;
};

struct ReservoirBufferParameters
{
    unsigned int reservoirBlockRowPitch;
    unsigned int reservoirArrayPitch;
    unsigned int pad1;
    unsigned int pad2;
};

struct SampleParameters
{
    unsigned int numLocalLightSamples;
    unsigned int numSunLightSamples;
    unsigned int numSkyLightSamples;
    unsigned int numBrdfSamples;

    unsigned int numMisSamples;

    float localLightMisWeight;
    float sunLightMisWeight;
    float skyLightMisWeight;
    float brdfMisWeight;

    float brdfCutoff;
};

// Encoding helper constants for PackedDIReservoir.mVisibility
static const unsigned int PackedDIReservoir_VisibilityMask = 0x3ffff;
static const unsigned int PackedDIReservoir_VisibilityChannelMax = 0x3f;
static const unsigned int PackedDIReservoir_VisibilityChannelShift = 6;
static const unsigned int PackedDIReservoir_MShift = 18;
static const unsigned int PackedDIReservoir_MaxM = 0x3fff;

// Encoding helper constants for PackedDIReservoir.distanceAge
static const unsigned int PackedDIReservoir_DistanceChannelBits = 8;
static const unsigned int PackedDIReservoir_DistanceXShift = 0;
static const unsigned int PackedDIReservoir_DistanceYShift = 8;
static const unsigned int PackedDIReservoir_AgeShift = 16;
static const unsigned int PackedDIReservoir_MaxAge = 0xff;
static const unsigned int PackedDIReservoir_DistanceMask = (1u << PackedDIReservoir_DistanceChannelBits) - 1;
static const int PackedDIReservoir_MaxDistance = int((1u << (PackedDIReservoir_DistanceChannelBits - 1)) - 1);

static const unsigned int RESERVOIR_BLOCK_SIZE = 16;
static const float kMinRoughness = 0.05f;

struct ReSTIRDIParameters
{
    unsigned int numLocalLightSamples;
    unsigned int numSunLightSamples;
    unsigned int numSkyLightSamples;
    unsigned int numBrdfSamples;

    float brdfCutoff;
    unsigned int enableInitialVisibility;
    unsigned int environmentMapImportanceSampling;

    float temporalDepthThreshold;
    float temporalNormalThreshold;
    unsigned int maxHistoryLength;

    unsigned int enablePermutationSampling;
    float permutationSamplingThreshold;

    unsigned int enableBoilingFilter;
    float boilingFilterStrength;

    unsigned int discardInvisibleSamples;

    float spatialDepthThreshold;
    float spatialNormalThreshold;
    unsigned int numSpatialSamples;

    unsigned int numDisocclusionBoostSamples;
    float spatialSamplingRadius;

    unsigned int enableFinalVisibility;
    unsigned int reuseFinalVisibility;
    unsigned int finalVisibilityMaxAge;
    float finalVisibilityMaxDistance;

    unsigned int enableDenoiserInputPacking;
};

INL_DEVICE ReSTIRDIParameters GetDefaultReSTIRDIParams()
{
    ReSTIRDIParameters params = {};

    params.numLocalLightSamples = 8;
    params.numSunLightSamples = 1;
    params.numSkyLightSamples = 1;
    params.numBrdfSamples = 1;

    params.brdfCutoff = 0.0001f;
    params.enableInitialVisibility = true;
    params.environmentMapImportanceSampling = 1;

    params.temporalDepthThreshold = 0.1f;
    params.temporalNormalThreshold = 0.5f;
    params.maxHistoryLength = 20;

    params.enablePermutationSampling = true;
    params.permutationSamplingThreshold = 0.9f;

    params.enableBoilingFilter = true;
    params.boilingFilterStrength = 0.2f;

    params.discardInvisibleSamples = false;

    params.spatialDepthThreshold = 0.1f;
    params.spatialNormalThreshold = 0.5f;
    params.numSpatialSamples = 1;

    params.numDisocclusionBoostSamples = 8;
    params.spatialSamplingRadius = 32.0f;

    params.enableFinalVisibility = true;
    params.reuseFinalVisibility = true;
    params.finalVisibilityMaxAge = 4;
    params.finalVisibilityMaxDistance = 16.f;

    params.enableDenoiserInputPacking = false;

    return params;
}

INL_DEVICE PackedDIReservoir PackDIReservoir(const DIReservoir reservoir)
{
    Int2 clampedSpatialDistance = clamp2i(reservoir.spatialDistance, Int2(-PackedDIReservoir_MaxDistance), Int2(PackedDIReservoir_MaxDistance));
    unsigned int clampedAge = clampu(reservoir.age, 0u, PackedDIReservoir_MaxAge);

    PackedDIReservoir data;
    data.lightData = reservoir.lightData;
    data.uvData = reservoir.uvData;

    data.mVisibility = reservoir.packedVisibility | (min(unsigned int(reservoir.M), PackedDIReservoir_MaxM) << PackedDIReservoir_MShift);

    data.distanceAge =
        ((clampedSpatialDistance.x & PackedDIReservoir_DistanceMask) << PackedDIReservoir_DistanceXShift) | ((clampedSpatialDistance.y & PackedDIReservoir_DistanceMask) << PackedDIReservoir_DistanceYShift) | (clampedAge << PackedDIReservoir_AgeShift);

    data.targetPdf = reservoir.targetPdf;
    data.weight = reservoir.weightSum;

    return data;
}

INL_DEVICE unsigned int ReservoirPositionToPointer(
    ReservoirBufferParameters reservoirParams,
    UInt2 reservoirPosition,
    unsigned int reservoirArrayIndex)
{
    UInt2 blockIdx = reservoirPosition / RESERVOIR_BLOCK_SIZE;
    UInt2 positionInBlock = reservoirPosition % RESERVOIR_BLOCK_SIZE;

    return reservoirArrayIndex * reservoirParams.reservoirArrayPitch + blockIdx.y * reservoirParams.reservoirBlockRowPitch + blockIdx.x * (RESERVOIR_BLOCK_SIZE * RESERVOIR_BLOCK_SIZE) + positionInBlock.y * RESERVOIR_BLOCK_SIZE + positionInBlock.x;
}

INL_DEVICE void StoreDIReservoir(
    const DIReservoir reservoir,
    ReservoirBufferParameters reservoirParams,
    UInt2 reservoirPosition,
    unsigned int reservoirArrayIndex,
    PackedDIReservoir *LIGHT_RESERVOIR_BUFFER)
{
    unsigned int pointer = ReservoirPositionToPointer(reservoirParams, reservoirPosition, reservoirArrayIndex);
    LIGHT_RESERVOIR_BUFFER[pointer] = PackDIReservoir(reservoir);
}

INL_DEVICE DIReservoir EmptyDIReservoir()
{
    DIReservoir s;
    s.lightData = 0;
    s.uvData = 0;
    s.targetPdf = 0;
    s.weightSum = 0;
    s.M = 0;
    s.packedVisibility = 0;
    s.spatialDistance = Int2(0, 0);
    s.age = 0;
    s.canonicalWeight = 0;
    return s;
}

INL_DEVICE DIReservoir UnpackDIReservoir(PackedDIReservoir data)
{
    DIReservoir res;
    res.lightData = data.lightData;
    res.uvData = data.uvData;
    res.targetPdf = data.targetPdf;
    res.weightSum = data.weight;
    res.M = (data.mVisibility >> PackedDIReservoir_MShift) & PackedDIReservoir_MaxM;
    res.packedVisibility = data.mVisibility & PackedDIReservoir_VisibilityMask;
    // Sign extend the shift values
    res.spatialDistance.x = int(data.distanceAge << (32 - PackedDIReservoir_DistanceXShift - PackedDIReservoir_DistanceChannelBits)) >> (32 - PackedDIReservoir_DistanceChannelBits);
    res.spatialDistance.y = int(data.distanceAge << (32 - PackedDIReservoir_DistanceYShift - PackedDIReservoir_DistanceChannelBits)) >> (32 - PackedDIReservoir_DistanceChannelBits);
    res.age = (data.distanceAge >> PackedDIReservoir_AgeShift) & PackedDIReservoir_MaxAge;
    res.canonicalWeight = 0.0f;

    // Discard reservoirs that have Inf/NaN
    if (isinf(res.weightSum) || isnan(res.weightSum))
    {
        res = EmptyDIReservoir();
    }

    return res;
}

INL_DEVICE DIReservoir LoadDIReservoir(
    ReservoirBufferParameters reservoirParams,
    UInt2 reservoirPosition,
    unsigned int reservoirArrayIndex,
    PackedDIReservoir *LIGHT_RESERVOIR_BUFFER)
{
    unsigned int pointer = ReservoirPositionToPointer(reservoirParams, reservoirPosition, reservoirArrayIndex);
    return UnpackDIReservoir(LIGHT_RESERVOIR_BUFFER[pointer]);
}

INL_DEVICE void StoreVisibilityInDIReservoir(
    DIReservoir &reservoir,
    Float3 visibility,
    bool discardIfInvisible)
{
    reservoir.packedVisibility = unsigned int(saturate(visibility.x) * PackedDIReservoir_VisibilityChannelMax) | (unsigned int(saturate(visibility.y) * PackedDIReservoir_VisibilityChannelMax)) << PackedDIReservoir_VisibilityChannelShift | (unsigned int(saturate(visibility.z) * PackedDIReservoir_VisibilityChannelMax)) << (PackedDIReservoir_VisibilityChannelShift * 2);

    reservoir.spatialDistance = Int2(0, 0);
    reservoir.age = 0;

    if (discardIfInvisible && visibility.x == 0 && visibility.y == 0 && visibility.z == 0)
    {
        // Keep M for correct resampling, remove the actual sample
        reservoir.lightData = 0;
        reservoir.weightSum = 0;
    }
}

// Structure that groups the parameters for GetReservoirVisibility(...)
// Reusing final visibility reduces the number of high-quality shadow rays needed to shade
// the scene, at the cost of somewhat softer or laggier shadows.
struct VisibilityReuseParameters
{
    // Controls the maximum age of the final visibility term, measured in frames, that can be reused from the
    // previous frame(s). Higher values result in better performance.
    unsigned int maxAge;

    // Controls the maximum distance in screen space between the current pixel and the pixel that has
    // produced the final visibility term. The distance does not include the motion vectors.
    // Higher values result in better performance and softer shadows.
    float maxDistance;
};

INL_DEVICE bool GetDIReservoirVisibility(
    const DIReservoir reservoir,
    const VisibilityReuseParameters params,
    Float3 &o_visibility)
{
    if (reservoir.age > 0 &&
        reservoir.age <= params.maxAge &&
        length(Float2(reservoir.spatialDistance.x, reservoir.spatialDistance.y)) < params.maxDistance)
    {
        o_visibility.x = float(reservoir.packedVisibility & PackedDIReservoir_VisibilityChannelMax) / PackedDIReservoir_VisibilityChannelMax;
        o_visibility.y = float((reservoir.packedVisibility >> PackedDIReservoir_VisibilityChannelShift) & PackedDIReservoir_VisibilityChannelMax) / PackedDIReservoir_VisibilityChannelMax;
        o_visibility.z = float((reservoir.packedVisibility >> (PackedDIReservoir_VisibilityChannelShift * 2)) & PackedDIReservoir_VisibilityChannelMax) / PackedDIReservoir_VisibilityChannelMax;

        return true;
    }

    o_visibility = Float3(0, 0, 0);
    return false;
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
        reservoir.packedVisibility = newReservoir.packedVisibility;
        reservoir.spatialDistance = newReservoir.spatialDistance;
        reservoir.age = newReservoir.age;
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

// Sample parameters struct
// Defined so that so these can be compile time constants as defined by the user
// brdfCutoff Value in range [0,1] to determine how much to shorten BRDF rays. 0 to disable shortening
INL_DEVICE SampleParameters InitSampleParameters(ReSTIRDIParameters params)
{
    SampleParameters result;

    result.numLocalLightSamples = params.numLocalLightSamples;
    result.numSunLightSamples = params.numSunLightSamples;
    result.numSkyLightSamples = params.numSkyLightSamples;
    result.numBrdfSamples = params.numBrdfSamples;

    result.numMisSamples = result.numLocalLightSamples + result.numSunLightSamples + result.numSkyLightSamples + result.numBrdfSamples;

    result.localLightMisWeight = float(result.numLocalLightSamples) / result.numMisSamples;
    result.sunLightMisWeight = float(result.numSunLightSamples) / result.numMisSamples;
    result.skyLightMisWeight = float(result.numSkyLightSamples) / result.numMisSamples;
    result.brdfMisWeight = float(result.numBrdfSamples) / result.numMisSamples;

    result.brdfCutoff = params.brdfCutoff;

    return result;
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

    Float3 reflectedRadiance = lightSample.radiance * f;

    return luminance(reflectedRadiance) / lightSample.solidAnglePdf;
}

// Heuristic to determine a max visibility ray length from a PDF wrt. solid angle.
INL_DEVICE float BrdfMaxDistanceFromPdf(float brdfCutoff, float pdf)
{
    const float kRayTMax = 3.402823466e+38F; // FLT_MAX
    return brdfCutoff > 0.f ? sqrt((1.f / brdfCutoff - 1.f) * pdf) : kRayTMax;
}

// Computes the multi importance sampling pdf for brdf and light sample.
// For light and BRDF PDFs wrt solid angle, blend between the two.
//      lightSelectionPdf is a dimensionless selection pdf
INL_DEVICE float LightBrdfMisWeight(Surface surface, LightSample lightSample, float lightSelectionPdf, float lightMisWeight, bool isEnvironmentMap, SampleParameters sampleParams)
{
    float lightSolidAnglePdf = lightSample.solidAnglePdf;

    if (sampleParams.brdfMisWeight == 0.0f || lightSolidAnglePdf <= 0.0f || isinf(lightSolidAnglePdf) || isnan(lightSolidAnglePdf))
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
    float maxDistance = BrdfMaxDistanceFromPdf(sampleParams.brdfCutoff, brdfPdf);
    if (!isEnvironmentMap && lightDistance > maxDistance)
    {
        brdfPdf = 0.0f;
    }

    // Convert light selection pdf (unitless) to a solid angle measurement
    float sourcePdfWrtSolidAngle = lightSelectionPdf * lightSolidAnglePdf;

    // MIS blending against solid angle pdfs.
    float blendedPdfWrtSolidangle = lightMisWeight * sourcePdfWrtSolidAngle + sampleParams.brdfMisWeight * brdfPdf;

    // Convert back, RTXDI divides shading again by this term later
    float blendedSourcePdf = blendedPdfWrtSolidangle / lightSolidAnglePdf;

    // if (OPTIX_CENTER_PIXEL())
    // {
    //     OPTIX_DEBUG_PRINT(lightDir);
    //     OPTIX_DEBUG_PRINT(lightSelectionPdf);
    //     OPTIX_DEBUG_PRINT(sourcePdfWrtSolidAngle);
    //     OPTIX_DEBUG_PRINT(brdfPdf);
    //     OPTIX_DEBUG_PRINT(blendedPdfWrtSolidangle);
    //     OPTIX_DEBUG_PRINT(lightSolidAnglePdf);
    //     OPTIX_DEBUG_PRINT(blendedSourcePdf);
    // }

    return blendedSourcePdf;
}

INL_DEVICE DIReservoir SampleLightsForSurface(
    const SystemParameter &sysParam,
    int &randIdx,
    const Surface &surface,
    const SampleParameters &sampleParams,
    LightSample &outLightSample)
{
    outLightSample = LightSample{};

    // Local light
    DIReservoir localReservoir = EmptyDIReservoir();
    LightSample localSample = LightSample{};
    for (unsigned int i = 0; i < sampleParams.numLocalLightSamples; i++)
    {
        float sourcePdf;
        int lightIndex = sysParam.lightAliasTable->sample(rand(sysParam, randIdx), sourcePdf);
        LightInfo lightInfo = sysParam.lights[lightIndex];

        Float2 uv = rand2(sysParam, randIdx);

        TriangleLight triLight = TriangleLight::Create(lightInfo);
        LightSample candidateSample = triLight.calcSample(uv, surface.pos);

        float blendedSourcePdf = LightBrdfMisWeight(surface, candidateSample, sourcePdf, sampleParams.localLightMisWeight, false, sampleParams);
        float targetPdf = GetLightSampleTargetPdfForSurface(candidateSample, surface);
        float risRnd = rand(sysParam, randIdx);

        if (blendedSourcePdf != 0.0f)
        {
            bool selected = StreamSample(localReservoir, lightIndex, uv, risRnd, targetPdf, 1.0f / blendedSourcePdf);
            if (selected)
            {
                localSample = candidateSample;
            }
        }
    }
    FinalizeResampling(localReservoir, 1.0, sampleParams.numMisSamples);
    localReservoir.M = 1;

    // Sun
    DIReservoir sunLightReservoir = EmptyDIReservoir();
    LightSample sunLightSample = LightSample{};
    for (unsigned int i = 0; i < sampleParams.numSunLightSamples; i++)
    {
        float sourcePdf;
        int sampledSunIdx = sysParam.sunAliasTable->sample(rand(sysParam, randIdx), sourcePdf);
        Int2 sunIdx(sampledSunIdx % sysParam.sunRes.x, sampledSunIdx / sysParam.sunRes.x);
        Float2 uv = Float2((sunIdx.x + 0.5f) / sysParam.sunRes.x, (sunIdx.y + 0.5f) / sysParam.sunRes.y);

        const float sunAngle = 0.51f; // angular diagram in degrees
        const float sunAngleCosThetaMax = cosf(sunAngle * M_PI / 180.0f / 2.0f);
        float solidAnglePdf = (sysParam.sunRes.x * sysParam.sunRes.y) / (TWO_PI * (1.0f - sunAngleCosThetaMax));

        Float3 rayDir = EqualAreaMapCone(sysParam.sunDir, uv.x, uv.y, sunAngleCosThetaMax);
        Float3 sunEmission = Load2DFloat4(sysParam.sunBuffer, sunIdx).xyz;

        LightSample candidateSample;
        candidateSample.position = rayDir;
        candidateSample.radiance = sunEmission;
        candidateSample.solidAnglePdf = solidAnglePdf;
        candidateSample.lightType = LightTypeSun;

        float blendedSourcePdf = LightBrdfMisWeight(surface, candidateSample, sourcePdf, sampleParams.sunLightMisWeight, true, sampleParams);
        float targetPdf = GetLightSampleTargetPdfForSurface(candidateSample, surface);
        float risRnd = rand(sysParam, randIdx);

        bool selected = StreamSample(sunLightReservoir, SunLightIndex, uv, risRnd, targetPdf, 1.0f / blendedSourcePdf);
        if (selected)
        {
            sunLightSample = candidateSample;
        }
    }
    FinalizeResampling(sunLightReservoir, 1.0, sampleParams.numMisSamples);
    sunLightReservoir.M = 1;

    // Sky
    DIReservoir skyLightReservoir = EmptyDIReservoir();
    LightSample skyLightSample = LightSample{};
    for (unsigned int i = 0; i < sampleParams.numSkyLightSamples; i++)
    {
        float sourcePdf;
        int sampledSkyIdx = sysParam.skyAliasTable->sample(rand16bits(sysParam, randIdx), sourcePdf);
        Int2 skyIdx(sampledSkyIdx % sysParam.skyRes.x, sampledSkyIdx / sysParam.skyRes.x);
        Float2 uv = Float2((skyIdx.x + 0.5f) / (float)sysParam.skyRes.x, (skyIdx.y + 0.5f) / (float)sysParam.skyRes.y);

        float solidAnglePdf = (sysParam.skyRes.x * sysParam.skyRes.y) / TWO_PI;

        Float3 rayDir = EqualAreaMap(uv.x, uv.y);
        Float3 skyEmission = Load2DFloat4(sysParam.skyBuffer, skyIdx).xyz;

        LightSample candidateSample;
        candidateSample.position = rayDir;
        candidateSample.radiance = skyEmission;
        candidateSample.solidAnglePdf = solidAnglePdf;
        candidateSample.lightType = LightTypeSky;

        float blendedSourcePdf = LightBrdfMisWeight(surface, candidateSample, sourcePdf, sampleParams.skyLightMisWeight, true, sampleParams);
        float targetPdf = GetLightSampleTargetPdfForSurface(candidateSample, surface);
        float risRnd = rand(sysParam, randIdx);

        bool selected = StreamSample(skyLightReservoir, SkyLightIndex, uv, risRnd, targetPdf, 1.0f / blendedSourcePdf);

        if (selected)
        {
            skyLightSample = candidateSample;
        }
    }
    FinalizeResampling(skyLightReservoir, 1.0, sampleParams.numMisSamples);
    skyLightReservoir.M = 1;

    // BSDF sample
    DIReservoir brdfReservoir = EmptyDIReservoir();
    LightSample brdfSample = LightSample{};
    for (unsigned int i = 0; i < sampleParams.numBrdfSamples; ++i)
    {
        float lightSourcePdf = 0.0f;
        Float3 sampleDir;
        unsigned int lightIndex = InvalidLightIndex;
        Float2 uv = Float2(0, 0);
        LightSample candidateSample = LightSample{};

        float brdfPdf;
        if (GetSurfaceBrdfSample(surface, rand3(sysParam, randIdx), sampleDir, brdfPdf))
        {
            float maxDistance = BrdfMaxDistanceFromPdf(sampleParams.brdfCutoff, brdfPdf);

            ShadowRayData rayData;
            rayData.lightIdx = InvalidLightIndex;
            UInt2 payload = splitPointer(&rayData);

            Float3 shadowRayOrig = surface.thinfilm ? (dot(sampleDir, surface.normal) > 0.0f ? surface.pos : surface.backfacePos) : surface.pos;
            optixTrace(sysParam.topObject,
                       (float3)shadowRayOrig, (float3)sampleDir,
                       0.0f, maxDistance, 0.0f, // tmin, tmax, time
                       OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                       1, 2, 1,
                       payload.x, payload.y);

            if (rayData.lightIdx == InvalidLightIndex)
            {
                lightIndex = InvalidLightIndex;
            }
            else if (rayData.lightIdx == SkyLightIndex)
            {
                const float sunAngle = 0.51f; // angular diagram in degrees
                const float sunAngleCosThetaMax = cosf(sunAngle * M_PI / 180.0f / 2.0f);
                bool hitDisk = EqualAreaMapCone(uv, sysParam.sunDir, sampleDir, sunAngleCosThetaMax);
                if (hitDisk)
                {
                    lightIndex = SunLightIndex;

                    Int2 sunIdx((int)(uv.x * sysParam.sunRes.x - 0.5f), (int)(uv.y * sysParam.sunRes.y - 0.5f));
                    // Wrapping around on X dimension, clamp on Y dimension
                    if (sunIdx.x >= sysParam.sunRes.x)
                    {
                        sunIdx.x %= sysParam.sunRes.x;
                    }
                    if (sunIdx.x < 0)
                    {
                        sunIdx.x = sysParam.sunRes.x - (-sunIdx.x) % sysParam.sunRes.x;
                    }
                    sunIdx.y = clampi(sunIdx.y, 0, sysParam.sunRes.y - 1);
                    Float3 sunEmission = Load2DFloat4(sysParam.sunBuffer, sunIdx).xyz;

                    float solidAnglePdf = (sysParam.sunRes.x * sysParam.sunRes.y) / (TWO_PI * (1.0f - sunAngleCosThetaMax));

                    candidateSample.position = sampleDir;
                    candidateSample.radiance = sunEmission;
                    candidateSample.solidAnglePdf = solidAnglePdf;
                    candidateSample.lightType = LightTypeSun;

                    lightSourcePdf = sysParam.sunAliasTable->PMF(sunIdx.y * sysParam.sunRes.x + sunIdx.x);
                }
                else
                {
                    lightIndex = SkyLightIndex;

                    uv = EqualAreaMap(sampleDir);
                    Int2 skyIdx((int)(uv.x * sysParam.skyRes.x - 0.5f), (int)(uv.y * sysParam.skyRes.y - 0.5f));
                    clamp2i(skyIdx, Int2(0), sysParam.skyRes - 1);
                    Float3 skyEmission = SampleBicubicSmoothStep<Load2DFuncFloat4<Float3>, Float3, BoundaryFuncRepeatXClampY>(sysParam.skyBuffer, uv, sysParam.skyRes);

                    float solidAnglePdf = (sysParam.skyRes.x * sysParam.skyRes.y) / TWO_PI;

                    candidateSample.position = sampleDir;
                    candidateSample.radiance = skyEmission;
                    candidateSample.solidAnglePdf = solidAnglePdf;
                    candidateSample.lightType = LightTypeSky;

                    lightSourcePdf = sysParam.skyAliasTable->PMF(skyIdx.y * sysParam.skyRes.x + skyIdx.x);

                    // if (OPTIX_CENTER_PIXEL())
                    // {
                    //     OPTIX_DEBUG_PRINT(lightIndex);
                    //     OPTIX_DEBUG_PRINT(surface.wo);
                    //     OPTIX_DEBUG_PRINT(surface.normal);
                    //     OPTIX_DEBUG_PRINT(sampleDir);
                    //     OPTIX_DEBUG_PRINT(uv);
                    //     OPTIX_DEBUG_PRINT(skyIdx);
                    //     OPTIX_DEBUG_PRINT(skyEmission);
                    //     OPTIX_DEBUG_PRINT(solidAnglePdf);
                    //     OPTIX_DEBUG_PRINT(lightSourcePdf);
                    // }
                }
            }
            else
            {
                lightIndex = rayData.lightIdx;

                LightInfo lightInfo = sysParam.lights[lightIndex];

                TriangleLight triLight = TriangleLight::Create(lightInfo);

                uv = InverseTriangleSample(rayData.bary);
                candidateSample = triLight.calcSample(uv, surface.pos);

                if (sampleParams.brdfCutoff > 0.f)
                {
                    float lightDistance = length(candidateSample.position - surface.pos);

                    float maxDistance = BrdfMaxDistanceFromPdf(sampleParams.brdfCutoff, brdfPdf);
                    if (lightDistance > maxDistance)
                    {
                        lightIndex = InvalidLightIndex;
                    }
                }

                if (lightIndex != InvalidLightIndex)
                {
                    lightSourcePdf = sysParam.lightAliasTable->PMF(lightIndex);
                }
            }
        }

        if (lightSourcePdf == 0.0f)
        {
            continue;
        }

        float targetPdf = GetLightSampleTargetPdfForSurface(candidateSample, surface);

        bool isEnvMapSample = lightIndex == SkyLightIndex || lightIndex == SunLightIndex;
        float misWeight = (lightIndex == SkyLightIndex) ? sampleParams.skyLightMisWeight : ((lightIndex == SunLightIndex) ? sampleParams.sunLightMisWeight : sampleParams.localLightMisWeight);

        float blendedSourcePdf = LightBrdfMisWeight(surface, candidateSample, lightSourcePdf, misWeight, isEnvMapSample, sampleParams);
        float risRnd = rand(sysParam, randIdx);

        bool selected = StreamSample(brdfReservoir, lightIndex, uv, risRnd, targetPdf, 1.0f / blendedSourcePdf);
        if (selected)
        {
            brdfSample = candidateSample;
        }

        // if (OPTIX_CENTER_PIXEL())
        // {
        //     OPTIX_DEBUG_PRINT(targetPdf);
        //     OPTIX_DEBUG_PRINT(isEnvMapSample);
        //     OPTIX_DEBUG_PRINT(misWeight);
        //     OPTIX_DEBUG_PRINT(blendedSourcePdf);
        //     OPTIX_DEBUG_PRINT(risRnd);
        //     OPTIX_DEBUG_PRINT(selected);
        // }
    }
    FinalizeResampling(brdfReservoir, 1.0f, sampleParams.numMisSamples);
    brdfReservoir.M = 1;

    // Merge samples
    DIReservoir state = EmptyDIReservoir();
    CombineDIReservoirs(state, localReservoir, 0.5f, localReservoir.targetPdf);
    bool selectSunLight = CombineDIReservoirs(state, sunLightReservoir, rand(sysParam, randIdx), sunLightReservoir.targetPdf);
    bool selectSkyLight = CombineDIReservoirs(state, skyLightReservoir, rand(sysParam, randIdx), skyLightReservoir.targetPdf);
    bool selectBrdf = CombineDIReservoirs(state, brdfReservoir, rand(sysParam, randIdx), brdfReservoir.targetPdf);

    FinalizeResampling(state, 1.0f, 1.0f);
    state.M = 1;

    if (selectBrdf)
    {
        outLightSample = brdfSample;
    }
    else if (selectSkyLight)
    {
        outLightSample = skyLightSample;
    }
    else if (selectSunLight)
    {
        outLightSample = sunLightSample;
    }
    else
    {
        outLightSample = localSample;
    }

    // if (OPTIX_CENTER_PIXEL())
    // {
    //     OPTIX_DEBUG_PRINT(lightIndex);
    // }

    return state;
}
