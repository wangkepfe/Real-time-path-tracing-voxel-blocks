#pragma once

#include "RestirCommon.h"
#include "OptixShaderCommon.h"
#include "Bsdf.h"

extern "C" __constant__ SystemParameter sysParam;

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

// A structure that groups the application-provided settings for spatio-temporal resampling.
struct DISpatioTemporalResamplingParameters
{
    // Screen-space motion vector, computed as (previousPosition - currentPosition).
    // The X and Y components are measured in pixels.
    // The Z component is in linear depth units.
    Float3 screenSpaceMotion;

    // The index of the reservoir buffer to pull the temporal samples from.
    unsigned int sourceBufferIndex;

    // Maximum history length for temporal reuse, measured in frames.
    // Higher values result in more stable and high quality sampling, at the cost of slow reaction to changes.
    unsigned int maxHistoryLength;

    // Controls the bias correction math for temporal reuse. Depending on the setting, it can add
    // some shader cost and one approximate shadow ray per pixel (or per two pixels if checkerboard sampling is enabled).
    // Ideally, these rays should be traced through the previous frame's BVH to get fully unbiased results.
    unsigned int biasCorrectionMode;

    // Surface depth similarity threshold for temporal reuse.
    // If the previous frame surface's depth is within this threshold from the current frame surface's depth,
    // the surfaces are considered similar. The threshold is relative, i.e. 0.1 means 10% of the current depth.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float depthThreshold;

    // Surface normal similarity threshold for temporal reuse.
    // If the dot product of two surfaces' normals is higher than this threshold, the surfaces are considered similar.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float normalThreshold;

    // Number of neighbor pixels considered for resampling (1-32)
    // Some of the may be skipped if they fail the surface similarity test.
    unsigned int numSamples;

    // Number of neighbor pixels considered when there is no temporal surface (1-32)
    // Setting this parameter equal or lower than `numSpatialSamples` effectively
    // disables the disocclusion boost.
    unsigned int numDisocclusionBoostSamples;

    // Screen-space radius for spatial resampling, measured in pixels.
    float samplingRadius;

    // Allows the temporal resampling logic to skip the bias correction ray trace for light samples
    // reused from the previous frame. Only safe to use when invisible light samples are discarded
    // on the previous frame, then any sample coming from the previous frame can be assumed visible.
    bool enableVisibilityShortcut;

    // Enables permuting the pixels sampled from the previous frame in order to add temporal
    // variation to the output signal and make it more denoiser friendly.
    bool enablePermutationSampling;

    // Enables the comparison of surface materials before taking a surface into resampling.
    bool enableMaterialSimilarityTest;

    // Prevents samples which are from the current frame or have no reasonable temporal history merged being spread to neighbors
    bool discountNaiveSamples;

    // Random number for permutation sampling that is the same for all pixels in the frame
    unsigned int uniformRandomNumber;
};

struct TemporalResamplingParameters
{
    float temporalDepthThreshold;
    float temporalNormalThreshold;
    uint32_t maxHistoryLength;

    uint32_t enablePermutationSampling;
    float permutationSamplingThreshold;
    uint32_t enableBoilingFilter;
    float boilingFilterStrength;

    uint32_t discardInvisibleSamples;
    uint32_t uniformRandomNumber;
    uint32_t pad2;
    uint32_t pad3;
};

struct SpatialResamplingParameters
{
    float spatialDepthThreshold;
    float spatialNormalThreshold;
    uint32_t numSpatialSamples;

    uint32_t numDisocclusionBoostSamples;
    float spatialSamplingRadius;
    uint32_t neighborOffsetMask;
    uint32_t discountNaiveSamples;
};

INL_DEVICE TemporalResamplingParameters getDefaultReSTIRDITemporalResamplingParams()
{
    TemporalResamplingParameters params = {};
    params.boilingFilterStrength = 0.2f;
    params.discardInvisibleSamples = false;
    params.enableBoilingFilter = true;
    params.enablePermutationSampling = true;
    params.maxHistoryLength = 20;
    params.permutationSamplingThreshold = 0.9f;
    params.temporalDepthThreshold = 0.1f;
    params.temporalNormalThreshold = 0.5f;
    return params;
}

INL_DEVICE SpatialResamplingParameters getDefaultReSTIRDISpatialResamplingParams()
{
    SpatialResamplingParameters params = {};
    params.numDisocclusionBoostSamples = 8;
    params.numSpatialSamples = 1;
    params.spatialDepthThreshold = 0.1f;
    params.spatialNormalThreshold = 0.5f;
    params.spatialSamplingRadius = 32.0f;
    return params;
}

INL_DEVICE DISpatioTemporalResamplingParameters GetDefaultDISpatioTemporalResamplingParameters()
{
    TemporalResamplingParameters temporalResamplingParams = getDefaultReSTIRDITemporalResamplingParams();
    SpatialResamplingParameters spatialResamplingParams = getDefaultReSTIRDISpatialResamplingParams();

    DISpatioTemporalResamplingParameters stparams;
    stparams.maxHistoryLength = 20;
    stparams.depthThreshold = 0.1f;
    stparams.normalThreshold = 0.5f;
    stparams.numSamples = 2;
    stparams.numDisocclusionBoostSamples = 8;
    stparams.samplingRadius = 32.0f;
    stparams.enableMaterialSimilarityTest = true;
    return stparams;

    // RTXDI_DISpatioTemporalResamplingParameters stparams;
    // stparams.screenSpaceMotion = motionVector;
    // stparams.sourceBufferIndex = g_Const.restirDI.bufferIndices.temporalResamplingInputBufferIndex;
    // stparams.maxHistoryLength = g_Const.restirDI.temporalResamplingParams.maxHistoryLength;
    // stparams.biasCorrectionMode = g_Const.restirDI.temporalResamplingParams.temporalBiasCorrection;
    // stparams.depthThreshold = g_Const.restirDI.temporalResamplingParams.temporalDepthThreshold;
    // stparams.normalThreshold = g_Const.restirDI.temporalResamplingParams.temporalNormalThreshold;
    // stparams.numSamples = g_Const.restirDI.spatialResamplingParams.numSpatialSamples + 1;
    // stparams.numDisocclusionBoostSamples = g_Const.restirDI.spatialResamplingParams.numDisocclusionBoostSamples;
    // stparams.samplingRadius = g_Const.restirDI.spatialResamplingParams.spatialSamplingRadius;
    // stparams.enableVisibilityShortcut = g_Const.restirDI.temporalResamplingParams.discardInvisibleSamples;
    // stparams.enablePermutationSampling = usePermutationSampling;
    // stparams.enableMaterialSimilarityTest = true;
    // stparams.uniformRandomNumber = g_Const.restirDI.temporalResamplingParams.uniformRandomNumber;
    // stparams.discountNaiveSamples = g_Const.discountNaiveSamples;
}

INL_DEVICE unsigned int ReservoirPositionToPointer(Int2 reservoirPosition)
{
    const unsigned int reservoirBlockRowPitch = sysParam.reservoirBlockRowPitch;
    const unsigned int reservoirArrayPitch = sysParam.reservoirArrayPitch;
    const unsigned int reservoirArrayIndex = sysParam.iterationIndex % 2;

    Int2 blockIdx = reservoirPosition / RESERVOIR_BLOCK_SIZE;
    Int2 positionInBlock = reservoirPosition % RESERVOIR_BLOCK_SIZE;

    return reservoirArrayIndex * reservoirArrayPitch + blockIdx.y * reservoirBlockRowPitch + blockIdx.x * (RESERVOIR_BLOCK_SIZE * RESERVOIR_BLOCK_SIZE) + positionInBlock.y * RESERVOIR_BLOCK_SIZE + positionInBlock.x;
}

INL_DEVICE void StoreDIReservoir(const DIReservoir reservoir, Int2 reservoirPosition)
{
    unsigned int pointer = ReservoirPositionToPointer(reservoirPosition);
    sysParam.reservoirBuffer[pointer] = reservoir;
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

INL_DEVICE DIReservoir LoadDIReservoir(Int2 reservoirPosition)
{
    unsigned int pointer = ReservoirPositionToPointer(reservoirPosition);
    return sysParam.reservoirBuffer[pointer];
}

// // Structure that groups the parameters for GetReservoirVisibility(...)
// // Reusing final visibility reduces the number of high-quality shadow rays needed to shade
// // the scene, at the cost of somewhat softer or laggier shadows.
// struct VisibilityReuseParameters
// {
//     // Controls the maximum age of the final visibility term, measured in frames, that can be reused from the
//     // previous frame(s). Higher values result in better performance.
//     unsigned int maxAge;

//     // Controls the maximum distance in screen space between the current pixel and the pixel that has
//     // produced the final visibility term. The distance does not include the motion vectors.
//     // Higher values result in better performance and softer shadows.
//     float maxDistance;
// };

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

    // Float3 reflectedRadiance = lightSample.radiance * f * abs(dot(wi, surface.normal));
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

    return blendedSourcePdf;
}

INL_DEVICE bool TraceVisibilityShadowRay(LightSample lightSample, Surface surface)
{
    Float3 sampleDir = (lightSample.lightType == LightTypeLocalTriangle) ? normalize(lightSample.position - surface.pos) : lightSample.position;
    float maxDistance = (lightSample.lightType == LightTypeLocalTriangle) ? length(lightSample.position - surface.pos) - 1e-2f : RayMax;

    bool isLightVisible = false;
    UInt2 visibilityRayPayload = splitPointer(&isLightVisible);
    Float3 shadowRayOrig = surface.thinfilm ? (dot(sampleDir, surface.normal) > 0.0f ? surface.pos : surface.backfacePos) : surface.pos;
    optixTrace(sysParam.topObject,
               (float3)shadowRayOrig, (float3)sampleDir,
               0.0f, maxDistance, 0.0f, // tmin, tmax, time
               OptixVisibilityMask(0xFE), OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               0, 2, 2,
               visibilityRayPayload.x, visibilityRayPayload.y);

    return isLightVisible;
}

// This function is called in the spatial resampling passes to make sure that
// the samples actually land on the screen and not outside of its boundaries.
// It can clamp the position or reflect it across the nearest screen edge.
// The simplest implementation will just return the input pixelPosition.
INL_DEVICE Int2 ClampSamplePositionIntoView(Int2 pixelPosition)
{
    int width = int(sysParam.camera.resolution.x);
    int height = int(sysParam.camera.resolution.y);

    // Reflect the position across the screen edges.
    // Compared to simple clamping, this prevents the spread of colorful blobs from screen edges.
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

INL_DEVICE DIReservoir SampleLightsForSurface(
    int &randIdx,
    const Surface &surface,
    LightSample &outLightSample)
{
    SampleParameters sampleParams;

    sampleParams.numLocalLightSamples = 8;
    sampleParams.numSunLightSamples = 1;
    sampleParams.numSkyLightSamples = 1;
    sampleParams.numBrdfSamples = 1;

    sampleParams.numMisSamples = sampleParams.numLocalLightSamples + sampleParams.numSunLightSamples + sampleParams.numSkyLightSamples + sampleParams.numBrdfSamples;

    sampleParams.localLightMisWeight = float(sampleParams.numLocalLightSamples) / sampleParams.numMisSamples;
    sampleParams.sunLightMisWeight = float(sampleParams.numSunLightSamples) / sampleParams.numMisSamples;
    sampleParams.skyLightMisWeight = float(sampleParams.numSkyLightSamples) / sampleParams.numMisSamples;
    sampleParams.brdfMisWeight = float(sampleParams.numBrdfSamples) / sampleParams.numMisSamples;

    sampleParams.brdfCutoff = 0.0001f;

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

        // Create candidate sample using the helper function.
        LightSample candidateSample = createSunLightSample(sampledSunIdx);

        // Recompute UV for stream sampling.
        Int2 sunIdx(sampledSunIdx % sysParam.sunRes.x, sampledSunIdx / sysParam.sunRes.x);
        Float2 uv((sunIdx.x + 0.5f) / float(sysParam.sunRes.x), (sunIdx.y + 0.5f) / float(sysParam.sunRes.y));

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

        // Create candidate sample using the helper function.
        LightSample candidateSample = createSkyLightSample(sampledSkyIdx);

        // Recompute UV for stream sampling.
        Int2 skyIdx(sampledSkyIdx % sysParam.skyRes.x, sampledSkyIdx / sysParam.skyRes.x);
        Float2 uv((skyIdx.x + 0.5f) / float(sysParam.skyRes.x), (skyIdx.y + 0.5f) / float(sysParam.skyRes.y));

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

                    // Compute UV from sampleDir for sun.
                    Int2 sunIdx((int)(uv.x * sysParam.sunRes.x - 0.5f), (int)(uv.y * sysParam.sunRes.y - 0.5f));
                    if (sunIdx.x >= sysParam.sunRes.x)
                    {
                        sunIdx.x %= sysParam.sunRes.x;
                    }
                    if (sunIdx.x < 0)
                    {
                        sunIdx.x = sysParam.sunRes.x - ((-sunIdx.x) % sysParam.sunRes.x);
                    }

                    sunIdx.y = clampi(sunIdx.y, 0, sysParam.sunRes.y - 1);
                    int sampledSunIdx = sunIdx.y * sysParam.sunRes.x + sunIdx.x;

                    candidateSample = createSunLightSample(sampledSunIdx);
                    candidateSample.position = sampleDir; // override with the BSDF sample direction

                    lightSourcePdf = sysParam.sunAliasTable->PMF(sampledSunIdx);
                }
                else
                {
                    lightIndex = SkyLightIndex;
                    uv = EqualAreaMap(sampleDir);

                    Int2 skyIdx((int)(uv.x * sysParam.skyRes.x - 0.5f), (int)(uv.y * sysParam.skyRes.y - 0.5f));
                    clamp2i(skyIdx, Int2(0), sysParam.skyRes - 1);

                    int sampledSkyIdx = skyIdx.y * sysParam.skyRes.x + skyIdx.x;

                    candidateSample = createSkyLightSample(sampledSkyIdx);
                    candidateSample.position = sampleDir; // override with the BSDF sample direction

                    lightSourcePdf = sysParam.skyAliasTable->PMF(sampledSkyIdx);
                }
            }
            else
            {
                lightIndex = rayData.lightIdx;

                LightInfo lightInfo = sysParam.lights[lightIndex];

                TriangleLight triLight = TriangleLight::Create(lightInfo);

                uv = InverseTriangleSample(rayData.bary);
                candidateSample = triLight.calcSample(uv, surface.pos);

                if (sampleParams.brdfCutoff > 0.0f)
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

    return state;
}

INL_DEVICE int TranslateLightIndex(int currentLightID)
{
    return currentLightID;
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

// Compares two values and returns true if their relative difference is lower than the threshold.
// Zero or negative threshold makes test always succeed, not fail.
INL_DEVICE bool CompareRelativeDifference(float reference, float candidate, float threshold)
{
    return (threshold <= 0) || abs(reference - candidate) <= threshold * max(reference, candidate);
}

// See if we will reuse this neighbor or history sample using
//    edge-stopping functions (e.g., per a bilateral filter).
INL_DEVICE bool IsValidNeighbor(Float3 ourNorm, Float3 theirNorm, float ourDepth, float theirDepth, float normalThreshold, float depthThreshold)
{
    return (dot(theirNorm, ourNorm) >= normalThreshold) && CompareRelativeDifference(ourDepth, theirDepth, depthThreshold);
}

// Compares the materials of two surfaces, returns true if the surfaces
// are similar enough that we can share the light reservoirs between them.
// If unsure, just return true.
INL_DEVICE bool AreMaterialsSimilar(Surface a, Surface b)
{
    const float roughnessThreshold = 0.5f;
    // const float reflectivityThreshold = 0.25f;
    const float albedoThreshold = 0.25f;

    if (!CompareRelativeDifference(a.roughness, b.roughness, roughnessThreshold))
        return false;

    // if (abs(luminance(a.specularF0) - luminance(b.specularF0)) > reflectivityThreshold)
    //     return false;

    if (abs(luminance(a.albedo) - luminance(b.albedo)) > albedoThreshold)
        return false;

    return true;
}

INL_DEVICE DIReservoir DISpatioTemporalResampling(
    Int2 pixelPosition,
    Surface surface,
    DIReservoir curSample,
    int &randIdx,
    Int2 &temporalSamplePixelPos,
    LightSample &selectedLightSample)
{
    temporalSamplePixelPos = Int2(-1, -1);

    DIReservoir state = EmptyDIReservoir();
    CombineDIReservoirs(state, curSample, /* random = */ 0.5f, curSample.targetPdf);

    Float3 currentWorldPos = surface.pos;
    Float3 prevWorldPos = currentWorldPos; // TODO: movement of the world
    Float2 prevUV = sysParam.prevCamera.worldDirectionToUV(normalize(prevWorldPos - sysParam.prevCamera.pos));
    Int2 prevPixelPos = Int2(prevUV.x * sysParam.camera.resolution.x, prevUV.y * sysParam.camera.resolution.y);
    float expectedPrevLinearDepth = distance(prevWorldPos, sysParam.prevCamera.pos);

    unsigned int numTemporalSamples = 9;
    unsigned int numSpatialSamples = 0;
    unsigned int numSamples = numTemporalSamples + numSpatialSamples;
    const float mCap = 20.0f;

    const Int2 temporalOffsets[9] = {
        Int2(0, 0), // distance 0

        Int2(0, -1), // distance 1
        Int2(0, 1),
        Int2(-1, 0),
        Int2(1, 0),

        Int2(-1, -1), // distance 1.414
        Int2(1, -1),
        Int2(1, 1),
        Int2(-1, 1),
    };

    unsigned int cachedResult = 0;

    for (int i = 0; i < numSamples; ++i)
    {
        Int2 offset;
        if (i < numTemporalSamples)
        {
            offset = temporalOffsets[i];
        }
        else
        {
            offset = Int2(rand(sysParam, randIdx) * 16.0f - 8.0f, rand(sysParam, randIdx) * 16.0f - 8.0f);
        }
        Int2 idx = prevPixelPos + offset;
        idx = ClampSamplePositionIntoView(idx);

        Surface temporalSurface;
        if (!GetPrevSurface(temporalSurface, idx))
            continue;

        bool isNormalValid = dot(surface.normal, temporalSurface.normal) >= 0.5f;
        bool isDepthValid = abs(expectedPrevLinearDepth - temporalSurface.depth) <= 0.1f * max(expectedPrevLinearDepth, temporalSurface.depth);
        bool isRoughValid = abs(surface.roughness - temporalSurface.roughness) <= 0.5f * max(surface.roughness, temporalSurface.roughness);
        bool isAlbedoValid = abs(luminance(surface.albedo) - luminance(temporalSurface.albedo)) < 0.25f;

        if (OPTIX_CENTER_PIXEL())
        {
            OPTIX_DEBUG_PRINT(Int4(isNormalValid, isDepthValid, isRoughValid, isAlbedoValid));
        }

        if (!(isNormalValid && isDepthValid && isRoughValid && isAlbedoValid))
            continue;

        cachedResult |= (1u << unsigned int(i));

        DIReservoir prevReservoir = LoadDIReservoir(idx);
        if (isnan(prevReservoir.weightSum) || isinf(prevReservoir.weightSum))
        {
            prevReservoir = EmptyDIReservoir();
        }

        if (prevReservoir.M > mCap)
        {
            prevReservoir.M = mCap;
        }

        unsigned int originalPrevLightID = GetDIReservoirLightIndex(prevReservoir);

        float neighborWeight = 0;
        LightSample candidateLightSample = {};
        if (IsValidDIReservoir(prevReservoir))
        {
            auto lightIndex = GetDIReservoirLightIndex(prevReservoir);

            if (lightIndex == SkyLightIndex)
            {
                Float2 uv = GetDIReservoirSampleUV(prevReservoir);
                int x = int(uv.x * sysParam.skyRes.x);
                int y = int(uv.y * sysParam.skyRes.y);
                x = clampi(x, 0, sysParam.skyRes.x - 1);
                y = clampi(y, 0, sysParam.skyRes.y - 1);
                int sampledSkyIdx = y * sysParam.skyRes.x + x;
                candidateLightSample = createSkyLightSample(sampledSkyIdx);
            }
            else if (lightIndex == SunLightIndex)
            {
                Float2 uv = GetDIReservoirSampleUV(prevReservoir);
                int x = int(uv.x * sysParam.sunRes.x);
                int y = int(uv.y * sysParam.sunRes.y);
                x = clampi(x, 0, sysParam.sunRes.x - 1);
                y = clampi(y, 0, sysParam.sunRes.y - 1);
                int sampledSunIdx = y * sysParam.sunRes.x + x;
                candidateLightSample = createSunLightSample(sampledSunIdx);
            }
            else
            {
                Float2 uv = GetDIReservoirSampleUV(prevReservoir);
                LightInfo candidateLight = sysParam.lights[lightIndex];
                TriangleLight triLight = TriangleLight::Create(candidateLight);
                candidateLightSample = triLight.calcSample(uv, surface.pos);
            }

            neighborWeight = GetLightSampleTargetPdfForSurface(candidateLightSample, surface);
        }

        if (CombineDIReservoirs(state, prevReservoir, rand(sysParam, randIdx), neighborWeight))
        {
            selectedLightSample = candidateLightSample;
        }

        if (i < numTemporalSamples)
        {
            i = numTemporalSamples - 1;
        }
    }

    if (!IsValidDIReservoir(state))
    {
        return state;
    }

    FinalizeResampling(state, 1.0, state.M);

    return state;
}

// selected = i;
// selectedLightPrevID = int(originalPrevLightID);

// if (!TraceVisibilityShadowRay(candidateLightSample, surface))
// {
//     continue;
// }

// int selected = -1;

// int selectedLightPrevID = -1;

// if (IsValidDIReservoir(curSample))
// {
//     selectedLightPrevID = TranslateLightIndex(GetDIReservoirLightIndex(curSample));
// }

// unsigned int sampleIdx = (startIdx + i) & neighborOffsetMask;
// Int2 spatialOffsetI = Int2(sysParam.neighborOffsetBuffer[sampleIdx * 2], sysParam.neighborOffsetBuffer[sampleIdx * 2 + 1]);
// Float2 spatialOffsetF = Float2(spatialOffsetI.x, spatialOffsetI.y) / 255.0f * stparams.samplingRadius;
// spatialOffset = Int2(spatialOffsetF.x, spatialOffsetF.y);

// Map the light ID from the previous frame into the current frame, if it still exists
// if (IsValidDIReservoir(prevReservoir))
// {
//     if (i == 0)
//     {
//         temporalSamplePixelPos = idx;
//     }

//     int mappedLightID = TranslateLightIndex(GetDIReservoirLightIndex(prevReservoir));

//     if (mappedLightID < 0)
//     {
//         // Kill the reservoir
//         prevReservoir.weightSum = 0;
//         prevReservoir.lightData = 0;
//     }
//     else
//     {
//         // Sample is valid - modify the light ID stored
//         prevReservoir.lightData = mappedLightID | DIReservoir_LightValidBit;
//     }
// }

// unsigned int startIdx = unsigned int(rand(sysParam, randIdx) * neighborOffsetMask);

// Backproject this pixel to last frame

// Int2 prevPixelPos = pixelPosition;

// bool foundTemporalSurface = true;
// const float temporalSearchRadius = 4;
// Int2 temporalSpatialOffset = Int2(0, 0);

// Try to find a matching surface in the neighborhood of the reprojected pixel
// for (i = 0; i < 9; i++)
// {
//     Int2 offset = Int2(0, 0);
//     if (i > 0)
//     {
//         offset.x = int((rand(sysParam, randIdx) - 0.5f) * temporalSearchRadius);
//         offset.y = int((rand(sysParam, randIdx) - 0.5f) * temporalSearchRadius);
//     }

//     Int2 idx = prevPixelPos + offset;

//     // Grab shading / g-buffer data from last frame
//     if (!GetPrevSurface(temporalSurface, idx))
//         continue;

//     // Test surface similarity, discard the sample if the surface is too different.
//     if (!IsValidNeighbor(surface.normal, temporalSurface.normal, expectedPrevLinearDepth, temporalSurface.depth, stparams.normalThreshold, stparams.depthThreshold))
//         continue;

//     temporalSpatialOffset = offset;
//     foundTemporalSurface = true;
//     break;
// }

// foundTemporalSurface = false;

// Clamp the sample count at 32 to make sure we can keep the neighbor mask in an unsigned int (cachedResult)

// Apply disocclusion boost if there is no temporal surface
// if (!foundTemporalSurface)
//     numSamples = clampu(stparams.numDisocclusionBoostSamples, numSamples, 32u);

// We loop through neighbors twice.  Cache the validity / edge-stopping function
//   results for the 2nd time through.

// Since we're using our bias correction scheme, we need to remember which light selection we made

// if (OPTIX_CENTER_PIXEL())
// {
//     OPTIX_DEBUG_PRINT(prevPixelPos);
// }

// const float mCap = 20.0f;

// Walk the specified number of neighbors, resampling using RIS

// if (!IsValidDIReservoir(state))
// {
//     return state;
// }

// if (1)
// {
//     FinalizeResampling(state, 1.0, state.M);
// }
// else
// {
//     // Compute the unbiased normalization term (instead of using 1/M)
//     float pi = state.targetPdf;
//     float piSum = state.targetPdf * curSample.M;

//     if (selectedLightPrevID >= 0)
//     {
//         // To do this, we need to walk our neighbors again
//         for (i = 0; i < numSamples; ++i)
//         {
//             // If we skipped this neighbor above, do so again.
//             if ((cachedResult & (1u << unsigned int(i))) == 0)
//                 continue;

//             Int2 spatialOffset;
//             if (i < numTemporalSamples)
//             {
//                 spatialOffset = temporalOffsets[i];
//             }
//             else
//             {
//                 unsigned int sampleIdx = (startIdx + i) & neighborOffsetMask;
//                 Int2 spatialOffsetI = Int2(sysParam.neighborOffsetBuffer[sampleIdx * 2], sysParam.neighborOffsetBuffer[sampleIdx * 2 + 1]);
//                 Float2 spatialOffsetF = Float2(spatialOffsetI.x, spatialOffsetI.y) / 255.0f * stparams.samplingRadius;
//                 spatialOffset = Int2(spatialOffsetF.x, spatialOffsetF.y);
//             }
//             Int2 idx = prevPixelPos + spatialOffset;
//             idx = ClampSamplePositionIntoView(idx);

//             // Load our neighbor's G-buffer
//             Surface neighborSurface;
//             GetPrevSurface(neighborSurface, idx);

//             // Get the PDF of the sample RIS selected in the first loop, above, *at this neighbor*
//             LightSample selectedSampleAtNeighbor;
//             if (selectedLightPrevID == SkyLightIndex)
//             {
//                 Float2 uv = GetDIReservoirSampleUV(state);
//                 int x = int(uv.x * sysParam.skyRes.x);
//                 int y = int(uv.y * sysParam.skyRes.y);
//                 x = clampi(x, 0, sysParam.skyRes.x - 1);
//                 y = clampi(y, 0, sysParam.skyRes.y - 1);
//                 int sampledSkyIdx = y * sysParam.skyRes.x + x;
//                 selectedSampleAtNeighbor = createSkyLightSample(sampledSkyIdx);
//             }
//             else if (selectedLightPrevID == SunLightIndex)
//             {
//                 Float2 uv = GetDIReservoirSampleUV(state);
//                 int x = int(uv.x * sysParam.sunRes.x);
//                 int y = int(uv.y * sysParam.sunRes.y);
//                 x = clampi(x, 0, sysParam.sunRes.x - 1);
//                 y = clampi(y, 0, sysParam.sunRes.y - 1);
//                 int sampledSunIdx = y * sysParam.sunRes.x + x;
//                 selectedSampleAtNeighbor = createSunLightSample(sampledSunIdx);
//             }
//             else
//             {
//                 LightInfo selectedLightPrev = sysParam.lights[selectedLightPrevID];
//                 TriangleLight triLight = TriangleLight::Create(selectedLightPrev);
//                 selectedSampleAtNeighbor = triLight.calcSample(GetDIReservoirSampleUV(state), neighborSurface.pos);
//             }

//             float ps = 0.0f;
//             if (TraceVisibilityShadowRay(selectedSampleAtNeighbor, neighborSurface))
//             {
//                 ps = GetLightSampleTargetPdfForSurface(selectedSampleAtNeighbor, neighborSurface);
//             }

//             Int2 neighborReservoirPos = idx;
//             DIReservoir prevReservoir = LoadDIReservoir(neighborReservoirPos);

//             if (isnan(prevReservoir.weightSum) || isinf(prevReservoir.weightSum))
//             {
//                 prevReservoir = EmptyDIReservoir();
//             }

//             if (prevReservoir.M > mCap)
//             {
//                 prevReservoir.M = mCap;
//             }

//             // Select this sample for the (normalization) numerator if this particular neighbor pixel
//             //     was the one we selected via RIS in the first loop, above.
//             pi = selected == i ? ps : pi;

//             // Add to the sums of weights for the (normalization) denominator
//             piSum += ps * prevReservoir.M;
//         }
//     }
//     // Use "MIS-like" normalization
//     FinalizeResampling(state, pi, piSum);
// }

// return state;