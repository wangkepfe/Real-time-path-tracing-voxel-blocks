#pragma once

#include "RestirHelper.h"
#include "OptixShaderCommon.h"

struct Surface
{
    Float3 pos;
    Float3 wo;
    float depth;
    Float3 normal;
    Float3 geoNormal;
    Float3 albedo;
    float roughness;
};

struct LightInfo
{
    // uint4[0]
    Float3 center;
    unsigned int scalars; // 2x float16

    // uint4[1]
    UInt2 radiance;          // fp16x4
    unsigned int direction1; // oct-encoded
    unsigned int direction2; // oct-encoded
};

enum LightType
{
    LightTypeInvalid,
    LightTypeSky,
    LightTypeSun,
    LightTypeLocalTriangle,
};

struct LightSample
{
    Float3 position; // Position for triangle light, direction for envionment/distant light
    Float3 normal;
    Float3 radiance;
    float solidAnglePdf;
    int lightType = LightTypeInvalid;
};

struct TriangleLight
{
    Float3 base;
    Float3 edge1;
    Float3 edge2;
    Float3 radiance;
    Float3 normal;
    float surfaceArea;

    __host__ __device__ TriangleLight() {};

    // Interface methods
    __device__ LightSample calcSample(const Float2 &random, const Float3 &viewerPosition)
    {
        LightSample result;

        Float3 bary = sampleTriangle(random);
        result.position = base + edge1 * bary.y + edge2 * bary.z;
        result.normal = normal;

        result.solidAnglePdf = calcSolidAnglePdf(viewerPosition, result.position, result.normal);

        result.radiance = radiance;
        result.lightType = LightTypeLocalTriangle;

        return result;
    }

    __device__ float calcSolidAnglePdf(const Float3 &viewerPosition,
                                       const Float3 &lightSamplePosition,
                                       const Float3 &lightSampleNormal)
    {
        Float3 L = lightSamplePosition - viewerPosition;
        float Ldist = length(L);
        L /= Ldist;

        const float areaPdf = 1.0 / surfaceArea;
        const float sampleCosTheta = saturate(dot(L, -lightSampleNormal));

        return PdfAtoW(areaPdf, Ldist, sampleCosTheta);
    }

    // Helper methods
    __device__ static TriangleLight Create(const LightInfo &lightInfo)
    {
        TriangleLight triLight;

        // Extract the lower and upper 16 bits from lightInfo.scalars:
        unsigned short half0_bits = static_cast<unsigned short>(lightInfo.scalars & 0xFFFF);
        unsigned short half1_bits = static_cast<unsigned short>(lightInfo.scalars >> 16);

        // Create __half values from these bit patterns.
        // One common idiom is to write directly into the __half's memory.
        __half h0, h1;
        *((unsigned short *)&h0) = half0_bits;
        *((unsigned short *)&h1) = half1_bits;

        // Convert each __half to a float using __half2float:
        float f0 = __half2float(h0); // equivalent to f16tof32(lightInfo.scalars)
        float f1 = __half2float(h1); // equivalent to f16tof32(lightInfo.scalars >> 16)

        triLight.edge1 = octToNdirUnorm32(lightInfo.direction1) * f0;
        triLight.edge2 = octToNdirUnorm32(lightInfo.direction2) * f1;
        triLight.base = lightInfo.center - (triLight.edge1 + triLight.edge2) / 3.0f;
        triLight.radiance = Unpack_R16G16B16A16_FLOAT(lightInfo.radiance).xyz;

        Float3 lightNormal = cross(triLight.edge1, triLight.edge2);
        float lightNormalLength = length(lightNormal);

        if (lightNormalLength > 0.0f)
        {
            triLight.surfaceArea = 0.5f * lightNormalLength;
            triLight.normal = lightNormal / lightNormalLength;
        }
        else
        {
            triLight.surfaceArea = 0.0f;
            triLight.normal = Float3(0.0f);
        }

        return triLight;
    }

    __device__ LightInfo Store()
    {
        LightInfo lightInfo{};

        lightInfo.radiance = Pack_R16G16B16A16_FLOAT(Float4(radiance, 0.0f));
        lightInfo.center = base + (edge1 + edge2) / 3.0f;
        lightInfo.direction1 = ndirToOctUnorm32(normalize(edge1));
        lightInfo.direction2 = ndirToOctUnorm32(normalize(edge2));
        lightInfo.scalars = pack_f32_to_f16_bits(length(edge1)) | (pack_f32_to_f16_bits(length(edge2)) << 16);

        return lightInfo;
    }
};

// // This structure represents a single light reservoir that stores the weights, the sample ref,
// // sample count (M), and visibility for reuse. It can be serialized into RTXDI_PackedDIReservoir for storage.
// struct RTXDI_DIReservoir
// {
//     // Light index (bits 0..30) and validity bit (31)
//     uint lightData;

//     // Sample UV encoded in 16-bit fixed point format
//     uint uvData;

//     // Overloaded: represents RIS weight sum during streaming,
//     // then reservoir weight (inverse PDF) after FinalizeResampling
//     float weightSum;

//     // Target PDF of the selected sample
//     float targetPdf;

//     // Number of samples considered for this reservoir (pairwise MIS makes this a float)
//     float M;

//     // Visibility information stored in the reservoir for reuse
//     uint packedVisibility;

//     // Screen-space distance between the current location of the reservoir
//     // and the location where the visibility information was generated,
//     // minus the motion vectors applied in temporal resampling
//     int2 spatialDistance;

//     // How many frames ago the visibility information was generated
//     uint age;

//     // Cannonical weight when using pairwise MIS (ignored except during pairwise MIS computations)
//     float canonicalWeight;
// };

// struct RTXDI_PackedDIReservoir
// {
//     uint32_t lightData;
//     uint32_t uvData;
//     uint32_t mVisibility;
//     uint32_t distanceAge;
//     float targetPdf;
//     float weight;
// };

// struct RTXDI_ReservoirBufferParameters
// {
//     uint32_t reservoirBlockRowPitch;
//     uint32_t reservoirArrayPitch;
//     uint32_t pad1;
//     uint32_t pad2;
// };

// struct RTXDI_SampleParameters
// {
//     uint numLocalLightSamples;
//     uint numSunLightSamples;
//     uint numSkyLightSamples;
//     uint numBrdfSamples;

//     uint numMisSamples;

//     float localLightMisWeight;
//     float sunLightMisWeight;
//     float skyLightMisWeight;
//     float brdfMisWeight;

//     float brdfCutoff;

//     Float3 brdfRayOffsetRayOrig;
// };

// // Encoding helper constants for RTXDI_PackedDIReservoir.mVisibility
// static const uint RTXDI_PackedDIReservoir_VisibilityMask = 0x3ffff;
// static const uint RTXDI_PackedDIReservoir_VisibilityChannelMax = 0x3f;
// static const uint RTXDI_PackedDIReservoir_VisibilityChannelShift = 6;
// static const uint RTXDI_PackedDIReservoir_MShift = 18;
// static const uint RTXDI_PackedDIReservoir_MaxM = 0x3fff;

// // Encoding helper constants for RTXDI_PackedDIReservoir.distanceAge
// static const uint RTXDI_PackedDIReservoir_DistanceChannelBits = 8;
// static const uint RTXDI_PackedDIReservoir_DistanceXShift = 0;
// static const uint RTXDI_PackedDIReservoir_DistanceYShift = 8;
// static const uint RTXDI_PackedDIReservoir_AgeShift = 16;
// static const uint RTXDI_PackedDIReservoir_MaxAge = 0xff;
// static const uint RTXDI_PackedDIReservoir_DistanceMask = (1u << RTXDI_PackedDIReservoir_DistanceChannelBits) - 1;
// static const int RTXDI_PackedDIReservoir_MaxDistance = int((1u << (RTXDI_PackedDIReservoir_DistanceChannelBits - 1)) - 1);

// // Light index helpers
// static const uint RTXDI_DIReservoir_LightValidBit = 0x80000000;
// static const uint RTXDI_DIReservoir_LightIndexMask = 0x7FFFFFFF;

// static const uint RTXDI_RESERVOIR_BLOCK_SIZE = 16;
// static const float kMinRoughness = 0.05f;

// static const uint sunLightIndex = 0x7FFFFFFD;
// static const uint skyLightIndex = 0x7FFFFFFE;

// struct ReSTIRDI_Parameters
// {
//     uint32_t numPrimaryLocalLightSamples;
//     uint32_t numPrimaryInfiniteLightSamples;
//     uint32_t numPrimaryEnvironmentSamples;
//     uint32_t numPrimaryBrdfSamples;

//     float brdfCutoff;
//     uint32_t enableInitialVisibility;
//     uint32_t environmentMapImportanceSampling;

//     float temporalDepthThreshold;
//     float temporalNormalThreshold;
//     uint32_t maxHistoryLength;

//     uint32_t enablePermutationSampling;
//     float permutationSamplingThreshold;

//     uint32_t enableBoilingFilter;
//     float boilingFilterStrength;

//     uint32_t discardInvisibleSamples;

//     float spatialDepthThreshold;
//     float spatialNormalThreshold;
//     uint32_t numSpatialSamples;

//     uint32_t numDisocclusionBoostSamples;
//     float spatialSamplingRadius;

//     uint32_t enableFinalVisibility;
//     uint32_t reuseFinalVisibility;
//     uint32_t finalVisibilityMaxAge;
//     float finalVisibilityMaxDistance;

//     uint32_t enableDenoiserInputPacking;
// };

// constexpr ReSTIRDI_Parameters GetDefaultReSTIRDIParams()
// {
//     ReSTIRDI_Parameters params = {};

//     params.numPrimaryBrdfSamples = 1;
//     params.numPrimaryEnvironmentSamples = 1;
//     params.numPrimaryInfiniteLightSamples = 1;
//     params.numPrimaryLocalLightSamples = 8;

//     params.brdfCutoff = 0.0001f;
//     params.enableInitialVisibility = true;
//     params.environmentMapImportanceSampling = 1;

//     params.temporalDepthThreshold = 0.1f;
//     params.temporalNormalThreshold = 0.5f;
//     params.maxHistoryLength = 20;

//     params.enablePermutationSampling = true;
//     params.permutationSamplingThreshold = 0.9f;

//     params.enableBoilingFilter = true;
//     params.boilingFilterStrength = 0.2f;

//     params.discardInvisibleSamples = false;

//     params.spatialDepthThreshold = 0.1f;
//     params.spatialNormalThreshold = 0.5f;
//     params.numSpatialSamples = 1;

//     params.numDisocclusionBoostSamples = 8;
//     params.spatialSamplingRadius = 32.0f;

//     params.enableFinalVisibility = true;
//     params.reuseFinalVisibility = true;
//     params.finalVisibilityMaxAge = 4;
//     params.finalVisibilityMaxDistance = 16.f;

//     params.enableDenoiserInputPacking = false;

//     return params;
// }

// RTXDI_PackedDIReservoir RTXDI_PackDIReservoir(const RTXDI_DIReservoir reservoir)
// {
//     int2 clampedSpatialDistance = clamp(reservoir.spatialDistance, -RTXDI_PackedDIReservoir_MaxDistance, RTXDI_PackedDIReservoir_MaxDistance);
//     uint clampedAge = clamp(reservoir.age, 0, RTXDI_PackedDIReservoir_MaxAge);

//     RTXDI_PackedDIReservoir data;
//     data.lightData = reservoir.lightData;
//     data.uvData = reservoir.uvData;

//     data.mVisibility = reservoir.packedVisibility | (min(uint(reservoir.M), RTXDI_PackedDIReservoir_MaxM) << RTXDI_PackedDIReservoir_MShift);

//     data.distanceAge =
//         ((clampedSpatialDistance.x & RTXDI_PackedDIReservoir_DistanceMask) << RTXDI_PackedDIReservoir_DistanceXShift) | ((clampedSpatialDistance.y & RTXDI_PackedDIReservoir_DistanceMask) << RTXDI_PackedDIReservoir_DistanceYShift) | (clampedAge << RTXDI_PackedDIReservoir_AgeShift);

//     data.targetPdf = reservoir.targetPdf;
//     data.weight = reservoir.weightSum;

//     return data;
// }

// uint RTXDI_ReservoirPositionToPointer(
//     RTXDI_ReservoirBufferParameters reservoirParams,
//     uint2 reservoirPosition,
//     uint reservoirArrayIndex)
// {
//     uint2 blockIdx = reservoirPosition / RTXDI_RESERVOIR_BLOCK_SIZE;
//     uint2 positionInBlock = reservoirPosition % RTXDI_RESERVOIR_BLOCK_SIZE;

//     return reservoirArrayIndex * reservoirParams.reservoirArrayPitch + blockIdx.y * reservoirParams.reservoirBlockRowPitch + blockIdx.x * (RTXDI_RESERVOIR_BLOCK_SIZE * RTXDI_RESERVOIR_BLOCK_SIZE) + positionInBlock.y * RTXDI_RESERVOIR_BLOCK_SIZE + positionInBlock.x;
// }

// void RTXDI_StoreDIReservoir(
//     const RTXDI_DIReservoir reservoir,
//     RTXDI_ReservoirBufferParameters reservoirParams,
//     uint2 reservoirPosition,
//     uint reservoirArrayIndex,
//     RTXDI_PackedDIReservoir *RTXDI_LIGHT_RESERVOIR_BUFFER)
// {
//     uint pointer = RTXDI_ReservoirPositionToPointer(reservoirParams, reservoirPosition, reservoirArrayIndex);
//     RTXDI_LIGHT_RESERVOIR_BUFFER[pointer] = RTXDI_PackDIReservoir(reservoir);
// }

// RTXDI_DIReservoir RTXDI_EmptyDIReservoir()
// {
//     RTXDI_DIReservoir s;
//     s.lightData = 0;
//     s.uvData = 0;
//     s.targetPdf = 0;
//     s.weightSum = 0;
//     s.M = 0;
//     s.packedVisibility = 0;
//     s.spatialDistance = int2(0, 0);
//     s.age = 0;
//     s.canonicalWeight = 0;
//     return s;
// }

// RTXDI_DIReservoir RTXDI_UnpackDIReservoir(RTXDI_PackedDIReservoir data)
// {
//     RTXDI_DIReservoir res;
//     res.lightData = data.lightData;
//     res.uvData = data.uvData;
//     res.targetPdf = data.targetPdf;
//     res.weightSum = data.weight;
//     res.M = (data.mVisibility >> RTXDI_PackedDIReservoir_MShift) & RTXDI_PackedDIReservoir_MaxM;
//     res.packedVisibility = data.mVisibility & RTXDI_PackedDIReservoir_VisibilityMask;
//     // Sign extend the shift values
//     res.spatialDistance.x = int(data.distanceAge << (32 - RTXDI_PackedDIReservoir_DistanceXShift - RTXDI_PackedDIReservoir_DistanceChannelBits)) >> (32 - RTXDI_PackedDIReservoir_DistanceChannelBits);
//     res.spatialDistance.y = int(data.distanceAge << (32 - RTXDI_PackedDIReservoir_DistanceYShift - RTXDI_PackedDIReservoir_DistanceChannelBits)) >> (32 - RTXDI_PackedDIReservoir_DistanceChannelBits);
//     res.age = (data.distanceAge >> RTXDI_PackedDIReservoir_AgeShift) & RTXDI_PackedDIReservoir_MaxAge;
//     res.canonicalWeight = 0.0f;

//     // Discard reservoirs that have Inf/NaN
//     if (isinf(res.weightSum) || isnan(res.weightSum))
//     {
//         res = RTXDI_EmptyDIReservoir();
//     }

//     return res;
// }

// RTXDI_DIReservoir RTXDI_LoadDIReservoir(
//     RTXDI_ReservoirBufferParameters reservoirParams,
//     uint2 reservoirPosition,
//     uint reservoirArrayIndex,
//     RTXDI_PackedDIReservoir *RTXDI_LIGHT_RESERVOIR_BUFFER)
// {
//     uint pointer = RTXDI_ReservoirPositionToPointer(reservoirParams, reservoirPosition, reservoirArrayIndex);
//     return RTXDI_UnpackDIReservoir(RTXDI_LIGHT_RESERVOIR_BUFFER[pointer]);
// }

// void RTXDI_StoreVisibilityInDIReservoir(
//     inout RTXDI_DIReservoir reservoir,
//     float3 visibility,
//     bool discardIfInvisible)
// {
//     reservoir.packedVisibility = uint(saturate(visibility.x) * RTXDI_PackedDIReservoir_VisibilityChannelMax) | (uint(saturate(visibility.y) * RTXDI_PackedDIReservoir_VisibilityChannelMax)) << RTXDI_PackedDIReservoir_VisibilityChannelShift | (uint(saturate(visibility.z) * RTXDI_PackedDIReservoir_VisibilityChannelMax)) << (RTXDI_PackedDIReservoir_VisibilityChannelShift * 2);

//     reservoir.spatialDistance = int2(0, 0);
//     reservoir.age = 0;

//     if (discardIfInvisible && visibility.x == 0 && visibility.y == 0 && visibility.z == 0)
//     {
//         // Keep M for correct resampling, remove the actual sample
//         reservoir.lightData = 0;
//         reservoir.weightSum = 0;
//     }
// }

// // Structure that groups the parameters for RTXDI_GetReservoirVisibility(...)
// // Reusing final visibility reduces the number of high-quality shadow rays needed to shade
// // the scene, at the cost of somewhat softer or laggier shadows.
// struct RTXDI_VisibilityReuseParameters
// {
//     // Controls the maximum age of the final visibility term, measured in frames, that can be reused from the
//     // previous frame(s). Higher values result in better performance.
//     uint maxAge;

//     // Controls the maximum distance in screen space between the current pixel and the pixel that has
//     // produced the final visibility term. The distance does not include the motion vectors.
//     // Higher values result in better performance and softer shadows.
//     float maxDistance;
// };

// bool RTXDI_GetDIReservoirVisibility(
//     const RTXDI_DIReservoir reservoir,
//     const RTXDI_VisibilityReuseParameters params,
//     out float3 o_visibility)
// {
//     if (reservoir.age > 0 &&
//         reservoir.age <= params.maxAge &&
//         length(float2(reservoir.spatialDistance)) < params.maxDistance)
//     {
//         o_visibility.x = float(reservoir.packedVisibility & RTXDI_PackedDIReservoir_VisibilityChannelMax) / RTXDI_PackedDIReservoir_VisibilityChannelMax;
//         o_visibility.y = float((reservoir.packedVisibility >> RTXDI_PackedDIReservoir_VisibilityChannelShift) & RTXDI_PackedDIReservoir_VisibilityChannelMax) / RTXDI_PackedDIReservoir_VisibilityChannelMax;
//         o_visibility.z = float((reservoir.packedVisibility >> (RTXDI_PackedDIReservoir_VisibilityChannelShift * 2)) & RTXDI_PackedDIReservoir_VisibilityChannelMax) / RTXDI_PackedDIReservoir_VisibilityChannelMax;

//         return true;
//     }

//     o_visibility = float3(0, 0, 0);
//     return false;
// }

// bool RTXDI_IsValidDIReservoir(const RTXDI_DIReservoir reservoir)
// {
//     return reservoir.lightData != 0;
// }

// uint RTXDI_GetDIReservoirLightIndex(const RTXDI_DIReservoir reservoir)
// {
//     return reservoir.lightData & RTXDI_DIReservoir_LightIndexMask;
// }

// float2 RTXDI_GetDIReservoirSampleUV(const RTXDI_DIReservoir reservoir)
// {
//     return float2(reservoir.uvData & 0xffff, reservoir.uvData >> 16) / float(0xffff);
// }

// float RTXDI_GetDIReservoirInvPdf(const RTXDI_DIReservoir reservoir)
// {
//     return reservoir.weightSum;
// }

// // Adds a new, non-reservoir light sample into the reservoir, returns true if this sample was selected.
// // Algorithm (3) from the ReSTIR paper, Streaming RIS using weighted reservoir sampling.
// bool RTXDI_StreamSample(
//     inout RTXDI_DIReservoir reservoir,
//     uint lightIndex,
//     float2 uv,
//     float random,
//     float targetPdf,
//     float invSourcePdf)
// {
//     // What's the current weight
//     float risWeight = targetPdf * invSourcePdf;

//     // Add one sample to the counter
//     reservoir.M += 1;

//     // Update the weight sum
//     reservoir.weightSum += risWeight;

//     // Decide if we will randomly pick this sample
//     bool selectSample = (random * reservoir.weightSum < risWeight);

//     // If we did select this sample, update the relevant data.
//     // New samples don't have visibility or age information, we can skip that.
//     if (selectSample)
//     {
//         reservoir.lightData = lightIndex | RTXDI_DIReservoir_LightValidBit;
//         reservoir.uvData = uint(saturate(uv.x) * 0xffff) | (uint(saturate(uv.y) * 0xffff) << 16);
//         reservoir.targetPdf = targetPdf;
//     }

//     return selectSample;
// }

// // Adds `newReservoir` into `reservoir`, returns true if the new reservoir's sample was selected.
// // This is a very general form, allowing input parameters to specfiy normalization and targetPdf
// // rather than computing them from `newReservoir`.  Named "internal" since these parameters take
// // different meanings (e.g., in RTXDI_CombineDIReservoirs() or RTXDI_StreamNeighborWithPairwiseMIS())
// bool RTXDI_InternalSimpleResample(
//     inout RTXDI_DIReservoir reservoir,
//     const RTXDI_DIReservoir newReservoir,
//     float random,
//     float targetPdf RTXDI_DEFAULT(1.0f),           // Usually closely related to the sample normalization,
//     float sampleNormalization RTXDI_DEFAULT(1.0f), //     typically off by some multiplicative factor
//     float sampleM RTXDI_DEFAULT(1.0f)              // In its most basic form, should be newReservoir.M
// )
// {
//     // What's the current weight (times any prior-step RIS normalization factor)
//     float risWeight = targetPdf * sampleNormalization;

//     // Our *effective* candidate pool is the sum of our candidates plus those of our neighbors
//     reservoir.M += sampleM;

//     // Update the weight sum
//     reservoir.weightSum += risWeight;

//     // Decide if we will randomly pick this sample
//     bool selectSample = (random * reservoir.weightSum < risWeight);

//     // If we did select this sample, update the relevant data
//     if (selectSample)
//     {
//         reservoir.lightData = newReservoir.lightData;
//         reservoir.uvData = newReservoir.uvData;
//         reservoir.targetPdf = targetPdf;
//         reservoir.packedVisibility = newReservoir.packedVisibility;
//         reservoir.spatialDistance = newReservoir.spatialDistance;
//         reservoir.age = newReservoir.age;
//     }

//     return selectSample;
// }

// // Adds `newReservoir` into `reservoir`, returns true if the new reservoir's sample was selected.
// // Algorithm (4) from the ReSTIR paper, Combining the streams of multiple reservoirs.
// // Normalization - Equation (6) - is postponed until all reservoirs are combined.
// bool RTXDI_CombineDIReservoirs(
//     inout RTXDI_DIReservoir reservoir,
//     const RTXDI_DIReservoir newReservoir,
//     float random,
//     float targetPdf)
// {
//     return RTXDI_InternalSimpleResample(
//         reservoir,
//         newReservoir,
//         random,
//         targetPdf,
//         newReservoir.weightSum * newReservoir.M,
//         newReservoir.M);
// }

// // Performs normalization of the reservoir after streaming. Equation (6) from the ReSTIR paper.
// void RTXDI_FinalizeResampling(
//     inout RTXDI_DIReservoir reservoir,
//     float normalizationNumerator,
//     float normalizationDenominator)
// {
//     float denominator = reservoir.targetPdf * normalizationDenominator;

//     reservoir.weightSum = (denominator == 0.0) ? 0.0 : (reservoir.weightSum * normalizationNumerator) / denominator;
// }

// // Sample parameters struct
// // Defined so that so these can be compile time constants as defined by the user
// // brdfCutoff Value in range [0,1] to determine how much to shorten BRDF rays. 0 to disable shortening
// RTXDI_SampleParameters RTXDI_InitSampleParameters(
//     ReSTIRDI_Parameters params, Float3 brdfRayOffsetRayOrig)
// {
//     RTXDI_SampleParameters result;

//     result.numLocalLightSamples = params.numPrimaryLocalLightSamples;
//     result.numSunLightSamples = params.numPrimaryInfiniteLightSamples;
//     result.numSkyLightSamples = params.numPrimaryEnvironmentSamples;
//     result.numBrdfSamples = params.numPrimaryBrdfSamples;

//     result.numMisSamples = result.numLocalLightSamples + result.numSunLightSamples + result.numSkyLightSamples + result.numBrdfSamples;

//     result.localLightMisWeight = float(result.numLocalLightSamples) / result.numMisSamples;
//     result.sunLightMisWeight = float(result.numSunLightSamples) / result.numMisSamples;
//     result.skyLightMisWeight = float(result.numSkyLightSamples) / result.numMisSamples;
//     result.brdfMisWeight = float(result.numBrdfSamples) / result.numMisSamples;

//     result.brdfCutoff = params.brdfCutoff;

//     result.brdfRayOffsetRayOrig = brdfRayOffsetRayOrig;

//     return result;
// }

// void RAB_GetLightDirDistance(RAB_Surface surface, RAB_LightSample lightSample, float3 &o_lightDir, float o_lightDistance)
// {
//     if (lightSample.lightType == LightTypeSky || lightSample.lightType == LightTypeSun)
//     {
//         o_lightDir = lightSample.position;
//         o_lightDistance = 1e20f;
//     }
//     else
//     {
//         float3 toLight = lightSample.position - surface.pos;
//         o_lightDistance = length(toLight);
//         o_lightDir = toLight / o_lightDistance;
//     }
// }

// float RAB_GetSurfaceBrdfPdf(Surface surface, float3 wi)
// {
//     Float3 f;
//     float pdf;

//     FresnelBlendReflectionBSDFEvaluate(surface.normal, surface.geoNormal, wi, wo, surface.albedo, Float3(0.0278f), fmaxf(surface.roughness, 0.01f), f, pdf);

//     return pdf;
// }

// bool RAB_GetSurfaceBrdfSample(Surface surface, Float3 u, float3 &wi, float &pdf)
// {
//     Float3 bsdfOverPdf;

//     FresnelBlendReflectionBSDFSample(u, surface.normal, surface.geoNormal, surface.wo, surface.albedo, Float3(0.0278f), fmaxf(surface.roughness, 0.01f), wi, bsdfOverPdf, pdf);

//     return pdf > 0.0f;
// }

// // Computes the weight of the given light samples when the given surface is
// // shaded using that light sample. Exact or approximate BRDF evaluation can be
// // used to compute the weight. ReSTIR will converge to a correct lighting result
// // even if all samples have a fixed weight of 1.0, but that will be very noisy.
// // Scaling of the weights can be arbitrary, as long as it's consistent
// // between all lights and surfaces.
// float RAB_GetLightSampleTargetPdfForSurface(RAB_LightSample lightSample, RAB_Surface surface)
// {
//     if (lightSample.solidAnglePdf <= 0)
//         return 0;

//     float3 wi = normalize(lightSample.position - surface.pos);

//     if (dot(wi, surface.geoNormal) <= 0)
//         return 0;

//     Float3 f;
//     float pdf;

//     FresnelBlendReflectionBSDFEvaluate(surface.normal, surface.geoNormal, wi, surface.wo, surface.albedo, Float3(0.0278f), fmaxf(surface.roughness, 0.01f), f, pdf);

//     float3 reflectedRadiance = lightSample.radiance * f;

//     return luminance(reflectedRadiance) / lightSample.solidAnglePdf;
// }

// // Heuristic to determine a max visibility ray length from a PDF wrt. solid angle.
// float RTXDI_BrdfMaxDistanceFromPdf(float brdfCutoff, float pdf)
// {
//     const float kRayTMax = 3.402823466e+38F; // FLT_MAX
//     return brdfCutoff > 0.f ? sqrt((1.f / brdfCutoff - 1.f) * pdf) : kRayTMax;
// }

// // Computes the multi importance sampling pdf for brdf and light sample.
// // For light and BRDF PDFs wrt solid angle, blend between the two.
// //      lightSelectionPdf is a dimensionless selection pdf
// float RTXDI_LightBrdfMisWeight(Surface surface, LightSample lightSample, float lightSelectionPdf, float lightMisWeight, bool isEnvironmentMap, RTXDI_SampleParameters sampleParams)
// {
//     float lightSolidAnglePdf = lightSample.solidAnglePdf;

//     if (sampleParams.brdfMisWeight == 0 || lightSolidAnglePdf <= 0 || isinf(lightSolidAnglePdf) || isnan(lightSolidAnglePdf))
//     {
//         // BRDF samples disabled or we can't trace BRDF rays MIS with analytical lights
//         return lightMisWeight * lightSelectionPdf;
//     }

//     float3 lightDir;
//     float lightDistance;
//     RAB_GetLightDirDistance(surface, lightSample, lightDir, lightDistance);

//     // Compensate for ray shortening due to brdf cutoff, does not apply to environment map sampling
//     float brdfPdf = RAB_GetSurfaceBrdfPdf(surface, lightDir);
//     float maxDistance = RTXDI_BrdfMaxDistanceFromPdf(sampleParams.brdfCutoff, brdfPdf);
//     if (!isEnvironmentMap && lightDistance > maxDistance)
//         brdfPdf = 0.0f;

//     // Convert light selection pdf (unitless) to a solid angle measurement
//     float sourcePdfWrtSolidAngle = lightSelectionPdf * lightSolidAnglePdf;

//     // MIS blending against solid angle pdfs.
//     float blendedPdfWrtSolidangle = lightMisWeight * sourcePdfWrtSolidAngle + sampleParams.brdfMisWeight * brdfPdf;

//     // Convert back, RTXDI divides shading again by this term later
//     return blendedPdfWrtSolidangle / lightSolidAnglePdf;
// }

// RTXDI_DIReservoir RTXDI_SampleLightsForSurface(
//     const SystemParameter &sysParam,
//     int &randIdx,
//     const Surface &surface,
//     const RTXDI_SampleParameters &sampleParams,
//     LightSample &o_lightSample)
// {
//     o_lightSample = LightSample{};

//     // Local light
//     RTXDI_DIReservoir localReservoir = RTXDI_EmptyDIReservoir();
//     LightSample localSample = LightSample{};
//     for (uint i = 0; i < sampleParams.numLocalLightSamples; i++)
//     {
//         float sourcePdf;
//         int lightIndex = sysParam.lightAliasTable.sample(rand(sysParam, randIdx), sourcePdf);
//         LightInfo lightInfo = sysParam.lights[lightIndex];

//         float2 uv = rand2(sysParam, randIdx);

//         LightSample candidateSample = triLight.calcSample(uv, surface.pos);

//         float blendedSourcePdf = RTXDI_LightBrdfMisWeight(surface, candidateSample, sourcePdf, sampleParams.localLightMisWeight, false, sampleParams);
//         float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
//         float risRnd = rand(sysParam, randIdx);

//         if (blendedSourcePdf != 0)
//         {
//             bool selected = RTXDI_StreamSample(localReservoir, lightIndex, uv, risRnd, targetPdf, 1.0f / blendedSourcePdf);
//             if (selected)
//             {
//                 localSample = candidateSample;
//             }
//         }
//     }
//     RTXDI_FinalizeResampling(localReservoir, 1.0, sampleParams.numMisSamples);
//     localReservoir.M = 1;

//     // Sun
//     RTXDI_DIReservoir sunLightReservoir = RTXDI_EmptyDIReservoir();
//     LightSample sunLightSample = LightSample{};
//     for (uint i = 0; i < sampleParams.numSunLightSamples; i++)
//     {
//         float sourcePdf;
//         int sampledSunIdx = sysParam.sunAliasTable.sample(rand(sysParam, randIdx), sourcePdf);
//         float2 uv = Float2(((sampledSunIdx % sysParam.sunRes.x) + 0.5f) / sysParam.sunRes.x, ((sampledSunIdx / sysParam.sunRes.x) + 0.5f) / sysParam.sunRes.y);

//         const float sunAngle = 0.51f; // angular diagram in degrees
//         const float sunAngleCosThetaMax = cosf(sunAngle * M_PI / 180.0f / 2.0f);
//         float solidAnglePdf = (sysParam.sunRes.x * sysParam.sunRes.y) / (TWO_PI * (1.0f - sunAngleCosThetaMax));

//         Float3 rayDir = EqualAreaMapCone(sysParam.sunDir, uv.x, uv.y, sunAngleCosThetaMax);
//         Float3 sunEmission = Load2DFloat4(sysParam.sunBuffer, sampledSunIdx).xyz;

//         LightSample candidateSample;
//         candidateSample.position = rayDir;
//         candidateSample.radiance = sunEmission;
//         candidateSample.solidAnglePdf = solidAnglePdf;
//         candidateSample.lightType = LightTypeSun;

//         float blendedSourcePdf = RTXDI_LightBrdfMisWeight(surface, candidateSample, sourcePdf, sampleParams.sunLightMisWeight, true, sampleParams);
//         float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
//         float risRnd = rand(sysParam, randIdx);

//         bool selected = RTXDI_StreamSample(sunLightReservoir, sunLightIndex, uv, risRnd, targetPdf, 1.0f / blendedSourcePdf);
//         if (selected)
//         {
//             sunLightSample = candidateSample;
//         }
//     }
//     RTXDI_FinalizeResampling(sunLightReservoir, 1.0, sampleParams.numMisSamples);
//     sunLightReservoir.M = 1;

//     // Sky
//     RTXDI_DIReservoir skyLightReservoir = RTXDI_EmptyDIReservoir();
//     LightSample skyLightSample = LightSample{};
//     for (uint i = 0; i < sampleParams.numSkyLightSamples; i++)
//     {
//         float sourcePdf;
//         int sampledSkyIdx = sysParam.skyAliasTable.sample(rand(sysParam, randIdx), sourcePdf);
//         float2 uv = Float2(((sampledSkyIdx % sysParam.skyRes.x) + 0.5f) / sysParam.skyRes.x, ((sampledSkyIdx / sysParam.skyRes.x) + 0.5f) / sysParam.skyRes.y);

//         float solidAnglePdf = (sysParam.skyRes.x * sysParam.skyRes.y) / TWO_PI;

//         Float3 rayDir = EqualAreaMap(uv.x, uv.y);
//         Float3 skyEmission = Load2DFloat4(sysParam.skyBuffer, sampledSkyIdx).xyz;

//         LightSample candidateSample;
//         candidateSample.position = rayDir;
//         candidateSample.radiance = skyEmission;
//         candidateSample.solidAnglePdf = solidAnglePdf;
//         candidateSample.lightType = LightTypeSky;

//         float blendedSourcePdf = RTXDI_LightBrdfMisWeight(surface, candidateSample, sourcePdf, sampleParams.skyLightMisWeight, true, sampleParams);
//         float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
//         float risRnd = rand(sysParam, randIdx);

//         bool selected = RTXDI_StreamSample(skyLightReservoir, skyLightIndex, uv, risRnd, targetPdf, 1.0f / blendedSourcePdf);
//         if (selected)
//         {
//             skyLightSample = candidateSample;
//         }
//     }
//     RTXDI_FinalizeResampling(skyLightReservoir, 1.0, sampleParams.numMisSamples);
//     skyLightReservoir.M = 1;

//     // BSDF sample
//     DIReservoir brdfReservoir = RTXDI_EmptyDIReservoir();
//     LightSample brdfSample = LightSample{};
//     for (uint i = 0; i < sampleParams.numBrdfSamples; ++i)
//     {
//         float lightSourcePdf = 0;
//         float3 sampleDir;
//         uint lightIndex = RTXDI_InvalidLightIndex;
//         float2 randXY = float2(0, 0);
//         LightSample candidateSample = LightSample{};

//         float brdfPdf;
//         if (RAB_GetSurfaceBrdfSample(surface, rng, sampleDir, brdfPdf))
//         {
//             float maxDistance = RTXDI_BrdfMaxDistanceFromPdf(sampleParams.brdfCutoff, brdfPdf);

//             bool hitAnything = RAB_TraceRayForLocalLight(RAB_GetSurfaceWorldPos(surface), sampleDir,
//                                                          sampleParams.brdfRayMinT, maxDistance, lightIndex, randXY);

//             if (lightIndex != RTXDI_InvalidLightIndex)
//             {
//                 RAB_LightInfo lightInfo = RAB_LoadLightInfo(lightIndex, false);
//                 candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, randXY);

//                 if (sampleParams.brdfCutoff > 0.f)
//                 {
//                     // If Mis cutoff is used, we need to evaluate the sample and make sure it actually could have been
//                     // generated by the area sampling technique. This is due to numerical precision.
//                     float3 lightDir;
//                     float lightDistance;
//                     RAB_GetLightDirDistance(surface, candidateSample, lightDir, lightDistance);

//                     brdfPdf = RAB_GetSurfaceBrdfPdf(surface, lightDir);
//                     float maxDistance = RTXDI_BrdfMaxDistanceFromPdf(sampleParams.brdfCutoff, brdfPdf);
//                     if (lightDistance > maxDistance)
//                         lightIndex = RTXDI_InvalidLightIndex;
//                 }

//                 if (lightIndex != RTXDI_InvalidLightIndex)
//                 {
//                     lightSourcePdf = RAB_EvaluateLocalLightSourcePdf(lightIndex);
//                 }
//             }
//             else if (!hitAnything && (lightBufferParams.environmentLightParams.lightPresent != 0))
//             {
//                 // sample environment light
//                 lightIndex = lightBufferParams.environmentLightParams.lightIndex;
//                 RAB_LightInfo lightInfo = RAB_LoadLightInfo(lightIndex, false);
//                 randXY = RAB_GetEnvironmentMapRandXYFromDir(sampleDir);
//                 candidateSample = RAB_SamplePolymorphicLight(lightInfo, surface, randXY);
//                 lightSourcePdf = RAB_EvaluateEnvironmentMapSamplingPdf(sampleDir);
//             }
//         }

//         if (lightSourcePdf == 0)
//         {
//             // Did not hit a visible light
//             continue;
//         }

//         bool isEnvMapSample = lightIndex == lightBufferParams.environmentLightParams.lightIndex;
//         float targetPdf = RAB_GetLightSampleTargetPdfForSurface(candidateSample, surface);
//         float blendedSourcePdf = RTXDI_LightBrdfMisWeight(surface, candidateSample, lightSourcePdf,
//                                                           isEnvMapSample ? sampleParams.environmentMapMisWeight : sampleParams.localLightMisWeight,
//                                                           isEnvMapSample,
//                                                           sampleParams);
//         float risRnd = RAB_GetNextRandom(rng);

//         bool selected = RTXDI_StreamSample(brdfreservoir, lightIndex, randXY, risRnd, targetPdf, 1.0f / blendedSourcePdf);
//         if (selected)
//         {
//             o_selectedSample = candidateSample;
//         }
//     }

//     RTXDI_FinalizeResampling(brdfreservoir, 1.0, sampleParams.numMisSamples);
//     brdfreservoir.M = 1;

//     //
//     RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();
//     RTXDI_CombineDIReservoirs(state, localReservoir, 0.5, localReservoir.targetPdf);
//     bool selectInfinite = RTXDI_CombineDIReservoirs(state, infiniteReservoir, RAB_GetNextRandom(rng), infiniteReservoir.targetPdf);
//     bool selectEnvironment = RTXDI_CombineDIReservoirs(state, environmentReservoir, RAB_GetNextRandom(rng), environmentReservoir.targetPdf);
//     bool selectBrdf = RTXDI_CombineDIReservoirs(state, brdfReservoir, RAB_GetNextRandom(rng), brdfReservoir.targetPdf);

//     RTXDI_FinalizeResampling(state, 1.0, 1.0);
//     state.M = 1;

//     if (selectBrdf)
//         o_lightSample = brdfSample;
//     else if (selectEnvironment)
//         o_lightSample = environmentSample;
//     else if (selectInfinite)
//         o_lightSample = infiniteSample;
//     else
//         o_lightSample = localSample;

//     return state;
// }
