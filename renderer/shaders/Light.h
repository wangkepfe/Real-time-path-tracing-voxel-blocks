#pragma once

#include "LinearMath.h"

struct InstanceLightMapping
{
    unsigned int instanceId;
    unsigned int lightOffset;
    unsigned int lightCount;
};

struct Surface
{
    Float3 pos;
    Float3 backfacePos;
    Float3 wo;
    float depth;
    Float3 normal;
    Float3 geoNormal;
    Float3 albedo;
    float roughness;
    bool bilambertian;
};

struct LightInfo
{
    // UInt4[0]
    Float3 center;
    unsigned int scalars; // 2x float16

    // UInt4[1]
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

    INL_HOST_DEVICE TriangleLight() {};

    // Interface methods
    INL_DEVICE LightSample calcSample(const Float2 &random, const Float3 &viewerPosition)
    {
        LightSample result;

        Float3 bary = SampleTriangle(random);
        result.position = base + edge1 * bary.y + edge2 * bary.z;
        result.normal = normal;

        result.solidAnglePdf = calcSolidAnglePdf(viewerPosition, result.position, result.normal);

        result.radiance = radiance;
        result.lightType = LightTypeLocalTriangle;

        return result;
    }

    INL_DEVICE float calcSolidAnglePdf(const Float3 &viewerPosition,
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
    INL_DEVICE static TriangleLight Create(const LightInfo &lightInfo)
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

    INL_DEVICE LightInfo Store()
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
