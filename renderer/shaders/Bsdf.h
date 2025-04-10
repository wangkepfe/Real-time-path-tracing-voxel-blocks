#pragma once

#include "LinearMath.h"

static constexpr float roughnessThreshold = 0.00001f;
static constexpr float translucencyThreshold = 0.001f;

INL_DEVICE void UnitSquareToCosineHemisphere(Float2 sample, Float3 axis, Float3 &w, float &pdf)
{
    // Choose a point on the local hemisphere coordinates about +z
    const float theta = 2.0f * M_PI * sample.x;
    const float r = sqrtf(sample.y);
    w.x = r * cosf(theta);
    w.y = r * sinf(theta);
    w.z = 1.0f - w.x * w.x - w.y * w.y;
    w.z = (0.0f < w.z) ? sqrtf(w.z) : 0.0f;

    pdf = w.z / M_PI;

    // Align with axis
    alignVector(axis, w);
}

INL_DEVICE float EvaluateFresnelDielectric(const float et, const float cosIn)
{
    const float cosi = fabsf(cosIn);

    float sint = 1.0f - cosi * cosi;
    sint = (0.0f < sint) ? sqrtf(sint) / et : 0.0f;

    // Handle total internal reflection.
    if (1.0f < sint)
    {
        return 1.0f;
    }

    float cost = 1.0f - sint * sint;
    cost = (0.0f < cost) ? sqrtf(cost) : 0.0f;

    const float et_cosi = et * cosi;
    const float et_cost = et * cost;

    const float rPerpendicular = (cosi - et_cost) / (cosi + et_cost);
    const float rParallel = (et_cosi - cost) / (et_cosi + cost);

    const float result = (rParallel * rParallel + rPerpendicular * rPerpendicular) * 0.5f;

    return (result <= 1.0f) ? result : 1.0f;
}

INL_DEVICE void LambertianReflectionBSDFSample(Float2 u, Float3 n, Float3 ng, Float3 albedo, Float3 &wi, Float3 &bsdfOverPdf, float &pdf)
{
    UnitSquareToCosineHemisphere(u, n, wi, pdf);

    if (pdf <= 0.0f || dot(wi, n) <= 0.0f || dot(wi, ng) <= 0.0f)
    {
        pdf = 0.0f;
        bsdfOverPdf = Float3(0.0f);
        return;
    }

    bsdfOverPdf = albedo; // f=albedo/pi; pdf=cos_wi/pi; this term = f/pdf*cos_wi = albedo
}

INL_DEVICE void LambertianReflectionBSDFEvaluate(Float3 n, Float3 ng, Float3 wi, Float3 albedo, Float3 &bsdf, float &pdf)
{
    if (dot(wi, n) <= 0 || dot(wi, ng) <= 0)
    {
        bsdf = Float3(0.0f);
        pdf = 0.0f;
        return;
    }

    bsdf = albedo / M_PI;                 // albedo/pi
    pdf = fmaxf(0.0f, dot(wi, n) / M_PI); // cos_wi/pi
}

INL_DEVICE void BiLambertianBSDFSample(Float3 u, Float3 n, Float3 ng, Float3 albedo, Float3 &wi, Float3 &bsdfOverPdf, float &pdf)
{
    UnitSquareToCosineHemisphere(u.xy, n, wi, pdf);

    if (pdf <= 0.0f)
    {
        pdf = 0.0f;
        bsdfOverPdf = Float3(0.0f);
        return;
    }

    if (u.z < 0.5f)
    {
        wi = -wi;
    }

    pdf *= 0.5f;
    bsdfOverPdf = albedo;
}

INL_DEVICE void BiLambertianBSDFEvaluate(Float3 n, Float3 ng, Float3 wi, Float3 albedo, Float3 &bsdf, float &pdf)
{
    bsdf = albedo / TWO_PI;                 // albedo/2pi
    pdf = fmaxf(0.0f, dot(wi, n) / TWO_PI); // cos_wi/2pi
}

INL_DEVICE void MicrofacetReflectionBSDFSample(Float2 u, Float3 n, Float3 ng, Float3 wo, Float3 F0, float alpha2, Float3 &wi, Float3 &bsdfOverPdf, float &pdf)
{
    // Sample normal
    Float3 wh;
    float cosTheta = 1.0f / sqrtf(1.0f + alpha2 * u.x / (1.0f - u.x));
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
    float phi = TWO_PI * u.y;
    wh = Float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    // Local to world space
    alignVector(n, wh);

    // Reflect
    wi = normalize(reflect3f(-wo, wh));

    // Bad half vector from shading normal sample
    if (dot(wi, ng) < 0)
    {
        pdf = 0.0f;
        bsdfOverPdf = Float3(0.0f);
        return;
    }

    // Fresnel function Shlick approximation
    float cosThetaWoWh = fmaxf(SAFE_COSINE_EPSI, dot(wh, wo));
    Float3 F = F0 + (Float3(1.0f) - F0) * pow5(1.0f - cosThetaWoWh);

    // Smith's Mask-shadowing function G
    float cosThetaWo = clampf(dot(wo, n), SAFE_COSINE_EPSI, 1.0f - SAFE_COSINE_EPSI);
    float cosThetaWi = fmaxf(SAFE_COSINE_EPSI, dot(wi, n));
    float tanThetaWo = sqrt(1.0f - cosThetaWo * cosThetaWo) / cosThetaWo;
    float G = 1.0f / (1.0f + (sqrtf(1.0f + alpha2 * tanThetaWo * tanThetaWo) - 1.0f) / 2.0f);

    // Trowbridge Reitz Distribution D
    float cosThetaWh = fmaxf(SAFE_COSINE_EPSI, dot(wh, n));
    float cosTheta2Wh = cosThetaWh * cosThetaWh;
    float tanTheta2Wh = (1.0f - cosTheta2Wh) / cosTheta2Wh;
    float e = tanTheta2Wh / alpha2 + 1.0f;
    float D = 1.0f / (M_PI * (alpha2 * cosTheta2Wh * cosTheta2Wh) * (e * e));

    // PDF
    pdf = (D * cosThetaWh) / (4.0f * cosThetaWoWh);

    // BRDF / PDF * cosThetaWi
    bsdfOverPdf = F * (G * cosThetaWoWh) / (cosThetaWh * cosThetaWo);
}

INL_DEVICE void MicrofacetReflectionBSDFEvaluate(Float3 n, Float3 ng, Float3 wi, Float3 wo, Float3 F0, float alpha2, Float3 &bsdf, float &pdf)
{
    if (dot(wo, n) <= 0 || dot(wi, n) <= 0 || dot(wo, ng) <= 0 || dot(wi, ng) <= 0)
    {
        bsdf = Float3(0.0f);
        pdf = 0.0f;
        return;
    }

    Float3 wh = normalize(wi + wo);

    // Fresnel function Shlick approximation
    float cosThetaWoWh = fmaxf(SAFE_COSINE_EPSI, dot(wh, wo));
    Float3 F = F0 + (Float3(1.0f) - F0) * pow5(1.0f - cosThetaWoWh);

    // Smith's Mask-shadowing function G
    float cosThetaWo = clampf(dot(wo, n), SAFE_COSINE_EPSI, 1.0f - SAFE_COSINE_EPSI);
    float cosThetaWi = fmaxf(SAFE_COSINE_EPSI, dot(wi, n));
    float tanThetaWo = sqrt(1.0f - cosThetaWo * cosThetaWo) / cosThetaWo;
    float G = 1.0f / (1.0f + (sqrtf(1.0f + alpha2 * tanThetaWo * tanThetaWo) - 1.0f) / 2.0f);

    // Trowbridge Reitz Distribution D
    float cosThetaWh = fmaxf(SAFE_COSINE_EPSI, dot(wh, n));
    float cosTheta2Wh = cosThetaWh * cosThetaWh;
    float tanTheta2Wh = (1.0f - cosTheta2Wh) / cosTheta2Wh;
    float e = tanTheta2Wh / alpha2 + 1.0f;
    float D = 1.0f / (M_PI * (alpha2 * cosTheta2Wh * cosTheta2Wh) * (e * e));

    // BRDF
    bsdf = F * (D * G) / (4.0f * cosThetaWo * cosThetaWi);

    // PDF
    pdf = (D * cosThetaWh) / (4.0f * cosThetaWoWh);
}

INL_DEVICE void SpecularReflectionSample(Float3 n, Float3 ng, Float3 wo, Float3 albedo, Float3 &wi, Float3 &bsdfOverPdf, float &pdf)
{
    wi = reflect3f(-wo, n);

    if (dot(wi, n) <= 0.0f || dot(wi, ng) <= 0.0f)
    {
        bsdfOverPdf = Float3(0.0f);
        pdf = 0.0f;
        return;
    }

    bsdfOverPdf = albedo;
    pdf = 1.0f;
}

INL_DEVICE void SpecularReflectionTransmissionSample(float u, Float3 n, Float3 ng, Float3 wo, Float3 albedo, Float3 &wi, Float3 &bsdfOverPdf, float &pdf, bool &transmissive)
{
    const float ior = 1.4f;
    const bool hitFrontFace = dot(wo, ng) > 0.0f;
    const float eta = hitFrontFace ? ior / 1.0f : 1.0f / ior;

    Float3 wr = reflect3f(-wo, n);
    Float3 wt;

    float R = 1.0f;
    if (refract(wt, -wo, n, eta))
    {
        R = EvaluateFresnelDielectric(eta, dot(wo, n));
    }

    if (u <= R)
    {
        wi = wr;
        pdf = R;
    }
    else
    {
        wi = wt;
        pdf = 1.0f - R;

        transmissive = true;
    }
    bsdfOverPdf = albedo / pdf;
}

INL_DEVICE void UberBSDFSample(Float4 u, Float3 n, Float3 ng, Float3 wo, Float3 albedo, bool metallic, float translucency, float roughness, Float3 &wi, Float3 &bsdfOverPdf, float &pdf, bool &transmissive)
{
    if (roughness < roughnessThreshold)
    {
        if (translucency < translucencyThreshold)
        {
            SpecularReflectionSample(n, ng, wo, albedo, wi, bsdfOverPdf, pdf);
        }
        else if (translucency > 1.0f - translucencyThreshold)
        {
            SpecularReflectionTransmissionSample(u.x, n, ng, wo, albedo, wi, bsdfOverPdf, pdf, transmissive);
        }
        else
        {
            bsdfOverPdf = Float3(0.0f);
            pdf = 0.0f;
        }
        return;
    }

    float alpha2 = roughness;

    float cosThetaWo = fmaxf(SAFE_COSINE_EPSI, dot(n, wo));
    constexpr float eta1 = 1.4f;
    constexpr float eta2 = 1.0f;
    constexpr float F0Dielectric = ((eta1 - eta2) / (eta1 + eta2)) * ((eta1 - eta2) / (eta1 + eta2));
    float reflectionProbability = clampf(F0Dielectric + (1.0f - F0Dielectric) * pow5(1.0f - cosThetaWo), SAFE_COSINE_EPSI, 1.0f - SAFE_COSINE_EPSI);
    float diffuseProbability = 1.0f - reflectionProbability;

    Float3 F0Metallic = albedo;
    Float3 F0 = metallic ? F0Metallic : Float3(F0Dielectric);
    Float3 diffuseAlbedo = metallic ? Float3(0.0f) : albedo;

    transmissive = u.z < translucency;

    if (transmissive)
    {
        n = dot(n, wo) < 0.0f ? -n : n;
        ng = dot(n, wo) < 0.0f ? -ng : ng;
        wo = reflect3f(wo, n);
        n = -n;
        ng = -ng;
    }

    if (u.w < reflectionProbability)
    {
        MicrofacetReflectionBSDFSample(u.xy, n, ng, wo, F0, alpha2, wi, bsdfOverPdf, pdf);
        pdf *= reflectionProbability;
    }
    else
    {
        LambertianReflectionBSDFSample(u.xy, n, ng, diffuseAlbedo, wi, bsdfOverPdf, pdf);
        pdf *= diffuseProbability;
    }

    if (transmissive)
    {
        pdf *= translucency;
    }
    else
    {
        pdf *= (1.0f - translucency);
    }
}

INL_DEVICE void UberBSDFEvaluate(Float3 n, Float3 ng, Float3 wi, Float3 wo, Float3 albedo, bool metallic, float translucency, float roughness, Float3 &bsdf, float &pdf)
{
    if (roughness < roughnessThreshold)
    {
        bsdf = Float3(0.0f);
        pdf = 0.0f;
        return;
    }

    float alpha2 = roughness;

    float cosThetaWo = fmaxf(SAFE_COSINE_EPSI, dot(n, wo));
    constexpr float eta1 = 1.4f;
    constexpr float eta2 = 1.0f;
    constexpr float F0Dielectric = ((eta1 - eta2) / (eta1 + eta2)) * ((eta1 - eta2) / (eta1 + eta2));
    float reflectionProbability = clampf(F0Dielectric + (1.0f - F0Dielectric) * pow5(1.0f - cosThetaWo), SAFE_COSINE_EPSI, 1.0f - SAFE_COSINE_EPSI);

    Float3 F0Metallic = albedo;
    Float3 F0 = metallic ? F0Metallic : Float3(F0Dielectric);
    Float3 diffuseAlbedo = metallic ? Float3(0.0f) : albedo;

    bool transmissive = (dot(n, wo) * dot(n, wi) < 0.0f) && (translucency > translucencyThreshold);
    float transmissionPdf = 1.0f;
    if (transmissive)
    {
        n = dot(n, wo) < 0.0f ? -n : n;
        ng = dot(n, wo) < 0.0f ? -ng : ng;
        wo = reflect3f(wo, n);
        n = -n;
        ng = -ng;

        transmissionPdf = translucency;
    }
    else
    {
        transmissionPdf = (1.0f - translucency);
    }

    Float3 bsdfReflection;
    float pdfReflection;

    MicrofacetReflectionBSDFEvaluate(n, ng, wi, wo, F0, alpha2, bsdfReflection, pdfReflection);

    Float3 bsdfDiffuse;
    float pdfDiffuse;

    LambertianReflectionBSDFEvaluate(n, ng, wi, albedo, bsdfDiffuse, pdfDiffuse);

    bsdf = lerp3f(bsdfDiffuse, bsdfReflection, reflectionProbability);
    pdf = lerpf(pdfDiffuse, pdfReflection, reflectionProbability) * transmissionPdf;
}