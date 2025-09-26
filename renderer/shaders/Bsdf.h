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

//
// Disney BSDF Implementation
// Based on: https://github.com/schuttejoe/Selas/blob/dev/Source/Core/Shading/Disney.cpp
// and https://schuttejoe.github.io/post/disneybsdf/
//

// Helper function for Disney diffuse
INL_DEVICE float DisneyDiffuseFresnel(float cosThetaWo, float cosThetaWi, float roughness)
{
    float energyBias = lerpf(0.0f, 0.5f, roughness);
    float energyFactor = lerpf(1.0f, 1.0f / 1.51f, roughness);
    float fd90 = energyBias + 2.0f * roughness * cosThetaWi * cosThetaWi;
    float f0 = 1.0f;
    float lightScatter = f0 + (fd90 - f0) * pow5(1.0f - cosThetaWo);
    float viewScatter = f0 + (fd90 - f0) * pow5(1.0f - cosThetaWi);
    return lightScatter * viewScatter * energyFactor;
}

// Helper function for GTR2 anisotropic distribution
INL_DEVICE float GTR2Aniso(float cosThetaH, float sinThetaH, float sinPhiH, float cosPhiH, float ax, float ay)
{
    float ax2 = ax * ax;
    float ay2 = ay * ay;
    float s = (cosPhiH * cosPhiH) / ax2 + (sinPhiH * sinPhiH) / ay2;
    float t = sinThetaH * sinThetaH * s + cosThetaH * cosThetaH;
    return 1.0f / (M_PI * ax * ay * t * t);
}

// Helper function for Smith G masking-shadowing for GTR2
INL_DEVICE float SmithGGGX(float cosTheta, float alpha)
{
    float a2 = alpha * alpha;
    float cosTheta2 = cosTheta * cosTheta;
    return 2.0f / (1.0f + sqrtf(1.0f + a2 * (1.0f - cosTheta2) / cosTheta2));
}

// Disney BSDF Sample function with separate diffuse/specular output
INL_DEVICE void DisneyBSDFSample(Float4 u, Float3 n, Float3 ng, Float3 wo, Float3 albedo, bool metallic, float translucency, float roughness, Float3 &wi, Float3 &bsdfOverPdf, float &pdf, bool &transmissive, bool &isSpecular)
{
    if (roughness < roughnessThreshold)
    {
        if (translucency < translucencyThreshold)
        {
            SpecularReflectionSample(n, ng, wo, albedo, wi, bsdfOverPdf, pdf);
            isSpecular = true; // Mirror-like reflection is always specular
        }
        else if (translucency > 1.0f - translucencyThreshold)
        {
            SpecularReflectionTransmissionSample(u.x, n, ng, wo, albedo, wi, bsdfOverPdf, pdf, transmissive);
            isSpecular = true; // Mirror-like transmission is also specular
        }
        else
        {
            bsdfOverPdf = Float3(0.0f);
            pdf = 0.0f;
            isSpecular = false;
        }
        return;
    }

    transmissive = false; // Disney BSDF doesn't handle transmission in this implementation

    // Disney parameters (simplified version)
    const float specular_param = 0.5f; // Default Disney specular amount for dielectrics
    const float metalness = metallic ? 1.0f : 0.0f;

    // Roughness remapping for Disney model
    float alpha = roughness * roughness;

    // Calculate Fresnel reflectance at normal incidence
    float cosThetaWo = fmaxf(SAFE_COSINE_EPSI, dot(n, wo));

    // Luminance for specular tint
    float luminance = 0.299f * albedo.x + 0.587f * albedo.y + 0.114f * albedo.z;
    Float3 tint = luminance > 0.0f ? albedo / luminance : Float3(1.0f);

    // Specular color for dielectrics (with optional tint)
    Float3 specularColor = lerp3f(Float3(1.0f), tint, 0.0f); // specularTint = 0 for now
    Float3 C0 = lerp3f(0.08f * specular_param * specularColor, albedo, metalness);

    // Calculate Fresnel
    Float3 F = C0 + (Float3(1.0f) - C0) * pow5(1.0f - cosThetaWo);
    float avgF = (F.x + F.y + F.z) / 3.0f;

    // Probability of choosing specular vs diffuse
    float specularWeight = avgF;
    float diffuseWeight = (1.0f - metalness) * (1.0f - avgF);

    // Normalize weights
    float totalWeight = specularWeight + diffuseWeight;
    if (totalWeight < SAFE_COSINE_EPSI)
    {
        bsdfOverPdf = Float3(0.0f);
        pdf = 0.0f;
        return;
    }

    float specularProb = specularWeight / totalWeight;

    // Sample either diffuse or specular
    if (u.w < specularProb)
    {
        isSpecular = true;
        // Sample microfacet normal using GGX distribution
        float cosTheta = sqrtf((1.0f - u.x) / (1.0f + (alpha * alpha - 1.0f) * u.x));
        float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
        float phi = TWO_PI * u.y;

        Float3 wh = Float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
        alignVector(n, wh);

        // Reflect to get wi
        wi = normalize(reflect3f(-wo, wh));

        if (dot(wi, n) <= 0.0f || dot(wi, ng) <= 0.0f)
        {
            bsdfOverPdf = Float3(0.0f);
            pdf = 0.0f;
            return;
        }

        // Calculate BSDF and PDF for specular
        float cosThetaWi = dot(wi, n);
        float cosThetaWh = dot(wh, n);
        float cosThetaWoWh = dot(wo, wh);

        // GTR2 distribution
        float D = GTR2Aniso(cosThetaWh, sinTheta, 0.0f, 1.0f, alpha, alpha);

        // Fresnel
        Float3 F_sample = C0 + (Float3(1.0f) - C0) * pow5(1.0f - cosThetaWoWh);

        // Masking-shadowing
        float G = SmithGGGX(cosThetaWo, alpha) * SmithGGGX(cosThetaWi, alpha);

        // BRDF
        Float3 brdf = F_sample * D * G / (4.0f * cosThetaWo * cosThetaWi);

        // PDF
        pdf = D * cosThetaWh / (4.0f * cosThetaWoWh) * specularProb;

        bsdfOverPdf = brdf * cosThetaWi / pdf;
    }
    else
    {
        isSpecular = false;
        // Sample diffuse using cosine-weighted hemisphere sampling
        float cosTheta = sqrtf(u.x);
        float sinTheta = sqrtf(1.0f - u.x);
        float phi = TWO_PI * u.y;

        wi = Float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
        alignVector(n, wi);

        if (dot(wi, ng) <= 0.0f)
        {
            bsdfOverPdf = Float3(0.0f);
            pdf = 0.0f;
            return;
        }

        // Calculate Disney diffuse
        float cosThetaWi = dot(wi, n);
        float fl = DisneyDiffuseFresnel(cosThetaWo, cosThetaWi, roughness);

        // Disney diffuse BRDF
        Float3 diffuseBrdf = albedo * (1.0f - metalness) * fl / M_PI;

        // PDF for cosine-weighted sampling
        pdf = cosThetaWi / M_PI * (1.0f - specularProb);

        bsdfOverPdf = diffuseBrdf * cosThetaWi / pdf;
    }
}

// Disney BSDF Evaluate function with separate diffuse/specular outputs
INL_DEVICE void DisneyBSDFEvaluate(Float3 n, Float3 ng, Float3 wi, Float3 wo, Float3 albedo, bool metallic, float translucency, float roughness, Float3 &diffuse, Float3 &specular, float &pdf)
{
    diffuse = Float3(0.0f);
    specular = Float3(0.0f);

    if (roughness < roughnessThreshold)
    {
        pdf = 0.0f;
        return;
    }

    if (dot(wo, n) <= 0 || dot(wi, n) <= 0 || dot(wo, ng) <= 0 || dot(wi, ng) <= 0)
    {
        pdf = 0.0f;
        return;
    }

    // Disney parameters
    const float specular_param = 0.5f;
    const float metalness = metallic ? 1.0f : 0.0f;

    // Roughness remapping
    float alpha = roughness * roughness;

    // Calculate angles
    float cosThetaWo = dot(wo, n);
    float cosThetaWi = dot(wi, n);

    // Half vector
    Float3 wh = normalize(wi + wo);
    float cosThetaWh = dot(wh, n);
    float cosThetaWoWh = dot(wo, wh);

    // Luminance and tint
    float luminance = 0.299f * albedo.x + 0.587f * albedo.y + 0.114f * albedo.z;
    Float3 tint = luminance > 0.0f ? albedo / luminance : Float3(1.0f);
    Float3 specularColor = lerp3f(Float3(1.0f), tint, 0.0f);
    Float3 C0 = lerp3f(0.08f * specular_param * specularColor, albedo, metalness);

    // Fresnel
    Float3 F = C0 + (Float3(1.0f) - C0) * pow5(1.0f - cosThetaWoWh);

    // Diffuse component (Disney diffuse) - zero for metals
    if (!metallic)
    {
        float fl = DisneyDiffuseFresnel(cosThetaWo, cosThetaWi, roughness);
        diffuse = albedo * (1.0f - metalness) * fl / M_PI;
    }

    // Specular component (GTR2/GGX)
    float sinThetaWh = sqrtf(1.0f - cosThetaWh * cosThetaWh);
    float D = GTR2Aniso(cosThetaWh, sinThetaWh, 0.0f, 1.0f, alpha, alpha);
    float G = SmithGGGX(cosThetaWo, alpha) * SmithGGGX(cosThetaWi, alpha);
    specular = F * D * G / (4.0f * cosThetaWo * cosThetaWi);

    // PDF calculation
    float avgF = (F.x + F.y + F.z) / 3.0f;
    float specularWeight = avgF;
    float diffuseWeight = (1.0f - metalness) * (1.0f - avgF);
    float totalWeight = specularWeight + diffuseWeight;

    if (totalWeight < SAFE_COSINE_EPSI)
    {
        pdf = 0.0f;
        return;
    }

    float specularProb = specularWeight / totalWeight;

    // Combined PDF
    float diffusePdf = cosThetaWi / M_PI;
    float specularPdf = D * cosThetaWh / (4.0f * cosThetaWoWh);

    pdf = diffusePdf * (1.0f - specularProb) + specularPdf * specularProb;
}