#include "SystemParameter.h"
#include "OptixShaderCommon.h"
#include "ShaderDebugUtils.h"
#include "Bsdf.h"

extern "C" __constant__ SystemParameter sysParam;

extern "C" __device__ void __direct_callable__sample_bsdf_diffuse_reflection(MaterialParameter const &parameters, MaterialState const &state, RayData *rayData, Float3 &wi, Float3 &bsdfOverPdf, float &pdf)
{
    LambertianReflectionBSDFSample(rand2(sysParam, rayData->randIdx), state.normal, state.geometricNormal, state.albedo, wi, bsdfOverPdf, pdf);

    if (pdf <= 0.0f)
    {
        rayData->shouldTerminate = true;
    }
}

extern "C" __device__ Float4 __direct_callable__eval_bsdf_diffuse_reflection(MaterialParameter const &parameters, MaterialState const &state, RayData *const rayData, const Float3 wi)
{
    Float3 f;
    float pdf;

    LambertianReflectionBSDFEvaluate(state.normal, state.geometricNormal, wi, state.albedo, f, pdf);

    return Float4(f, pdf);
}

extern "C" __device__ void __direct_callable__sample_bsdf_microfacet_reflection(MaterialParameter const &parameters, MaterialState const &state, RayData *rayData, Float3 &wi, Float3 &bsdfOverPdf, float &pdf)
{
    float alpha2 = fmaxf(state.roughness, 0.01f); // roughness = alpha.x * alpha.y , in case of isotropic surface, roughness = alpha^2

    constexpr float eta1 = 1.4f;
    constexpr float eta2 = 1.0f;
    constexpr float F0 = ((eta1 - eta2) / (eta1 + eta2)) * ((eta1 - eta2) / (eta1 + eta2));

    UberBSDFSample(rand3(sysParam, rayData->randIdx), state.normal, state.geometricNormal, state.wo, state.albedo, Float3(F0), alpha2, false, wi, bsdfOverPdf, pdf);

    if (pdf <= 0.0f)
    {
        rayData->shouldTerminate = true;
    }
}

extern "C" __device__ Float4 __direct_callable__eval_bsdf_microfacet_reflection(MaterialParameter const &parameters, MaterialState const &state, RayData *const rayData, const Float3 wi)
{
    float alpha2 = fmaxf(state.roughness, 0.01f); // roughness = alpha.x * alpha.y , in case of isotropic surface, roughness = alpha^2

    constexpr float eta1 = 1.4f;
    constexpr float eta2 = 1.0f;
    constexpr float F0 = ((eta1 - eta2) / (eta1 + eta2)) * ((eta1 - eta2) / (eta1 + eta2));

    Float3 f;
    float pdf;

    UberBSDFEvaluate(state.normal, state.geometricNormal, wi, state.wo, state.albedo, Float3(F0), alpha2, false, f, pdf);

    if (pdf <= 0.0f)
    {
        rayData->shouldTerminate = true;
    }

    return Float4(f, pdf);
}

extern "C" __device__ void __direct_callable__sample_bsdf_microfacet_reflection_metal(MaterialParameter const &parameters, MaterialState const &state, RayData *rayData, Float3 &wi, Float3 &bsdfOverPdf, float &pdf)
{
    float roughness = state.roughness * 0.1f; // Fudge factor

    float alpha2 = fmaxf(roughness, 0.0001f); // roughness = alpha.x * alpha.y , in case of isotropic surface, roughness = alpha^2

    constexpr float eta1 = 1.4f;
    constexpr float eta2 = 1.0f;
    constexpr float F0Dielectric = ((eta1 - eta2) / (eta1 + eta2)) * ((eta1 - eta2) / (eta1 + eta2));
    Float3 F0Metallic = state.albedo;
    Float3 F0 = lerp3f(Float3(F0Dielectric), F0Metallic, state.metallic);

    Float3 diffuseAlbedo = lerp3f(state.albedo, Float3(0.0f), state.metallic);

    UberBSDFSample(rand3(sysParam, rayData->randIdx), state.normal, state.geometricNormal, state.wo, diffuseAlbedo, F0, alpha2, false, wi, bsdfOverPdf, pdf);

    if (pdf <= 0.0f)
    {
        rayData->shouldTerminate = true;
    }
}

extern "C" __device__ Float4 __direct_callable__eval_bsdf_microfacet_reflection_metal(MaterialParameter const &parameters, MaterialState const &state, RayData *const rayData, const Float3 wi)
{
    float roughness = state.roughness * 0.1f; // Fudge factor

    float alpha2 = fmaxf(roughness, 0.0001f); // roughness = alpha.x * alpha.y , in case of isotropic surface, roughness = alpha^2

    constexpr float eta1 = 1.4f;
    constexpr float eta2 = 1.0f;
    constexpr float F0Dielectric = ((eta1 - eta2) / (eta1 + eta2)) * ((eta1 - eta2) / (eta1 + eta2));
    Float3 F0Metallic = state.albedo;
    Float3 F0 = lerp3f(Float3(F0Dielectric), F0Metallic, state.metallic);

    Float3 diffuseAlbedo = lerp3f(state.albedo, Float3(0.0f), state.metallic);

    Float3 f;
    float pdf;

    UberBSDFEvaluate(state.normal, state.geometricNormal, wi, state.wo, diffuseAlbedo, F0, alpha2, false, f, pdf);

    if (pdf <= 0.0f)
    {
        rayData->shouldTerminate = true;
    }

    return Float4(f, pdf);
}

extern "C" __device__ void __direct_callable__sample_bsdf_specular_reflection(MaterialParameter const &parameters, MaterialState const &state, RayData *rayData, Float3 &wi, Float3 &f_over_pdf, float &pdf)
{
    wi = reflect3f(-rayData->wo, state.normal);

    // Do not sample opaque materials below the geometric surface.
    // if (pdf <= 0.0f || dot(wi, state.normalGeo) <= 0.0f)
    if (pdf <= 0.0f || dot(wi, state.normal) <= 0.0f)
    {
        rayData->shouldTerminate = true;
        return;
    }

    f_over_pdf = Float3(1.0f);
    pdf = 1.0f; // Not 0.0f to make sure the path is not terminated. Otherwise unused for specular events.
}

// This function evaluates a Fresnel dielectric function when the transmitting cosine ("cost")
// is unknown and the incident index of refraction is assumed to be 1.0f.
// \param et     The transmitted index of refraction.
// \param costIn The cosine of the angle between the incident direction and normal direction.
__forceinline__ __device__ float evaluateFresnelDielectric(const float et, const float cosIn)
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

extern "C" __device__ void __direct_callable__sample_bsdf_specular_reflection_transmission(MaterialParameter const &parameters, MaterialState const &state, RayData *rayData, Float3 &wi, Float3 &f_over_pdf, float &pdf)
{
    rayData->absorptionIor = Float4(parameters.absorption, parameters.ior);

    const bool hitFrontFace = rayData->hitFrontFace;
    const float eta = hitFrontFace ? rayData->absorptionIor.w / 1.0f : 1.0f / rayData->absorptionIor.w;

    Float3 wReflection = reflect3f(-rayData->wo, state.normal);
    Float3 wRefraction;
    bool canRefract = refract(wRefraction, -rayData->wo, state.normal, eta);

    float reflective = 1.0f;

    if (canRefract)
    {
        reflective = evaluateFresnelDielectric(eta, dot(rayData->wo, state.normal));
    }

    if (rayData->isInsideVolume)
    {
        if (reflective == 1.0f)
        {
            wi = wReflection;
        }
        else
        {
            if (rand(sysParam, rayData->randIdx) < reflective)
            {
                wi = wReflection;
            }
            else
            {
                wi = wRefraction;
                rayData->transmissionEvent = true;
            }
        }
        f_over_pdf = Float3(1.0f);
    }
    else
    {
        if (rand(sysParam, rayData->randIdx) < reflective)
        {
            wi = wReflection;
        }
        else
        {
            wi = wRefraction;
            rayData->transmissionEvent = true;
        }
        f_over_pdf = Float3(1.0f);

        pdf = 1.0f;
    }
}

extern "C" __device__ void __direct_callable__sample_bsdf_diffuse_reflection_transmission_thinfilm(MaterialParameter const &parameters, MaterialState const &state, RayData *rayData, Float3 &wi, Float3 &bsdfOverPdf, float &pdf)
{
    BiLambertianBSDFSample(rand3(sysParam, rayData->randIdx), state.normal, state.geometricNormal, state.albedo, wi, bsdfOverPdf, pdf);
}

extern "C" __device__ Float4 __direct_callable__eval_bsdf_diffuse_reflection_transmission_thinfilm(MaterialParameter const &parameters, MaterialState const &state, RayData *const rayData, const Float3 wi)
{
    Float3 f;
    float pdf;
    BiLambertianBSDFEvaluate(state.normal, state.geometricNormal, wi, state.albedo, f, pdf);
    return Float4(f, pdf);
}