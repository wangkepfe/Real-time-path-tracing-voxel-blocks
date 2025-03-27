// #include "SystemParameter.h"
// #include "OptixShaderCommon.h"
// #include "ShaderDebugUtils.h"
// #include "Bsdf.h"

// extern "C" __constant__ SystemParameter sysParam;

// extern "C" __device__ void __direct_callable__sample_bsdf_diffuse_reflection(MaterialParameter const &parameters, MaterialState const &state, RayData *rayData, int &randIdx, Float3 &wi, Float3 &bsdfOverPdf, float &pdf)
// {
//     LambertianReflectionBSDFSample(rand2(sysParam, randIdx), state.normal, state.geoNormal, state.albedo, wi, bsdfOverPdf, pdf);
// }

// extern "C" __device__ Float4 __direct_callable__eval_bsdf_diffuse_reflection(MaterialParameter const &parameters, MaterialState const &state, RayData *const rayData, const Float3 wi)
// {
//     Float3 f;
//     float pdf;

//     LambertianReflectionBSDFEvaluate(state.normal, state.geoNormal, wi, state.albedo, f, pdf);

//     return Float4(f, pdf);
// }

// extern "C" __device__ void __direct_callable__sample_bsdf_microfacet_reflection(MaterialParameter const &parameters, MaterialState const &state, RayData *rayData, int &randIdx, Float3 &wi, Float3 &bsdfOverPdf, float &pdf)
// {
//     float alpha2 = fmaxf(state.roughness, 0.01f); // roughness = alpha.x * alpha.y , in case of isotropic surface, roughness = alpha^2

//     constexpr float eta1 = 1.4f;
//     constexpr float eta2 = 1.0f;
//     constexpr float F0 = ((eta1 - eta2) / (eta1 + eta2)) * ((eta1 - eta2) / (eta1 + eta2));

//     UberBSDFSample(rand3(sysParam, randIdx), state.normal, state.geoNormal, state.wo, state.albedo, Float3(F0), alpha2, wi, bsdfOverPdf, pdf);
// }

// extern "C" __device__ Float4 __direct_callable__eval_bsdf_microfacet_reflection(MaterialParameter const &parameters, MaterialState const &state, RayData *const rayData, const Float3 wi)
// {
//     float alpha2 = fmaxf(state.roughness, 0.01f); // roughness = alpha.x * alpha.y , in case of isotropic surface, roughness = alpha^2

//     constexpr float eta1 = 1.4f;
//     constexpr float eta2 = 1.0f;
//     constexpr float F0 = ((eta1 - eta2) / (eta1 + eta2)) * ((eta1 - eta2) / (eta1 + eta2));

//     Float3 f;
//     float pdf;

//     UberBSDFEvaluate(state.normal, state.geoNormal, wi, state.wo, state.albedo, Float3(F0), alpha2, f, pdf);

//     return Float4(f, pdf);
// }

// extern "C" __device__ void __direct_callable__sample_bsdf_microfacet_reflection_metal(MaterialParameter const &parameters, MaterialState const &state, RayData *rayData, int &randIdx, Float3 &wi, Float3 &bsdfOverPdf, float &pdf)
// {
//     float roughness = state.roughness * 0.1f; // Fudge factor

//     float alpha2 = fmaxf(roughness, 0.0001f); // roughness = alpha.x * alpha.y , in case of isotropic surface, roughness = alpha^2

//     constexpr float eta1 = 1.4f;
//     constexpr float eta2 = 1.0f;
//     constexpr float F0Dielectric = ((eta1 - eta2) / (eta1 + eta2)) * ((eta1 - eta2) / (eta1 + eta2));
//     Float3 F0Metallic = state.albedo;
//     Float3 F0 = lerp3f(Float3(F0Dielectric), F0Metallic, state.metallic);

//     Float3 diffuseAlbedo = lerp3f(state.albedo, Float3(0.0f), state.metallic);

//     UberBSDFSample(rand3(sysParam, randIdx), state.normal, state.geoNormal, state.wo, diffuseAlbedo, F0, alpha2, wi, bsdfOverPdf, pdf);
// }

// extern "C" __device__ Float4 __direct_callable__eval_bsdf_microfacet_reflection_metal(MaterialParameter const &parameters, MaterialState const &state, RayData *const rayData, const Float3 wi)
// {
//     float roughness = state.roughness * 0.1f; // Fudge factor

//     float alpha2 = fmaxf(roughness, 0.0001f); // roughness = alpha.x * alpha.y , in case of isotropic surface, roughness = alpha^2

//     constexpr float eta1 = 1.4f;
//     constexpr float eta2 = 1.0f;
//     constexpr float F0Dielectric = ((eta1 - eta2) / (eta1 + eta2)) * ((eta1 - eta2) / (eta1 + eta2));
//     Float3 F0Metallic = state.albedo;
//     Float3 F0 = lerp3f(Float3(F0Dielectric), F0Metallic, state.metallic);

//     Float3 diffuseAlbedo = lerp3f(state.albedo, Float3(0.0f), state.metallic);

//     Float3 f;
//     float pdf;

//     UberBSDFEvaluate(state.normal, state.geoNormal, wi, state.wo, diffuseAlbedo, F0, alpha2, f, pdf);

//     return Float4(f, pdf);
// }

// extern "C" __device__ void __direct_callable__sample_bsdf_specular_reflection(MaterialParameter const &parameters, MaterialState const &state, RayData *rayData, int &randIdx, Float3 &wi, Float3 &bsdfOverPdf, float &pdf)
// {
//     wi = reflect3f(-state.wo, state.normal);

//     // Do not sample opaque materials below the geometric surface.
//     // || dot(wi, state.normalGeo) <= 0.0f)
//     if (dot(wi, state.normal) <= 0.0f)
//     {
//         pdf = 0.0f;
//         return;
//     }

//     bsdfOverPdf = Float3(1.0f);
//     pdf = 1.0f; // Not 0.0f to make sure the path is not terminated. Otherwise unused for specular events.
// }

// extern "C" __device__ void __direct_callable__sample_bsdf_specular_reflection_transmission(MaterialParameter const &parameters, MaterialState const &state, RayData *rayData, int &randIdx, Float3 &wi, Float3 &bsdfOverPdf, float &pdf)
// {
//     rayData->absorptionIor = Float4(parameters.absorption, parameters.ior);

//     const bool hitFrontFace = rayData->hitFrontFace;
//     const float eta = hitFrontFace ? rayData->absorptionIor.w / 1.0f : 1.0f / rayData->absorptionIor.w;

//     Float3 wReflection = reflect3f(-rayData->wo, state.normal);
//     Float3 wRefraction;
//     bool canRefract = refract(wRefraction, -rayData->wo, state.normal, eta);

//     float reflective = 1.0f;

//     if (canRefract)
//     {
//         reflective = EvaluateFresnelDielectric(eta, dot(rayData->wo, state.normal));
//     }

//     if (rayData->isInsideVolume)
//     {
//         if (reflective == 1.0f)
//         {
//             wi = wReflection;
//         }
//         else
//         {
//             if (rand(sysParam, randIdx) < reflective)
//             {
//                 wi = wReflection;
//             }
//             else
//             {
//                 wi = wRefraction;
//                 rayData->transmissionEvent = true;
//             }
//         }
//         bsdfOverPdf = Float3(1.0f);
//     }
//     else
//     {
//         if (rand(sysParam, randIdx) < reflective)
//         {
//             wi = wReflection;
//         }
//         else
//         {
//             wi = wRefraction;
//             rayData->transmissionEvent = true;
//         }
//         bsdfOverPdf = Float3(1.0f);

//         pdf = 1.0f;
//     }
// }

// extern "C" __device__ void __direct_callable__sample_bsdf_diffuse_reflection_transmission_thinfilm(MaterialParameter const &parameters, MaterialState const &state, RayData *rayData, int &randIdx, Float3 &wi, Float3 &bsdfOverPdf, float &pdf)
// {
//     BiLambertianBSDFSample(rand3(sysParam, randIdx), state.normal, state.geoNormal, state.albedo, wi, bsdfOverPdf, pdf);
// }

// extern "C" __device__ Float4 __direct_callable__eval_bsdf_diffuse_reflection_transmission_thinfilm(MaterialParameter const &parameters, MaterialState const &state, RayData *const rayData, const Float3 wi)
// {
//     Float3 f;
//     float pdf;
//     BiLambertianBSDFEvaluate(state.normal, state.geoNormal, wi, state.albedo, f, pdf);
//     return Float4(f, pdf);
// }