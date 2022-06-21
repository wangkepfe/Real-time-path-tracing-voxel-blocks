#include "SystemParameter.h"
#include "OptixShaderCommon.h"

namespace jazzfusion
{

__forceinline__ __device__ void alignVector(Float3 const& axis, Float3& w)
{
    // Align w with axis.
    const float s = copysignf(1.0f, axis.z);
    w.z *= s;
    const Float3 h = Float3(axis.x, axis.y, axis.z + s);
    const float k = dot(w, h) / (1.0f + fabsf(axis.z));
    w = k * h - w;
}

__forceinline__ __device__ void unitSquareToCosineHemisphere(const Float2 sample, Float3 const& axis, Float3& w, float& pdf)
{
    // Choose a point on the local hemisphere coordinates about +z.
    const float theta = 2.0f * M_PIf * sample.x;
    const float r = sqrtf(sample.y);
    w.x = r * cosf(theta);
    w.y = r * sinf(theta);
    w.z = 1.0f - w.x * w.x - w.y * w.y;
    w.z = (0.0f < w.z) ? sqrtf(w.z) : 0.0f;

    pdf = w.z * M_1_PIf;

    // Align with axis.
    alignVector(axis, w);
}

extern "C" __device__ void __direct_callable__sample_bsdf_diffuse_reflection(MaterialParameter const& parameters, State const& state, PerRayData * prd, Float3 & wi, Float3 & f_over_pdf, float& pdf)
{
    unitSquareToCosineHemisphere(rng2(prd->seed), state.normal, wi, pdf);

    if (pdf <= 0.0f || dot(wi, state.normalGeo) <= 0.0f)
    {
        prd->flags |= FLAG_TERMINATE;
        return;
    }

    f_over_pdf = state.albedo;
}

// The parameter wiL is the lightSample.direction (direct lighting), not the next ray segment's direction prd.wi (indirect lighting).
extern "C" __device__ Float4 __direct_callable__eval_bsdf_diffuse_reflection(MaterialParameter const& parameters, State const& state, PerRayData* const prd, const Float3 wiL)
{
    const Float3 f = state.albedo * M_1_PIf;
    const float pdf = fmaxf(0.0f, dot(wiL, state.normal) * M_1_PIf);

    return Float4(f, pdf);
}


extern "C" __device__ void __direct_callable__sample_bsdf_specular_reflection(MaterialParameter const& parameters, State const& state, PerRayData * prd, Float3 & wi, Float3 & f_over_pdf, float& pdf)
{
    wi = reflect3f(-prd->wo, state.normal);

    if (dot(wi, state.normalGeo) <= 0.0f) // Do not sample opaque materials below the geometric surface.
    {
        prd->flags |= FLAG_TERMINATE;
        return;
    }

    f_over_pdf = state.albedo;
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

extern "C" __device__ void __direct_callable__sample_bsdf_specular_reflection_transmission(MaterialParameter const& parameters, State const& state, PerRayData * prd, Float3 & wi, Float3 & f_over_pdf, float& pdf)
{
    // Return the current material's absorption coefficient and ior to the integrator to be able to support nested materials.
    prd->absorption_ior = Float4(parameters.absorption, parameters.ior);

    // Need to figure out here which index of refraction to use if the ray is already inside some refractive medium.
    // This needs to happen with the original FLAG_FRONTFACE condition to find out from which side of the geometry we're looking!
    // ior.xy are the current volume's IOR and the surrounding volume's IOR.
    // Thin-walled materials have no volume, always use the frontface eta for them!
    const float eta = (prd->flags & (FLAG_FRONTFACE | FLAG_THINWALLED))
        ? prd->absorption_ior.w / prd->ior.x
        : prd->ior.y / prd->absorption_ior.w;

    const Float3 R = reflect3f(-prd->wo, state.normal);

    float reflective = 1.0f;

    if (refract(wi, -prd->wo, state.normal, eta))
    {
        if (prd->flags & FLAG_THINWALLED)
        {
            wi = -prd->wo; // Straight through, no volume.
        }
        // Total internal reflection will leave this reflection probability at 1.0f.
        reflective = evaluateFresnelDielectric(eta, dot(prd->wo, state.normal));
    }

    const float pseudo = rng(prd->seed);
    if (pseudo < reflective)
    {
        wi = R; // Fresnel reflection or total internal reflection.
    }
    else if (!(prd->flags & FLAG_THINWALLED)) // Only non-thinwalled materials have a volume and transmission events.
    {
        prd->flags |= FLAG_TRANSMISSION;
    }

    // No Fresnel factor here. The probability to pick one or the other side took care of that.
    f_over_pdf = state.albedo;
    pdf = 1.0f; // Not 0.0f to make sure the path is not terminated. Otherwise unused for specular events.
}

}