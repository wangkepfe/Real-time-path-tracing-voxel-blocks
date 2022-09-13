#include "SystemParameter.h"
#include "OptixShaderCommon.h"

namespace jazzfusion
{

extern "C" __constant__ SystemParameter sysParam;

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

extern "C" __device__ void __direct_callable__sample_bsdf_diffuse_reflection(MaterialParameter const& parameters, State const& state, PerRayData * rayData, Float3 & wi, Float3 & f_over_pdf, float& pdf)
{
    unitSquareToCosineHemisphere(rayData->rand2(sysParam), state.normal, wi, pdf);

    if (!(rayData->flags & FLAG_DIFFUSED))
    {
        rayData->material |= RAY_MAT_FLAG_DIFFUSE << (2 * rayData->depth);
    }

    // if (pdf <= 0.0f || dot(wi, state.normalGeo) <= 0.0f)
    if (pdf <= 0.0f || dot(wi, state.normal) <= 0.0f)
    {
        rayData->flags |= FLAG_TERMINATE;
        return;
    }

    f_over_pdf = Float3(1.0f);
}

// The parameter wiL is the lightSample.direction (direct lighting), not the next ray segment's direction rayData.wi (indirect lighting).
extern "C" __device__ Float4 __direct_callable__eval_bsdf_diffuse_reflection(MaterialParameter const& parameters, State const& state, PerRayData* const rayData, const Float3 wiL)
{
    const Float3 f = Float3(1.0f) * M_1_PIf;
    const float pdf = fmaxf(0.0f, dot(wiL, state.normal) * M_1_PIf);

    return Float4(f, pdf);
}


extern "C" __device__ void __direct_callable__sample_bsdf_specular_reflection(MaterialParameter const& parameters, State const& state, PerRayData * rayData, Float3 & wi, Float3 & f_over_pdf, float& pdf)
{
    wi = reflect3f(-rayData->wo, state.normal);

    // Do not sample opaque materials below the geometric surface.
    // if (pdf <= 0.0f || dot(wi, state.normalGeo) <= 0.0f)
    if (pdf <= 0.0f || dot(wi, state.normal) <= 0.0f)
    {
        rayData->flags |= FLAG_TERMINATE;
        return;
    }

    if (!(rayData->flags & FLAG_DIFFUSED))
    {
        rayData->material |= RAY_MAT_FLAG_REFL_OR_REFR << (2 * rayData->depth);

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

extern "C" __device__ void __direct_callable__sample_bsdf_specular_reflection_transmission(MaterialParameter const& parameters, State const& state, PerRayData * rayData, Float3 & wi, Float3 & f_over_pdf, float& pdf)
{
    // Return the current material's absorption coefficient and ior to the integrator to be able to support nested materials.
    rayData->absorption_ior = Float4(parameters.absorption, parameters.ior);

    // Need to figure out here which index of refraction to use if the ray is already inside some refractive medium.
    // This needs to happen with the original FLAG_FRONTFACE condition to find out from which side of the geometry we're looking!
    // ior.xy are the current volume's IOR and the surrounding volume's IOR.
    // Thin-walled materials have no volume, always use the frontface eta for them!
    const float eta = (rayData->flags & (FLAG_FRONTFACE | FLAG_THINWALLED))
        ? rayData->absorption_ior.w / rayData->ior.x
        : rayData->ior.y / rayData->absorption_ior.w;

    const Float3 R = reflect3f(-rayData->wo, state.normal);

    float reflective = 1.0f;

    if (refract(wi, -rayData->wo, state.normal, eta))
    {
        if (rayData->flags & FLAG_THINWALLED)
        {
            wi = -rayData->wo; // Straight through, no volume.
        }
        // Total internal reflection will leave this reflection probability at 1.0f.
        reflective = evaluateFresnelDielectric(eta, dot(rayData->wo, state.normal));
    }

    if (rayData->flags & FLAG_VOLUME) // If we are inside a volumn
    {
        if (!(rayData->flags & FLAG_DIFFUSED))
        {
            rayData->material |= RAY_MAT_FLAG_REFL_OR_REFR << (2 * rayData->depth);
        }

        if (reflective == 1.0f) // Either total reflection
        {
            wi = R;
        }
        else // Or total transmission
        {
            rayData->flags |= FLAG_TRANSMISSION;
        }
        f_over_pdf = Float3(1.0f);
    }
    else
    {
        if (!(rayData->flags & FLAG_DIFFUSED))
        {
            rayData->material |= RAY_MAT_FLAG_REFR_AND_REFL << (2 * rayData->depth);
        }

        if (rayData->sampleIdx & 0x1)
        {
            wi = R; // Fresnel reflection or total internal reflection.
            f_over_pdf = Float3(reflective);
        }
        else if (!(rayData->flags & FLAG_THINWALLED)) // Only non-thinwalled materials have a volume and transmission events.
        {
            rayData->flags |= FLAG_TRANSMISSION;
            f_over_pdf = Float3(1.0f - reflective);
        }

        // const float pseudo = rayData->rand(sysParam);
        // if (pseudo < reflective)
        // {
        //     wi = R; // Fresnel reflection or total internal reflection.
        // }
        // else if (!(rayData->flags & FLAG_THINWALLED)) // Only non-thinwalled materials have a volume and transmission events.
        // {
        //     rayData->flags |= FLAG_TRANSMISSION;
        // }
    }

    // No Fresnel factor here. The probability to pick one or the other side took care of that.
    // f_over_pdf = Float3(1.0f);
    pdf = 1.0f; // Not 0.0f to make sure the path is not terminated. Otherwise unused for specular events.
}

}