#pragma once

#include "LinearMath.h"

struct MaterialState
{
    Float3 normal;
    Float3 geoNormal;
    Float3 albedo;
    Float3 wo;
    float roughness;
    bool metallic;
    float translucency;
};