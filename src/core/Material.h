#pragma once

#include <cuda_runtime.h>

namespace jazzfusion {

struct MaterialParameterGUI
{
    int indexBSDF; // BSDF index to use in the closest hit program
    float3 albedo;           // Tint, throughput change for specular materials
    bool useAlbedoTexture;
    bool thinwalled;
    float3 absorptionColor; // absorption color and distance scale together build the absorption coefficient
    float volumeDistanceScale;
    float ior; // index of refraction
};

}