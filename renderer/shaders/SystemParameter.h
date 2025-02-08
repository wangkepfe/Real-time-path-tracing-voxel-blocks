#pragma once

#include "LinearMath.h"
#include "AliasTable.h"
#include "Camera.h"
#include "RandGen.h"
#include "Restir.h"
#include <optix.h>

// Just some hardcoded material parameter system which allows to show a few fundamental BSDFs.
// Alignment of all data types used here is 4 bytes.
struct __align__(16) MaterialParameter
{
    TexObj textureAlbedo = 0;
    TexObj textureNormal = 0;
    TexObj textureRoughness = 0;
    TexObj textureMetallic = 0;

    Float3 albedo; // Albedo, tint, throughput change for specular surfaces. Pick your meaning.
    int indexBSDF; // BSDF index to use in the closest hit program

    Float3 absorption; // Absorption coefficient
    float ior;         // Index of refraction

    Float2 texSize = Float2(1024.0f);
    unsigned int flags = 0;
    float uvScale = 1.0f;
};

struct SystemParameter
{
    Camera camera;

    OptixTraversableHandle topObject;

    SurfObj outputBuffer;
    SurfObj outNormal;
    SurfObj outDepth;
    SurfObj outAlbedo;
    SurfObj outMaterial;
    SurfObj outMotionVector;
    SurfObj outUiBuffer;

    MaterialParameter *materialParameters;

    SurfObj skyBuffer;
    SurfObj sunBuffer;
    AliasTable *skyAliasTable;
    AliasTable *sunAliasTable;
    float accumulatedSkyLuminance;
    float accumulatedSunLuminance;
    Int2 skyRes;
    Int2 sunRes;
    Float3 sunDir;

    LightInfo *lights;
    AliasTable *lightAliasTable;

    Float3 *edgeToHighlight;

    int iterationIndex;
    int samplePerIteration;
    int sampleIndex;
    float sceneEpsilon;

    BlueNoiseRandGenerator randGen;
    int accumulationCounter;
    float timeInSecond;
};

struct VertexAttributes
{
    Float3 vertex;
    Float2 texcoord;
};

// SBT Record data for the hit group.
struct GeometryInstanceData
{
    Int3 *indices;
    VertexAttributes *attributes;

    int materialIndex;
};

#ifdef __CUDA_ARCH__
INL_DEVICE float rand(const SystemParameter &sysParam, int &randIdx)
{
    UInt2 idx = UInt2(optixGetLaunchIndex());
    return sysParam.randGen.rand(idx.x, idx.y, sysParam.iterationIndex * sysParam.samplePerIteration + sysParam.sampleIndex, randIdx++);
}

INL_DEVICE Float2 rand2(const SystemParameter &sysParam, int &randIdx)
{
    return Float2(rand(sysParam, randIdx), rand(sysParam, randIdx));
}
#endif // __CUDA_ARCH__