#pragma once

#include "LinearMath.h"
#include "Camera.h"
#include "RandGen.h"
#include <optix.h>

namespace jazzfusion
{

// Just some hardcoded material parameter system which allows to show a few fundamental BSDFs.
// Alignment of all data types used here is 4 bytes.
struct __align__(16) MaterialParameter
{
    TexObj textureAlbedo = 0;
    TexObj textureNormal = 0;
    TexObj textureRoughness = 0;
    TexObj textureUnused = 0;

    Float3 albedo;      // Albedo, tint, throughput change for specular surfaces. Pick your meaning.
    int indexBSDF;      // BSDF index to use in the closest hit program

    Float3 absorption;  // Absorption coefficient
    float ior;          // Index of refraction

    Float2 texSize = Float2(1024.0f);
    uint flags; // Thin-walled on/off
    float uvScale = 1.0f;
};

enum LightType
{
    LIGHT_ENVIRONMENT = 0, // constant color or spherical environment map.
    LIGHT_PARALLELOGRAM = 1, // Parallelogram area light.

    NUM_LIGHT_TYPES = 2
};

struct __align__(16) LightDefinition
{
    LightType type; // Constant or spherical environment, rectangle (parallelogram).

    // Rectangle lights are defined in world coordinates as footpoint and two vectors spanning a parallelogram.
    // All in world coordinates with no scaling.
    Float3 position;
    Float3 vecU;
    Float3 vecV;
    Float3 normal;
    float  area;
    Float3 emission;

    // Manual padding to float4 alignment goes here.
    float unused0;
    float unused1;
    float unused2;
};

struct LightSample
{
    Float3 position;
    float  distance;
    Float3 direction;
    Float3 emission;
    float  pdf;
};

struct SystemParameter
{
    Camera camera;
    HistoryCamera historyCamera;

    OptixTraversableHandle topObject;

    SurfObj outputBuffer;
    SurfObj outNormal;
    SurfObj outDepth;
    SurfObj outAlbedo;
    SurfObj outMaterial;
    SurfObj outMotionVector;

    LightDefinition* lightDefinitions;

    MaterialParameter* materialParameters;

    // TexObj envTexture;
    // float* envCDF_U; // 2D, size (envWidth + 1) * envHeight
    // float* envCDF_V; // 1D, size (envHeight + 1)
    // uint envWidth; // The original size of the environment texture.
    // uint envHeight;
    // float envIntegral;
    // float envRotation;

    SurfObj skyBuffer;
    SurfObj sunBuffer;
    float* skyCdf;
    float* sunCdf;
    Int2 skyRes;
    Int2 sunRes;
    Float3 sunDir;

    int iterationIndex;
    int samplePerIteration;
    int sampleIndex;
    float sceneEpsilon;

    int numLights;
    BlueNoiseRandGenerator randGen;
    float noiseBlend;
    int accumulationCounter;

    // INL_DEVICE float rand(int& randIdx) const
    // {
    //     UInt2 idx = UInt2(optixGetLaunchIndex());
    //     return randGen.rand(idx.x, idx.y, iterationIndex, randIdx++);
    // }

    // INL_DEVICE Float2 rand2(int& randIdx) const
    // {
    //     return Float2(rand(randIdx), rand(randIdx));
    // }

};

struct VertexAttributes
{
    Float3 vertex;
    Float3 tangent;
    Float3 normal;
    Float3 texcoord;
};

// SBT Record data for the hit group.
struct GeometryInstanceData
{
    Int3* indices;
    VertexAttributes* attributes;

    int materialIndex;
    int lightIndex; // Negative means not a light.
};

}