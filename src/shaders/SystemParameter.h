#pragma once

#include "LinearMath.h"
#include "Camera.h"
#include <optix.h>

namespace jazzfusion
{

enum FunctionIndexSpecular
{
    INDEX_BSDF_SPECULAR_REFLECTION = 0,
    INDEX_BSDF_SPECULAR_REFLECTION_TRANSMISSION = 1,
    NUM_SPECULAR_BSDF = 2,
};

enum FunctionIndexDiffuse
{
    INDEX_BSDF_DIFFUSE_REFLECTION = 2,
};

// Just some hardcoded material parameter system which allows to show a few fundamental BSDFs.
// Alignment of all data types used here is 4 bytes.
struct __align__(16) MaterialParameter
{
    // 8 byte alignment.
    TexObj textureAlbedo;

    // 4 byte alignment.
    int indexBSDF;      // BSDF index to use in the closest hit program
    Float3 albedo;      // Albedo, tint, throughput change for specular surfaces. Pick your meaning.
    Float3 absorption;  // Absorption coefficient
    float ior;          // Index of refraction
    uint flags; // Thin-walled on/off
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

    TexObj envTexture;

    float* envCDF_U; // 2D, size (envWidth + 1) * envHeight
    float* envCDF_V; // 1D, size (envHeight + 1)

    uint envWidth; // The original size of the environment texture.
    uint envHeight;
    float envIntegral;
    float envRotation;

    int iterationIndex;
    float sceneEpsilon;

    int numLights;
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