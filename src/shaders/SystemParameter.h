#pragma once

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
    cudaTextureObject_t textureAlbedo;

    // 4 byte alignment.
    int indexBSDF;      // BSDF index to use in the closest hit program
    float3 albedo;      // Albedo, tint, throughput change for specular surfaces. Pick your meaning.
    float3 absorption;  // Absorption coefficient
    float ior;          // Index of refraction
    unsigned int flags; // Thin-walled on/off
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
    float3 position;
    float3 vecU;
    float3 vecV;
    float3 normal;
    float  area;
    float3 emission;

    // Manual padding to float4 alignment goes here.
    float unused0;
    float unused1;
    float unused2;
};

struct LightSample
{
    float3 position;
    float  distance;
    float3 direction;
    float3 emission;
    float  pdf;
};

struct SystemParameter
{
    // 8 byte alignment
    OptixTraversableHandle topObject;

    cudaSurfaceObject_t outputBuffer;

    LightDefinition* lightDefinitions;

    MaterialParameter* materialParameters;

    cudaTextureObject_t envTexture;

    float* envCDF_U; // 2D, size (envWidth + 1) * envHeight
    float* envCDF_V; // 1D, size (envHeight + 1)

    int2 pathLengths;

    unsigned int envWidth; // The original size of the environment texture.
    unsigned int envHeight;
    float envIntegral;
    float envRotation;

    int iterationIndex;
    float sceneEpsilon;

    int numLights;

    int cameraType;
    float3 cameraPosition;
    float3 cameraU;
    float3 cameraV;
    float3 cameraW;
};

struct VertexAttributes
{
    float3 vertex;
    float3 tangent;
    float3 normal;
    float3 texcoord;
};

// SBT Record data for the hit group.
struct GeometryInstanceData
{
    int3* indices;
    VertexAttributes* attributes;

    int materialIndex;
    int lightIndex; // Negative means not a light.
};