#pragma once

#include "LinearMath.h"
#include "AliasTable.h"
#include "Camera.h"
#include "RandGen.h"
#include "Light.h"
#include "RestirCommon.h"
#include <optix.h>

struct __align__(16) MaterialParameter
{
    // If texture is not 0, use texture
    TexObj textureAlbedo = 0;
    TexObj textureNormal = 0;
    TexObj textureRoughness = 0;
    TexObj textureMetallic = 0;

    // Otherwise fall back to default parameter here
    Float3 albedo = Float3(0.75f);
    float roughness = 0.5f;
    bool metallic = false;
    float translucency = 0.0f;

    // Material ID for sorting rays
    int materialId = -1;

    // Texture scale
    Float2 texSize = Float2(1024.0f);
    float uvScale = 1.0f;

    // Customized UV
    bool useWorldGridUV = false;

    //
    bool isEmissive = false;
    bool isThinfilm = false;
};

struct SystemParameter
{
    Camera camera;
    Camera prevCamera;

    OptixTraversableHandle topObject;
    OptixTraversableHandle prevTopObject;

    SurfObj illuminationBuffer;

    // G buffers
    SurfObj normalRoughnessBuffer;
    SurfObj depthBuffer;
    SurfObj albedoBuffer;
    SurfObj materialBuffer;
    SurfObj geoNormalThinfilmBuffer;
    SurfObj materialParameterBuffer;

    // Previous frame G buffers
    SurfObj prevNormalRoughnessBuffer;
    SurfObj prevDepthBuffer;
    SurfObj prevAlbedoBuffer;
    SurfObj prevMaterialBuffer;
    SurfObj prevGeoNormalThinfilmBuffer;
    SurfObj prevMaterialParameterBuffer;

    SurfObj motionVectorBuffer;
    SurfObj UIBuffer;

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

    InstanceLightMapping *instanceLightMapping;
    unsigned int instanceLightMappingSize;

    LightInfo *lights;
    AliasTable *lightAliasTable;
    float accumulatedLocalLightLuminance;
    unsigned int numLights;
    unsigned int prevNumLights;
    int *prevLightIdToCurrentId;
    bool lightsStateDirty;

    Float3 *edgeToHighlight;

    int iterationIndex;

    BlueNoiseRandGenerator randGen;
    float timeInSecond;

    uint32_t reservoirBlockRowPitch;
    uint32_t reservoirArrayPitch;
    DIReservoir *reservoirBuffer;
    uint8_t *neighborOffsetBuffer;
};

struct VertexAttributes
{
    Float3 vertex;
    Float2 texcoord;
};

struct VertexSkinningData
{
    Int4 jointIndices;   // 4 joint indices for skeletal animation
    Float4 jointWeights; // 4 joint weights for skeletal animation
};

// SBT Record data for the hit group.
struct GeometryInstanceData
{
    Int3 *indices;
    VertexAttributes *attributes;

    int materialIndex;
};

struct SbtRecordHeader
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

template <typename T>
struct SbtRecordData
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecordData<GeometryInstanceData> SbtRecordGeometryInstanceData;

#ifdef __CUDA_ARCH__
INL_DEVICE float rand(const SystemParameter &sysParam, int &randIdx)
{
    UInt2 idx = UInt2(optixGetLaunchIndex());
    return sysParam.randGen.rand(idx.x, idx.y, sysParam.iterationIndex, randIdx++);
}

INL_DEVICE float randPrev(const SystemParameter &sysParam, int &randIdx)
{
    UInt2 idx = UInt2(optixGetLaunchIndex());
    return sysParam.randGen.rand(idx.x, idx.y, sysParam.iterationIndex - 1, randIdx++);
}

INL_DEVICE Float2 rand2(const SystemParameter &sysParam, int &randIdx)
{
    return Float2(rand(sysParam, randIdx), rand(sysParam, randIdx));
}

INL_DEVICE Float3 rand3(const SystemParameter &sysParam, int &randIdx)
{
    return Float3(rand(sysParam, randIdx), rand(sysParam, randIdx), rand(sysParam, randIdx));
}

INL_DEVICE Float4 rand4(const SystemParameter &sysParam, int &randIdx)
{
    return Float4(rand(sysParam, randIdx), rand(sysParam, randIdx), rand(sysParam, randIdx), rand(sysParam, randIdx));
}

INL_DEVICE float rand16bits(const SystemParameter &sysParam, int &randIdx)
{
    Float2 u = rand2(sysParam, randIdx);
    return u.x + u.y / 256.0f;
}
#endif // __CUDA_ARCH__
