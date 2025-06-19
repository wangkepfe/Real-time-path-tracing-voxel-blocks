#pragma once

// Always include this before any OptiX headers!
#include <cuda_runtime.h>

#include <optix.h>

// OptiX 7 function table structure.
#include <optix_function_table.h>

#include <vector>
#include <array>
#include <functional>
#include <unordered_set>
#include <set>

#include "shaders/SystemParameter.h"

#include "util/DebugUtils.h"

// The actual geometries are tracked in m_geometries.
struct GeometryData
{
    unsigned int *indices;
    VertexAttributes *attributes;
    size_t numIndices;    // Count of unsigned ints, not triplets.
    size_t numAttributes; // Count of VertexAttributes structs.
    CUdeviceptr gas;
};

class Scene
{
public:
    static Scene &Get()
    {
        static Scene instance;
        return instance;
    }
    Scene(Scene const &) = delete;
    void operator=(Scene const &) = delete;

    // Multi-chunk geometry management
    // Each chunk has its own list of geometry for each object type
    // Structure: m_chunkGeometryAttributes[chunkIndex][objectId]
    std::vector<std::vector<VertexAttributes *>> m_chunkGeometryAttributes;
    std::vector<std::vector<unsigned int *>> m_chunkGeometryIndices;
    std::vector<std::vector<unsigned int>> m_chunkGeometryAttributeSize;
    std::vector<std::vector<unsigned int>> m_chunkGeometryIndicesSize;

    // Chunk configuration
    unsigned int numChunks = 0;

    bool needSceneUpdate = false;
    bool needSceneReloadUpdate = false;
    std::vector<unsigned int> sceneUpdateObjectId;
    std::vector<unsigned int> sceneUpdateInstanceId;
    std::vector<unsigned int> sceneUpdateChunkId; // Track which chunks need updates
    Float3 *edgeToHighlight;

    int uninstancedGeometryCount;
    int instancedGeometryCount;

    std::unordered_map<int, std::set<int>> geometryInstanceIdMap;
    std::unordered_map<int, std::array<float, 12>> instanceTransformMatrices;

    LightInfo *m_lights = nullptr;
    AliasTable lightAliasTable;
    AliasTable *d_lightAliasTable = nullptr;
    float accumulatedLocalLightLuminance = 0.0f;

    std::vector<InstanceLightMapping> instanceLightMapping;
    InstanceLightMapping *d_instanceLightMapping;
    unsigned int numInstancedLightMesh;

    // Initialize chunk-based geometry buffers
    void initChunkGeometry(unsigned int numChunksParam, unsigned int numObjects);

    // Get geometry data for a specific chunk and object
    VertexAttributes** getChunkGeometryAttributes(unsigned int chunkIndex, unsigned int objectId);
    unsigned int** getChunkGeometryIndices(unsigned int chunkIndex, unsigned int objectId);
    unsigned int& getChunkGeometryAttributeSize(unsigned int chunkIndex, unsigned int objectId);
    unsigned int& getChunkGeometryIndicesSize(unsigned int chunkIndex, unsigned int objectId);

    static OptixTraversableHandle CreateGeometry(
        OptixFunctionTable &api,
        OptixDeviceContext &context,
        CUstream cudaStream,
        GeometryData &geometry,
        VertexAttributes *d_attributes,
        unsigned int *d_indices,
        unsigned int attributeSize,
        unsigned int indicesSize);

private:
    Scene();
    ~Scene();
};