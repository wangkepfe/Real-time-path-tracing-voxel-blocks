#pragma once

// Always include this before any OptiX headers!
#include <cuda_runtime.h>

#include <optix.h>

// OptiX 7 function table structure.
#include <optix_function_table.h>

#include <unordered_map>
#include <functional>

#include <vector>
#include <array>
#include <functional>
#include <unordered_set>
#include <set>
#include <memory>

#include "shaders/SystemParameter.h"
#include "Entity.h"

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

    // Multi-chunk geometry management for uninstanced objects
    // Each chunk has its own list of geometry for each object type
    // Structure: m_chunkGeometryAttributes[chunkIndex][objectId]
    std::vector<std::vector<VertexAttributes *>> m_chunkGeometryAttributes;
    std::vector<std::vector<unsigned int *>> m_chunkGeometryIndices;
    std::vector<std::vector<unsigned int>> m_chunkGeometryAttributeSize;
    std::vector<std::vector<unsigned int>> m_chunkGeometryIndicesSize;

    // Instanced geometry arrays (shared across all chunks)
    // These have fixed geometry and are used to create BLAS once per object type
    std::vector<VertexAttributes *> m_instancedGeometryAttributes;
    std::vector<unsigned int *> m_instancedGeometryIndices;
    std::vector<unsigned int> m_instancedGeometryAttributeSize;
    std::vector<unsigned int> m_instancedGeometryIndicesSize;

    // Chunk configuration
    unsigned int numChunks = 0;

    bool needSceneUpdate = false;
    bool needSceneReloadUpdate = false;
    std::vector<unsigned int> sceneUpdateObjectId;
    std::vector<unsigned int> sceneUpdateInstanceId;
    std::vector<unsigned int> sceneUpdateChunkId; // Track which chunks need updates
    Float3 *edgeToHighlight = nullptr;

    int uninstancedGeometryCount;
    int instancedGeometryCount;

    std::unordered_map<int, std::set<int>> geometryInstanceIdMap;
    std::unordered_map<int, std::array<float, 12>> instanceTransformMatrices;

    LightInfo *m_lights = nullptr;
    AliasTable lightAliasTable;
    AliasTable *d_lightAliasTable = nullptr;
    float accumulatedLocalLightLuminance = 0.0f;

    std::vector<InstanceLightMapping> instanceLightMapping;
    InstanceLightMapping *d_instanceLightMapping = nullptr;
    unsigned int instanceLightMappingSize = 0;

    // Dynamic light management
    unsigned int m_currentNumLights = 0;
    unsigned int m_prevNumLights = 0;
    unsigned int m_maxLightCapacity = 0;
    bool m_lightsNeedUpdate = false;

    // Light ID mapping for temporal coherence (ReSTIR)
    // Maps previous frame light array index to current frame light array index
    // -1 means the light no longer exists
    std::vector<int> m_prevLightIdToCurrentId;
    int *d_prevLightIdToCurrentId = nullptr;

    // Entity management
    std::vector<std::unique_ptr<Entity>> m_entities;

    // Initialize chunk-based geometry buffers
    void initChunkGeometry(unsigned int numChunksParam, unsigned int numObjects);

    // Initialize instanced geometry buffers
    void initInstancedGeometry(unsigned int numInstancedObjects);

    // Get geometry data for a specific chunk and object
    VertexAttributes **getChunkGeometryAttributes(unsigned int chunkIndex, unsigned int objectId);
    unsigned int **getChunkGeometryIndices(unsigned int chunkIndex, unsigned int objectId);
    unsigned int &getChunkGeometryAttributeSize(unsigned int chunkIndex, unsigned int objectId);
    unsigned int &getChunkGeometryIndicesSize(unsigned int chunkIndex, unsigned int objectId);

public:
    // Entity management functions
    void addEntity(std::unique_ptr<Entity> entity);
    void removeEntity(size_t index);
    void clearEntities();
    size_t getEntityCount() const { return m_entities.size(); }
    Entity *getEntity(size_t index) { return index < m_entities.size() ? m_entities[index].get() : nullptr; }
    const std::vector<std::unique_ptr<Entity>> &getEntities() const { return m_entities; }

    static OptixTraversableHandle CreateGeometry(
        OptixFunctionTable &api,
        OptixDeviceContext &context,
        CUstream cudaStream,
        GeometryData &geometry,
        VertexAttributes *d_attributes,
        unsigned int *d_indices,
        unsigned int attributeSize,
        unsigned int indicesSize,
        bool allowUpdate = false);

    static OptixTraversableHandle UpdateGeometry(
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