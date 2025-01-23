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

#include "shaders/SystemParameter.h"

#include "util/DebugUtils.h"

namespace jazzfusion
{

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

        // Scene meshes
        std::vector<VertexAttributes *> m_geometryAttibutes;
        std::vector<unsigned int *> m_geometryIndices;
        std::vector<unsigned int> m_geometryAttibuteSize;
        std::vector<unsigned int> m_geometryIndicesSize;

        bool needSceneUpdate = false;
        bool needSceneReloadUpdate = false;
        std::vector<unsigned int> sceneUpdateObjectId;
        std::vector<unsigned int> sceneUpdateInstanceId;
        Float3 *edgeToHighlight;

        int uninstancedGeometryCount;
        int instancedGeometryCount;

        std::unordered_map<int, std::unordered_set<int>> geometryInstanceIdMap;
        std::unordered_map<int, std::array<float, 12>> instanceTransformMatrices;

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

}