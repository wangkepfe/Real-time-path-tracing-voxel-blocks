#pragma once

// Always include this before any OptiX headers!
#include <cuda_runtime.h>

#include <optix.h>

// OptiX 7 function table structure.
#include <optix_function_table.h>

#include <vector>
#include <functional>

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
        void createGeometries(
            OptixFunctionTable &api,
            OptixDeviceContext &context,
            CUstream cudaStream,
            std::vector<GeometryData> &geometries,
            std::vector<OptixInstance> &instances);

        void updateGeometry(
            OptixFunctionTable &api,
            OptixDeviceContext &context,
            CUstream cudaStream,
            std::vector<GeometryData> &geometries,
            std::vector<OptixInstance> &instances,
            int objectId);

        // Scene meshes
        std::vector<VertexAttributes *> m_geometryAttibutes;
        std::vector<unsigned int *> m_geometryIndices;
        std::vector<unsigned int> m_geometryAttibuteSize;
        std::vector<unsigned int> m_geometryIndicesSize;

        bool needSceneUpdate = false;
        std::vector<unsigned int> sceneUpdateObjectId;
        Float3 *edgeToHighlight;

    private:
        Scene();
        ~Scene();

        static OptixTraversableHandle CreateGeometry(
            OptixFunctionTable &api,
            OptixDeviceContext &context,
            CUstream cudaStream,
            GeometryData &geometry,
            VertexAttributes *d_attributes,
            unsigned int *d_indices,
            unsigned int attributeSize,
            unsigned int indicesSize);
    };

}