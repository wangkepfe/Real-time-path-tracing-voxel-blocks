#pragma once

// Always include this before any OptiX headers!
#include <cuda_runtime.h>

#include <optix.h>

// OptiX 7 function table structure.
#include <optix_function_table.h>

#include <vector>

#include "shaders/system_parameter.h"
#include "shaders/function_indices.h"
#include "shaders/light_definition.h"
#include "shaders/vertex_attributes.h"
#include "shaders/vector_math.h"

#include "DebugUtils.h"

namespace jazzfusion
{

// The actual geometries are tracked in m_geometries.
struct GeometryData
{
    CUdeviceptr indices;
    CUdeviceptr attributes;
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

    static OptixTraversableHandle createBox(OptixFunctionTable &api,
                                            OptixDeviceContext &context,
                                            CUstream cudaStream,
                                            std::vector<GeometryData> &geometries);

    static OptixTraversableHandle createPlane(OptixFunctionTable &api,
                                                OptixDeviceContext &context,
                                                CUstream cudaStream,
                                                std::vector<GeometryData> &geometries,
                                                const unsigned int tessU,
                                                const unsigned int tessV,
                                                const unsigned int upAxis);

    static OptixTraversableHandle createSphere(OptixFunctionTable &api,
                                                OptixDeviceContext &context,
                                                CUstream cudaStream,
                                                std::vector<GeometryData> &geometries,
                                                const unsigned int tessU,
                                                const unsigned int tessV,
                                                const float radius,
                                                const float maxTheta);

    static OptixTraversableHandle createTorus(OptixFunctionTable &api,
                                                OptixDeviceContext &context,
                                                CUstream cudaStream,
                                                std::vector<GeometryData> &geometries,
                                                const unsigned int tessU,
                                                const unsigned int tessV,
                                                const float innerRadius,
                                                const float outerRadius);

    static OptixTraversableHandle createParallelogram(OptixFunctionTable &api,
                                                        OptixDeviceContext &context,
                                                        CUstream cudaStream,
                                                        std::vector<GeometryData> &geometries,
                                                        float3 const &position,
                                                        float3 const &vecU,
                                                        float3 const &vecV,
                                                        float3 const &normal);

    static OptixTraversableHandle createGeometry(
        OptixFunctionTable &api,
        OptixDeviceContext &context,
        CUstream cudaStream,
        std::vector<GeometryData> &geometries,
        std::vector<VertexAttributes> const &attributes,
        std::vector<unsigned int> const &indices);

private:
    Scene() {}
};

}