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
    CUdeviceptr indices;
    CUdeviceptr attributes;
    size_t numIndices;    // Count of unsigned ints, not triplets.
    size_t numAttributes; // Count of VertexAttributes structs.
    CUdeviceptr gas;
};

class Scene
{
public:
    static Scene& Get()
    {
        static Scene instance;
        return instance;
    }
    Scene(Scene const&) = delete;
    void operator=(Scene const&) = delete;

    // static OptixTraversableHandle createBox(OptixFunctionTable& api,
    //     OptixDeviceContext& context,
    //     CUstream cudaStream,
    //     std::vector<GeometryData>& geometries);

    // static OptixTraversableHandle createPlane(OptixFunctionTable& api,
    //     OptixDeviceContext& context,
    //     CUstream cudaStream,
    //     std::vector<GeometryData>& geometries,
    //     const unsigned int tessU,
    //     const unsigned int tessV,
    //     const unsigned int upAxis);

    // static OptixTraversableHandle createSphere(OptixFunctionTable& api,
    //     OptixDeviceContext& context,
    //     CUstream cudaStream,
    //     std::vector<GeometryData>& geometries,
    //     const unsigned int tessU,
    //     const unsigned int tessV,
    //     const float radius,
    //     const float maxTheta);

    // static OptixTraversableHandle createTorus(OptixFunctionTable& api,
    //     OptixDeviceContext& context,
    //     CUstream cudaStream,
    //     std::vector<GeometryData>& geometries,
    //     const unsigned int tessU,
    //     const unsigned int tessV,
    //     const float innerRadius,
    //     const float outerRadius);

    // static OptixTraversableHandle createParallelogram(OptixFunctionTable& api,
    //     OptixDeviceContext& context,
    //     CUstream cudaStream,
    //     std::vector<GeometryData>& geometries,
    //     Float3 const& position,
    //     Float3 const& vecU,
    //     Float3 const& vecV,
    //     Float3 const& normal);



    void createGeometries(
        OptixFunctionTable& api,
        OptixDeviceContext& context,
        CUstream cudaStream,
        std::vector<GeometryData>& geometries,
        std::vector<OptixInstance>& instances);

    void updateGeometry(
        OptixFunctionTable& api,
        OptixDeviceContext& context,
        CUstream cudaStream,
        std::vector<GeometryData>& geometries,
        std::vector<OptixInstance>& instances,
        int objectId);

    // Scene meshes
    std::vector<std::vector<VertexAttributes>> m_geometryAttibutes;
    std::vector<std::vector<uint>>             m_geometryIndices;

    // Scene update callback
    std::function<void(int)> m_updateCallback;

private:
    Scene() {}

    static OptixTraversableHandle CreateGeometry(
        OptixFunctionTable& api,
        OptixDeviceContext& context,
        CUstream cudaStream,
        std::vector<GeometryData>& geometries,
        std::vector<VertexAttributes> const& attributes,
        std::vector<unsigned int> const& indices);


};

}