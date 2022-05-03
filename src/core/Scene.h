#pragma once

#include <cuda_runtime.h>

#include "Material.h"

namespace jazzfusion {

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

private:
    Scene() {}
    Scene(Scene const&);
    void operator=(Scene const&);

    OptixTraversableHandle createBox();
    OptixTraversableHandle createPlane(const unsigned int tessU, const unsigned int tessV, const unsigned int upAxis);
    OptixTraversableHandle createSphere(const unsigned int tessU, const unsigned int tessV, const float radius, const float maxTheta);
    OptixTraversableHandle createTorus(const unsigned int tessU, const unsigned int tessV, const float innerRadius, const float outerRadius);
    OptixTraversableHandle createParallelogram(float3 const &position, float3 const &vecU, float3 const &vecV, float3 const &normal);
    OptixTraversableHandle createGeometry(std::vector<VertexAttributes> const &attributes, std::vector<unsigned int> const &indices);
    void createLights();
    void updateMaterialParameters();

    std::vector<MaterialParameterGUI> m_guiMaterialParameters;
    std::vector<GeometryData> m_geometries;
};

}