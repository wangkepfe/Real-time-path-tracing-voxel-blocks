#include "core/Scene.h"
#include "shaders/LinearMath.h"

namespace jazzfusion
{

    Scene::Scene()
    {
        CUDA_CHECK(cudaMallocManaged(&edgeToHighlight, 4 * sizeof(Float3)));
    }

    Scene::~Scene()
    {
        CUDA_CHECK(cudaFree(edgeToHighlight));
    }

    void Scene::updateGeometry(
        OptixFunctionTable &api,
        OptixDeviceContext &context,
        CUstream cudaStream,
        std::vector<GeometryData> &geometries,
        std::vector<OptixInstance> &instances,
        int objectId)
    {
        GeometryData &geometry = geometries[objectId];

        CUDA_CHECK(cudaFree((void *)geometry.gas));

        OptixInstance &instance = instances[objectId];
        OptixTraversableHandle blasHandle = CreateGeometry(api, context, cudaStream, geometry, m_geometryAttibutes[objectId], m_geometryIndices[objectId], m_geometryAttibuteSize[objectId], m_geometryIndicesSize[objectId]);

        const float transformMatrix[12] =
            {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f};

        memcpy(instance.transform, transformMatrix, sizeof(float) * 12);

        instance.instanceId = objectId;
        instance.visibilityMask = 255;
        instance.sbtOffset = objectId;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = blasHandle;
    }

    void Scene::createGeometries(
        OptixFunctionTable &api,
        OptixDeviceContext &context,
        CUstream cudaStream,
        std::vector<GeometryData> &geometries,
        std::vector<OptixInstance> &instances)
    {
        for (int i = 0; i < m_geometryAttibutes.size(); ++i)
        {
            GeometryData geometry = {};

            OptixInstance instance = {};
            OptixTraversableHandle blasHandle = CreateGeometry(api, context, cudaStream, geometry, m_geometryAttibutes[i], m_geometryIndices[i], m_geometryAttibuteSize[i], m_geometryIndicesSize[i]);

            const float transformMatrix[12] =
                {
                    1.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 1.0f, 0.0f};

            memcpy(instance.transform, transformMatrix, sizeof(float) * 12);

            instance.instanceId = i;
            instance.visibilityMask = 255;
            instance.sbtOffset = i;
            instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            instance.traversableHandle = blasHandle;

            instances.push_back(instance);
            geometries.push_back(geometry);
        }
    }

    OptixTraversableHandle Scene::CreateGeometry(
        OptixFunctionTable &api,
        OptixDeviceContext &context,
        CUstream cudaStream,
        GeometryData &geometry,
        VertexAttributes *d_attributes,
        unsigned int *d_indices,
        unsigned int attributeSize,
        unsigned int indicesSize)
    {
        const size_t attributesSizeInBytes = sizeof(VertexAttributes) * attributeSize;
        const size_t indicesSizeInBytes = sizeof(unsigned int) * indicesSize;

        OptixBuildInput triangleInput = {};

        triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput.triangleArray.vertexStrideInBytes = sizeof(VertexAttributes);
        triangleInput.triangleArray.numVertices = attributeSize;
        triangleInput.triangleArray.vertexBuffers = (const CUdeviceptr *)(&d_attributes);

        triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput.triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;

        triangleInput.triangleArray.numIndexTriplets = indicesSize / 3;
        triangleInput.triangleArray.indexBuffer = (CUdeviceptr)d_indices;

        unsigned int triangleInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

        triangleInput.triangleArray.flags = triangleInputFlags;
        triangleInput.triangleArray.numSbtRecords = 1;

        OptixAccelBuildOptions accelBuildOptions = {};

        accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
        accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes accelBufferSizes;

        OPTIX_CHECK(api.optixAccelComputeMemoryUsage(context, &accelBuildOptions, &triangleInput, 1, &accelBufferSizes));

        CUdeviceptr d_gas; // This holds the geometry acceleration structure.

        CUDA_CHECK(cudaMalloc((void **)&d_gas, accelBufferSizes.outputSizeInBytes));

        CUdeviceptr d_tmp;

        CUDA_CHECK(cudaMalloc((void **)&d_tmp, accelBufferSizes.tempSizeInBytes));

        OptixTraversableHandle traversableHandle = 0; // This is the GAS handle which gets returned.

        OPTIX_CHECK(api.optixAccelBuild(context, cudaStream,
                                        &accelBuildOptions, &triangleInput, 1,
                                        d_tmp, accelBufferSizes.tempSizeInBytes,
                                        d_gas, accelBufferSizes.outputSizeInBytes,
                                        &traversableHandle, nullptr, 0));

        CUDA_CHECK(cudaStreamSynchronize(cudaStream));

        CUDA_CHECK(cudaFree((void *)d_tmp));

        // Track the GeometryData to be able to set them in the SBT record GeometryInstanceData and free them on exit.
        geometry.indices = d_indices;
        geometry.attributes = d_attributes;
        geometry.numAttributes = attributeSize;
        geometry.numIndices = indicesSize;
        geometry.gas = d_gas;

        return traversableHandle;
    }

} // namespace jazzfusion