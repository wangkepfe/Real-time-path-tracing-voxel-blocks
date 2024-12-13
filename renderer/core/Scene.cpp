#include "core/Scene.h"
#include "shaders/LinearMath.h"

namespace jazzfusion
{

    void Scene::updateGeometry(
        OptixFunctionTable &api,
        OptixDeviceContext &context,
        CUstream cudaStream,
        std::vector<GeometryData> &geometries,
        std::vector<OptixInstance> &instances,
        int objectId)
    {
        CUDA_CHECK(cudaFree((void *)geometries[objectId].indices));
        CUDA_CHECK(cudaFree((void *)geometries[objectId].attributes));
        CUDA_CHECK(cudaFree((void *)geometries[objectId].gas));

        OptixInstance &instance = instances[objectId];
        instance = OptixInstance{};

        OptixTraversableHandle blasHandle = CreateGeometry(api, context, cudaStream, geometries, m_geometryAttibutes[objectId], m_geometryIndices[objectId]);

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
            OptixInstance instance = {};
            OptixTraversableHandle blasHandle = CreateGeometry(api, context, cudaStream, geometries, m_geometryAttibutes[i], m_geometryIndices[i]);

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
        }
    }

    OptixTraversableHandle Scene::CreateGeometry(
        OptixFunctionTable &api,
        OptixDeviceContext &context,
        CUstream cudaStream,
        std::vector<GeometryData> &geometries,
        std::vector<VertexAttributes> const &attributes,
        std::vector<unsigned int> const &indices)
    {
        CUdeviceptr d_attributes;
        CUdeviceptr d_indices;

        const size_t attributesSizeInBytes = sizeof(VertexAttributes) * attributes.size();

        CUDA_CHECK(cudaMalloc((void **)&d_attributes, attributesSizeInBytes));
        CUDA_CHECK(cudaMemcpy((void *)d_attributes, attributes.data(), attributesSizeInBytes, cudaMemcpyHostToDevice));

        const size_t indicesSizeInBytes = sizeof(unsigned int) * indices.size();

        CUDA_CHECK(cudaMalloc((void **)&d_indices, indicesSizeInBytes));
        CUDA_CHECK(cudaMemcpy((void *)d_indices, indices.data(), indicesSizeInBytes, cudaMemcpyHostToDevice));

        OptixBuildInput triangleInput = {};

        triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput.triangleArray.vertexStrideInBytes = sizeof(VertexAttributes);
        triangleInput.triangleArray.numVertices = (unsigned int)attributes.size();
        triangleInput.triangleArray.vertexBuffers = &d_attributes;

        triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput.triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;

        triangleInput.triangleArray.numIndexTriplets = (unsigned int)indices.size() / 3;
        triangleInput.triangleArray.indexBuffer = d_indices;

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
        GeometryData geometry;

        geometry.indices = d_indices;
        geometry.attributes = d_attributes;
        geometry.numIndices = indices.size();
        geometry.numAttributes = attributes.size();
        geometry.gas = d_gas;

        geometries.push_back(geometry);

        return traversableHandle;
    }

} // namespace jazzfusion