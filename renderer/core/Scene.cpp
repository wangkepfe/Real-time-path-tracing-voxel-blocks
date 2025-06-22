#include "core/Scene.h"
#include "shaders/LinearMath.h"

Scene::Scene()
{
    CUDA_CHECK(cudaMallocManaged(&edgeToHighlight, 4 * sizeof(Float3)));
    CUDA_CHECK(cudaMalloc(&d_lightAliasTable, sizeof(AliasTable)));
}

Scene::~Scene()
{
    CUDA_CHECK(cudaFree(edgeToHighlight));
    CUDA_CHECK(cudaFree(d_lightAliasTable));

    CUDA_CHECK(cudaFree(m_lights));

    CUDA_CHECK(cudaFree(d_instanceLightMapping));

    // Clean up chunk-based geometry buffers
    for (unsigned int chunkIndex = 0; chunkIndex < numChunks; ++chunkIndex)
    {
        for (unsigned int objectId = 0; objectId < m_chunkGeometryAttributes[chunkIndex].size(); ++objectId)
        {
            if (m_chunkGeometryAttributes[chunkIndex][objectId])
            {
                CUDA_CHECK(cudaFree(m_chunkGeometryAttributes[chunkIndex][objectId]));
            }
            if (m_chunkGeometryIndices[chunkIndex][objectId])
            {
                CUDA_CHECK(cudaFree(m_chunkGeometryIndices[chunkIndex][objectId]));
            }
        }
    }

    // Clean up instanced geometry buffers
    for (unsigned int objectId = 0; objectId < m_instancedGeometryAttributes.size(); ++objectId)
    {
        if (m_instancedGeometryAttributes[objectId])
        {
            CUDA_CHECK(cudaFree(m_instancedGeometryAttributes[objectId]));
        }
        if (m_instancedGeometryIndices[objectId])
        {
            CUDA_CHECK(cudaFree(m_instancedGeometryIndices[objectId]));
        }
    }
}

void Scene::initChunkGeometry(unsigned int numChunksParam, unsigned int numObjects)
{
    numChunks = numChunksParam;

    m_chunkGeometryAttributes.resize(numChunks);
    m_chunkGeometryIndices.resize(numChunks);
    m_chunkGeometryAttributeSize.resize(numChunks);
    m_chunkGeometryIndicesSize.resize(numChunks);

    for (unsigned int chunkIndex = 0; chunkIndex < numChunks; ++chunkIndex)
    {
        m_chunkGeometryAttributes[chunkIndex].resize(numObjects, nullptr);
        m_chunkGeometryIndices[chunkIndex].resize(numObjects, nullptr);
        m_chunkGeometryAttributeSize[chunkIndex].resize(numObjects, 0);
        m_chunkGeometryIndicesSize[chunkIndex].resize(numObjects, 0);
    }
}

void Scene::initInstancedGeometry(unsigned int numInstancedObjects)
{
    m_instancedGeometryAttributes.resize(numInstancedObjects, nullptr);
    m_instancedGeometryIndices.resize(numInstancedObjects, nullptr);
    m_instancedGeometryAttributeSize.resize(numInstancedObjects, 0);
    m_instancedGeometryIndicesSize.resize(numInstancedObjects, 0);
}

VertexAttributes** Scene::getChunkGeometryAttributes(unsigned int chunkIndex, unsigned int objectId)
{
    return &m_chunkGeometryAttributes[chunkIndex][objectId];
}

unsigned int** Scene::getChunkGeometryIndices(unsigned int chunkIndex, unsigned int objectId)
{
    return &m_chunkGeometryIndices[chunkIndex][objectId];
}

unsigned int& Scene::getChunkGeometryAttributeSize(unsigned int chunkIndex, unsigned int objectId)
{
    return m_chunkGeometryAttributeSize[chunkIndex][objectId];
}

unsigned int& Scene::getChunkGeometryIndicesSize(unsigned int chunkIndex, unsigned int objectId)
{
    return m_chunkGeometryIndicesSize[chunkIndex][objectId];
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
