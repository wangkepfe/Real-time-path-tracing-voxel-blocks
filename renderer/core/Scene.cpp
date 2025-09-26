#include "core/Scene.h"
#include "shaders/LinearMath.h"

Scene::Scene()
{
    printf("DEBUG: Scene constructor called - this=%p\n", this);
    CUDA_CHECK(cudaMallocManaged(&edgeToHighlight, 4 * sizeof(Float3)));
    CUDA_CHECK(cudaMalloc(&d_lightAliasTable, sizeof(AliasTable)));

    // Initialize the alias table to empty state
    AliasTable emptyTable;
    CUDA_CHECK(cudaMemcpy(d_lightAliasTable, &emptyTable, sizeof(AliasTable), cudaMemcpyHostToDevice));

    // Pre-allocate light buffer to avoid null pointer issues
    m_maxLightCapacity = 100; // Initial capacity
    CUDA_CHECK(cudaMalloc((void **)&m_lights, m_maxLightCapacity * sizeof(LightInfo)));
    CUDA_CHECK(cudaMemset(m_lights, 0, m_maxLightCapacity * sizeof(LightInfo)));
    m_currentNumLights = 0;

    // Initialize light ID mapping
    m_prevLightIdToCurrentId.resize(m_maxLightCapacity, -1);
}

Scene::~Scene()
{
    if (edgeToHighlight)
    {
        CUDA_CHECK(cudaFree(edgeToHighlight));
    }
    if (d_lightAliasTable)
    {
        CUDA_CHECK(cudaFree(d_lightAliasTable));
    }

    if (m_lights)
    {
        CUDA_CHECK(cudaFree(m_lights));
    }

    if (d_instanceLightMapping)
    {
        CUDA_CHECK(cudaFree(d_instanceLightMapping));
        d_instanceLightMapping = nullptr;
    }

    if (d_prevLightIdToCurrentId)
    {
        CUDA_CHECK(cudaFree(d_prevLightIdToCurrentId));
        d_prevLightIdToCurrentId = nullptr;
    }

    // Note: Geometry memory is owned and freed by OptixRenderer::clear()
    // Scene only holds pointers to this memory, so we don't free it here to avoid double-free errors
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

VertexAttributes **Scene::getChunkGeometryAttributes(unsigned int chunkIndex, unsigned int objectId)
{
    return &m_chunkGeometryAttributes[chunkIndex][objectId];
}

unsigned int **Scene::getChunkGeometryIndices(unsigned int chunkIndex, unsigned int objectId)
{
    return &m_chunkGeometryIndices[chunkIndex][objectId];
}

unsigned int &Scene::getChunkGeometryAttributeSize(unsigned int chunkIndex, unsigned int objectId)
{
    return m_chunkGeometryAttributeSize[chunkIndex][objectId];
}

unsigned int &Scene::getChunkGeometryIndicesSize(unsigned int chunkIndex, unsigned int objectId)
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
    unsigned int indicesSize,
    bool allowUpdate)
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

    // Set build flags based on whether updates are allowed
    accelBuildOptions.buildFlags = allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : OPTIX_BUILD_FLAG_NONE;
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

OptixTraversableHandle Scene::UpdateGeometry(
    OptixFunctionTable &api,
    OptixDeviceContext &context,
    CUstream cudaStream,
    GeometryData &geometry,
    VertexAttributes *d_attributes,
    unsigned int *d_indices,
    unsigned int attributeSize,
    unsigned int indicesSize)
{
    // Validate inputs
    if (!d_attributes || !d_indices || attributeSize == 0 || indicesSize == 0 || geometry.gas == 0)
    {
        return 0;
    }

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

    // Use update operation for existing BLAS
    accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accelBuildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;

    OptixAccelBufferSizes accelBufferSizes;

    OPTIX_CHECK(api.optixAccelComputeMemoryUsage(context, &accelBuildOptions, &triangleInput, 1, &accelBufferSizes));

    CUdeviceptr d_tmp;
    CUDA_CHECK(cudaMalloc((void **)&d_tmp, accelBufferSizes.tempUpdateSizeInBytes));

    OptixTraversableHandle traversableHandle = 0; // This is the GAS handle which gets returned.

    OPTIX_CHECK(api.optixAccelBuild(context, cudaStream,
                                    &accelBuildOptions, &triangleInput, 1,
                                    d_tmp, accelBufferSizes.tempUpdateSizeInBytes,
                                    geometry.gas, accelBufferSizes.outputSizeInBytes,
                                    &traversableHandle, nullptr, 0));

    CUDA_CHECK(cudaStreamSynchronize(cudaStream));

    CUDA_CHECK(cudaFree((void *)d_tmp));

    // Update the GeometryData
    geometry.indices = d_indices;
    geometry.attributes = d_attributes;
    geometry.numAttributes = attributeSize;
    geometry.numIndices = indicesSize;

    return traversableHandle;
}

// Entity management functions
void Scene::addEntity(std::unique_ptr<Entity> entity)
{
    m_entities.push_back(std::move(entity));
    needSceneUpdate = true;
}

void Scene::removeEntity(size_t index)
{
    if (index < m_entities.size())
    {
        m_entities.erase(m_entities.begin() + index);
        needSceneUpdate = true;
    }
}

void Scene::clearEntities()
{
    m_entities.clear();
    needSceneUpdate = true;
}