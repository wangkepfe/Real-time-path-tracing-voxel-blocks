#include "core/Scene.h"
#include "shaders/LinearMath.h"
#include <vector>

Scene::Scene()
{
    CUDA_CHECK(cudaMallocManaged(&edgeToHighlight, 4 * sizeof(Float3)));
    CUDA_CHECK(cudaMalloc(&d_lightAliasTable, sizeof(AliasTable)));
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
    }

    // Note: Geometry memory is owned and freed by OptixRenderer::clear()
    // Scene only holds pointers to this memory, so we don't free it here to avoid double-free errors
}

OptixTraversableHandle Scene::CreateDummyGeometry(
    OptixFunctionTable &api,
    OptixDeviceContext &context,
    CUstream cudaStream,
    GeometryData &geometry)
{
    printf("CREATING DUMMY GEOMETRY: Fallback for empty scene\n");
    
    // Create a single tiny triangle as dummy geometry
    std::vector<VertexAttributes> dummyVertices(3);
    dummyVertices[0] = {Float3(0.0f, 0.0f, 0.0f), Float2(0.0f, 0.0f), Int4(0), Float4(0.0f)};
    dummyVertices[1] = {Float3(0.1f, 0.0f, 0.0f), Float2(1.0f, 0.0f), Int4(0), Float4(0.0f)};
    dummyVertices[2] = {Float3(0.0f, 0.1f, 0.0f), Float2(0.0f, 1.0f), Int4(0), Float4(0.0f)};
    
    std::vector<unsigned int> dummyIndices = {0, 1, 2};
    
    // Allocate GPU memory
    VertexAttributes *d_attributes;
    unsigned int *d_indices;
    
    CUDA_CHECK(cudaMalloc(&d_attributes, sizeof(VertexAttributes) * 3));
    CUDA_CHECK(cudaMalloc(&d_indices, sizeof(unsigned int) * 3));
    
    CUDA_CHECK(cudaMemcpy(d_attributes, dummyVertices.data(), sizeof(VertexAttributes) * 3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, dummyIndices.data(), sizeof(unsigned int) * 3, cudaMemcpyHostToDevice));
    
    // Use existing CreateGeometry function for consistency
    return CreateGeometry(api, context, cudaStream, geometry, d_attributes, d_indices, 3, 3);
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
    unsigned int indicesSize,
    bool allowUpdate)
{
    // BULLETPROOF VALIDATION: Comprehensive input validation to prevent CUDA errors
    
    // 1. NULL pointer validation
    if (!d_attributes || !d_indices) {
        printf("GEOMETRY VALIDATION ERROR: Null geometry pointers (d_attributes=%p, d_indices=%p)\n", 
               d_attributes, d_indices);
        geometry.indices = nullptr;
        geometry.attributes = nullptr;
        geometry.numAttributes = 0;
        geometry.numIndices = 0;
        geometry.gas = 0;
        return 0;
    }
    
    // 2. Empty geometry validation
    if (attributeSize == 0 || indicesSize == 0) {
        printf("GEOMETRY VALIDATION INFO: Empty geometry (vertices=%u, indices=%u) - skipping\n", 
               attributeSize, indicesSize);
        geometry.indices = nullptr;
        geometry.attributes = nullptr;
        geometry.numAttributes = 0;
        geometry.numIndices = 0;
        geometry.gas = 0;
        return 0;
    }
    
    // 3. Triangle count validation
    if (indicesSize % 3 != 0) {
        printf("GEOMETRY VALIDATION ERROR: Invalid triangle count (indices=%u, not multiple of 3)\n", 
               indicesSize);
        geometry.indices = nullptr;
        geometry.attributes = nullptr;
        geometry.numAttributes = 0;
        geometry.numIndices = 0;
        geometry.gas = 0;
        return 0;
    }
    
    // 4. Minimum geometry threshold
    if (indicesSize < 3 || attributeSize < 3) {
        printf("GEOMETRY VALIDATION ERROR: Insufficient geometry data (vertices=%u, indices=%u)\n", 
               attributeSize, indicesSize);
        geometry.indices = nullptr;
        geometry.attributes = nullptr;
        geometry.numAttributes = 0;
        geometry.numIndices = 0;
        geometry.gas = 0;
        return 0;
    }
    
    // 5. CRITICAL: Index bounds validation - prevent illegal memory access
    // Read indices on CPU to validate bounds (expensive but necessary for stability)
    std::vector<unsigned int> hostIndices(indicesSize);
    cudaError_t copyResult = cudaMemcpy(hostIndices.data(), d_indices, 
                                       sizeof(unsigned int) * indicesSize, 
                                       cudaMemcpyDeviceToHost);
    
    if (copyResult != cudaSuccess) {
        printf("GEOMETRY VALIDATION ERROR: Failed to copy indices for validation: %s\n", 
               cudaGetErrorString(copyResult));
        geometry.indices = nullptr;
        geometry.attributes = nullptr;
        geometry.numAttributes = 0;
        geometry.numIndices = 0;
        geometry.gas = 0;
        return 0;
    }
    
    // Check all triangle vertex indices are within bounds
    unsigned int maxValidIndex = attributeSize - 1;
    for (unsigned int i = 0; i < indicesSize; i++) {
        if (hostIndices[i] >= attributeSize) {
            printf("GEOMETRY VALIDATION ERROR: Index out of bounds at position %u: index=%u, max_valid=%u\n", 
                   i, hostIndices[i], maxValidIndex);
            geometry.indices = nullptr;
            geometry.attributes = nullptr;
            geometry.numAttributes = 0;
            geometry.numIndices = 0;
            geometry.gas = 0;
            return 0;
        }
    }
    
    // 6. Memory size validation to prevent overflow
    const size_t maxReasonableGeometry = 1024 * 1024 * 1024; // 1GB limit
    const size_t totalGeometrySize = (sizeof(VertexAttributes) * attributeSize) + 
                                    (sizeof(unsigned int) * indicesSize);
    
    if (totalGeometrySize > maxReasonableGeometry) {
        printf("GEOMETRY VALIDATION ERROR: Geometry too large (%zu bytes > %zu bytes limit)\n", 
               totalGeometrySize, maxReasonableGeometry);
        geometry.indices = nullptr;
        geometry.attributes = nullptr;
        geometry.numAttributes = 0;
        geometry.numIndices = 0;
        geometry.gas = 0;
        return 0;
    }
    
    printf("GEOMETRY VALIDATION SUCCESS: vertices=%u, triangles=%u, size=%zu bytes\n", 
           attributeSize, indicesSize/3, totalGeometrySize);

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

    printf("GEOMETRY TRACKING: Created geometry with numIndices=%zu, numAttributes=%zu, d_indices=%p, d_attributes=%p\n", 
           geometry.numIndices, geometry.numAttributes, geometry.indices, geometry.attributes);
    
    // DEBUG: Sample vertices from both beginning and end of buffer
    if (attributeSize >= 6) {
        std::vector<VertexAttributes> sampleVertices(6);
        
        // Get first 3 vertices
        cudaError_t copyResult1 = cudaMemcpy(sampleVertices.data(), d_attributes, 
                                           sizeof(VertexAttributes) * 3, cudaMemcpyDeviceToHost);
        
        // Get last 3 vertices  
        cudaError_t copyResult2 = cudaMemcpy(&sampleVertices[3], 
                                           d_attributes + (attributeSize - 3), 
                                           sizeof(VertexAttributes) * 3, cudaMemcpyDeviceToHost);
        
        if (copyResult1 == cudaSuccess && copyResult2 == cudaSuccess) {
            printf("VERTEX DEBUG: First 3 vertices:\n");
            for (int i = 0; i < 3; i++) {
                printf("  v[%d]: pos=(%.2f, %.2f, %.2f) texcoord=(%.2f, %.2f)\n", i,
                       sampleVertices[i].vertex.x, sampleVertices[i].vertex.y, sampleVertices[i].vertex.z,
                       sampleVertices[i].texcoord.x, sampleVertices[i].texcoord.y);
            }
            printf("VERTEX DEBUG: Last 3 vertices (indices %u-%u):\n", attributeSize-3, attributeSize-1);
            for (int i = 3; i < 6; i++) {
                printf("  v[%d]: pos=(%.2f, %.2f, %.2f) texcoord=(%.2f, %.2f)\n", attributeSize-6+i,
                       sampleVertices[i].vertex.x, sampleVertices[i].vertex.y, sampleVertices[i].vertex.z,
                       sampleVertices[i].texcoord.x, sampleVertices[i].texcoord.y);
            }
        }
    }
    
    // DEBUG: Sample a few triangle indices to verify they're valid
    if (indicesSize >= 3) {
        std::vector<unsigned int> sampleIndices(3);
        cudaError_t copyResult = cudaMemcpy(sampleIndices.data(), d_indices, 
                                          sizeof(unsigned int) * 3, cudaMemcpyDeviceToHost);
        if (copyResult == cudaSuccess) {
            printf("INDEX DEBUG: First triangle indices: (%u, %u, %u) - max vertex index should be < %u\n",
                   sampleIndices[0], sampleIndices[1], sampleIndices[2], attributeSize);
        }
    }

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
    // BULLETPROOF VALIDATION: UpdateGeometry input validation
    
    // 1. NULL pointer validation
    if (!d_attributes || !d_indices) {
        printf("UPDATE GEOMETRY ERROR: Null pointers (d_attributes=%p, d_indices=%p)\n", 
               d_attributes, d_indices);
        return 0;
    }
    
    // 2. Empty geometry validation
    if (attributeSize == 0 || indicesSize == 0) {
        printf("UPDATE GEOMETRY ERROR: Empty geometry (vertices=%u, indices=%u)\n", 
               attributeSize, indicesSize);
        return 0;
    }
    
    // 3. Existing GAS validation
    if (geometry.gas == 0) {
        printf("UPDATE GEOMETRY ERROR: No existing GAS to update\n");
        return 0;
    }
    
    // 4. Triangle count validation
    if (indicesSize % 3 != 0) {
        printf("UPDATE GEOMETRY ERROR: Invalid triangle count (indices=%u, not multiple of 3)\n", 
               indicesSize);
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
