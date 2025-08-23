#include "VoxelEngine.h"
#include "VoxelMath.h"
#include "voxelengine/BlockType.h"

#include "core/Scene.h"
#ifndef OFFLINE_MODE
#include "core/InputHandler.h"
#endif
#include "core/RenderCamera.h"
#include "core/Character.h"
#include "core/Entity.h"
#include "core/SceneConfig.h"
#include "core/GlobalSettings.h"

#include "assets/ModelUtils.h"
#include "assets/ModelManager.h"
#include "assets/BlockManager.h"
#include "assets/AssetRegistry.h"
#include "assets/MaterialManager.h"
#include "shaders/ShaderDebugUtils.h"

#ifndef OFFLINE_MODE
#include <GLFW/glfw3.h>
#else
#include <chrono>
#endif

// Helper function: applies a 3x4 transform (stored as 12 floats in row-major order)
// to a given 3D position.
__device__ inline Float3 applyTransform(const Float3 &v, const float *t)
{
    // t[0..3] is the first row, t[4..7] is the second row, t[8..11] is the third row.
    return Float3(
        t[0] * v.x + t[1] * v.y + t[2] * v.z + t[3],
        t[4] * v.x + t[5] * v.y + t[6] * v.z + t[7],
        t[8] * v.x + t[9] * v.y + t[10] * v.z + t[11]);
}

//-----------------------------------------------------------------------------
// CUDA kernel to generate LightInfo entries for each instance of a mesh.
// Inputs:
//   - vertices: pointer to VertexAttributes array (each must contain a member "position" of type Float3).
//   - indices: pointer to unsigned int index array (assumed to form triangles).
//   - numIndices: total number of indices (must be a multiple of 3).
//   - transforms: pointer to a flattened array of instance transforms (each instance has 12 floats).
//   - numInstances: number of instance transforms.
//   - radiance: emissive radiance value (a scalar); to get radiant intensity multiply by area.
// Output:
//   - output: pointer to device buffer of LightInfo, with length = (numIndices/3) * numInstances.
__global__ void generateLightInfosKernel(const VertexAttributes *vertices,
                                         const unsigned int *indices,
                                         unsigned int numIndices,
                                         const float *transforms, // flattened; each instance: 12 floats.
                                         unsigned int numInstances,
                                         Float3 radiance,
                                         LightInfo *globalLights,
                                         unsigned int globalOffset,
                                         unsigned int maxLightCapacity)
{
    // Compute total number of triangles in the mesh.
    unsigned int numTriangles = numIndices / 3;

    // Each thread handles one triangle for one instance.
    unsigned int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int totalWork = numTriangles * numInstances;
    if (globalId >= totalWork)
        return;

    // Map the global id to an instance index and a triangle index.
    unsigned int instanceIdx = globalId / numTriangles;
    unsigned int triangleIdx = globalId % numTriangles;

    // Pointer to the current instance's transform (12 floats).
    const float *t = &transforms[instanceIdx * 12];

    // Read the three vertex indices for this triangle.
    unsigned int idx0 = indices[triangleIdx * 3 + 0];
    unsigned int idx1 = indices[triangleIdx * 3 + 1];
    unsigned int idx2 = indices[triangleIdx * 3 + 2];

    // Load the triangle's vertices (assuming "position" is a Float3 member of VertexAttributes).
    Float3 v0ModelSpace = vertices[idx0].vertex;
    Float3 v1ModelSpace = vertices[idx1].vertex;
    Float3 v2ModelSpace = vertices[idx2].vertex;

    // Transform each vertex by the instance transform.
    Float3 v0 = applyTransform(v0ModelSpace, t);
    Float3 v1 = applyTransform(v1ModelSpace, t);
    Float3 v2 = applyTransform(v2ModelSpace, t);

    // Create a TriangleLight. We choose v0 as the "base" and compute edge vectors:
    TriangleLight triLight;
    triLight.base = v0;       // Base point.
    triLight.edge1 = v1 - v0; // Edge from v0 to v1.
    triLight.edge2 = v2 - v0; // Edge from v0 to v2.

    // Compute the area of the triangle.
    Float3 crossVal = cross(triLight.edge1, triLight.edge2);
    float area = 0.5f * length(crossVal);

    triLight.radiance = radiance;

    // Pack the triangle light data into a LightInfo using the provided interface.
    LightInfo li = triLight.Store();

    // Write out the result with bounds checking to prevent buffer overflow.
    unsigned int outputIndex = globalOffset + globalId;
    if (outputIndex < maxLightCapacity)
    {
        globalLights[outputIndex] = li;
    }
    // Note: If bounds check fails, we silently drop this light to prevent crash
}

//-----------------------------------------------------------------------------
// Example host code to launch the kernel.
//
// Note: Adjust grid/block sizes as needed and ensure that the device arrays
//       (d_vertices, d_indices, d_transforms, d_lightInfos) are allocated and filled.
void launchGenerateLightInfos(const VertexAttributes *d_vertices,
                              const unsigned int *d_indices,
                              unsigned int numIndices,
                              const float *d_transforms, // device array with numInstances*12 floats
                              unsigned int numInstances,
                              Float3 radiance,
                              LightInfo *d_lightInfos,
                              unsigned int currentGlobalOffset, // output buffer of size (numIndices/3)*numInstances
                              unsigned int maxLightCapacity)
{
    unsigned int numTriangles = numIndices / 3;
    unsigned int totalWork = numTriangles * numInstances; // Number of triangle lights for the current type of geometry
    const unsigned int blockSize = 256;
    unsigned int numBlocks = (totalWork + blockSize - 1) / blockSize;

    generateLightInfosKernel<<<numBlocks, blockSize>>>(d_vertices,
                                                       d_indices,
                                                       numIndices,
                                                       d_transforms,
                                                       numInstances,
                                                       radiance,
                                                       d_lightInfos,
                                                       currentGlobalOffset,
                                                       maxLightCapacity);

    // Check for errors and synchronize as needed.
    cudaDeviceSynchronize();
}

__global__ void extractRadianceKernel(const LightInfo *lights, float *d_radiance, unsigned int numLights)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numLights)
    {
        TriangleLight triLight = TriangleLight::Create(lights[idx]);
        d_radiance[idx] = luminance(triLight.radiance) * triLight.surfaceArea;
    }
}

//-----------------------------------------------------------------------------
// Step 2. Build the device radiance array and update the alias table.
void buildAliasTable(LightInfo *d_lights, unsigned int totalLights, AliasTable &aliasTable, float &accumulatedLocalLightLuminance)
{
    // Allocate device memory to store one radiance weight per light.
    float *d_radiance = nullptr;
    cudaError_t err = cudaMalloc((void **)&d_radiance, totalLights * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "Error allocating radiance buffer: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Launch a kernel to extract radiance weights from the merged LightInfo buffer.
    const unsigned int blockSize = 256;
    unsigned int numBlocks = (totalLights + blockSize - 1) / blockSize;
    extractRadianceKernel<<<numBlocks, blockSize>>>(d_lights, d_radiance, totalLights);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "Error synchronizing after extractRadianceKernel: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_radiance);
        return;
    }

    // Build the alias table based on the radiance weights
    aliasTable.update(d_radiance, totalLights, accumulatedLocalLightLuminance);

    // Optionally, you can free the d_radiance buffer if it is no longer needed.
    cudaFree(d_radiance);
}

static void MouseButtonCallback(int button, int action, int mods)
{
    if (button == 0 && action == 1)
    {
        VoxelEngine::Get().leftMouseButtonClicked = true;
    }
}

VoxelEngine::~VoxelEngine()
{
}

// Coordinate conversion helper functions
unsigned int VoxelEngine::getChunkIndex(unsigned int chunkX, unsigned int chunkY, unsigned int chunkZ) const
{
    return chunkX + chunkConfig.chunksX * (chunkZ + chunkConfig.chunksZ * chunkY);
}

void VoxelEngine::globalToChunkCoords(unsigned int globalX, unsigned int globalY, unsigned int globalZ,
                                      unsigned int &chunkX, unsigned int &chunkY, unsigned int &chunkZ,
                                      unsigned int &localX, unsigned int &localY, unsigned int &localZ) const
{
    chunkX = globalX / VoxelChunk::width;
    chunkY = globalY / VoxelChunk::width;
    chunkZ = globalZ / VoxelChunk::width;

    localX = globalX % VoxelChunk::width;
    localY = globalY % VoxelChunk::width;
    localZ = globalZ % VoxelChunk::width;
}

void VoxelEngine::chunkToGlobalCoords(unsigned int chunkX, unsigned int chunkY, unsigned int chunkZ,
                                      unsigned int localX, unsigned int localY, unsigned int localZ,
                                      unsigned int &globalX, unsigned int &globalY, unsigned int &globalZ) const
{
    globalX = chunkX * VoxelChunk::width + localX;
    globalY = chunkY * VoxelChunk::width + localY;
    globalZ = chunkZ * VoxelChunk::width + localZ;
}

Voxel VoxelEngine::getVoxelAtGlobal(unsigned int globalX, unsigned int globalY, unsigned int globalZ) const
{
    unsigned int chunkX, chunkY, chunkZ, localX, localY, localZ;
    globalToChunkCoords(globalX, globalY, globalZ, chunkX, chunkY, chunkZ, localX, localY, localZ);

    // Check bounds
    if (chunkX >= chunkConfig.chunksX || chunkY >= chunkConfig.chunksY || chunkZ >= chunkConfig.chunksZ)
    {
        Voxel emptyVoxel;
        emptyVoxel.id = BlockTypeEmpty;
        return emptyVoxel;
    }

    unsigned int chunkIndex = getChunkIndex(chunkX, chunkY, chunkZ);
    return voxelChunks[chunkIndex].get(localX, localY, localZ);
}

void VoxelEngine::setVoxelAtGlobal(unsigned int globalX, unsigned int globalY, unsigned int globalZ, unsigned int blockId)
{
    unsigned int chunkX, chunkY, chunkZ, localX, localY, localZ;
    globalToChunkCoords(globalX, globalY, globalZ, chunkX, chunkY, chunkZ, localX, localY, localZ);

    // Check bounds
    if (chunkX >= chunkConfig.chunksX || chunkY >= chunkConfig.chunksY || chunkZ >= chunkConfig.chunksZ)
        return;

    unsigned int chunkIndex = getChunkIndex(chunkX, chunkY, chunkZ);
    voxelChunks[chunkIndex].set(localX, localY, localZ, blockId);
}

void VoxelEngine::initInstanceGeometry()
{
    auto &scene = Scene::Get();
    auto &sceneGeometryAttributes = scene.m_instancedGeometryAttributes;
    auto &sceneGeometryIndices = scene.m_instancedGeometryIndices;
    auto &sceneGeometryAttributeSize = scene.m_instancedGeometryAttributeSize;
    auto &sceneGeometryIndicesSize = scene.m_instancedGeometryIndicesSize;

    for (unsigned int objectId = Assets::BlockManager::Get().getInstancedObjectIdBegin(); objectId < Assets::BlockManager::Get().getInstancedObjectIdEnd(); ++objectId)
    {
        // Convert objectId to array index (0-based)
        unsigned int arrayIndex = objectId - Assets::BlockManager::Get().getInstancedObjectIdBegin();

        sceneGeometryAttributeSize[arrayIndex] = 0;
        sceneGeometryIndicesSize[arrayIndex] = 0;

        unsigned int blockId = Assets::BlockManager::Get().objectIdToBlockId(objectId);

        // Get geometry from ModelManager instead of loading directly
        const Assets::LoadedGeometry *geometry = Assets::ModelManager::Get().getGeometryForBlock(blockId);

        if (geometry && geometry->d_attributes && geometry->d_indices)
        {
            // Reference the ModelManager's loaded geometry directly (we don't own this memory)
            sceneGeometryAttributes[arrayIndex] = geometry->d_attributes;
            sceneGeometryAttributeSize[arrayIndex] = static_cast<unsigned int>(geometry->attributeSize);

            // ModelManager stores Int3* but VoxelEngine expects unsigned int*
            // We need to convert Int3* to unsigned int* by casting
            // Since Int3 contains 3 consecutive unsigned ints (x, y, z), we can cast the pointer
            sceneGeometryIndices[arrayIndex] = reinterpret_cast<unsigned int *>(geometry->d_indices);
            // Each Int3 contains 3 indices, so total unsigned int count = triangleCount * 3
            sceneGeometryIndicesSize[arrayIndex] = static_cast<unsigned int>(geometry->triangleCount * 3);
        }
        else
        {
            std::cerr << "Failed to get geometry for block type " << blockId << " from ModelManager" << std::endl;
            sceneGeometryAttributes[arrayIndex] = nullptr;
            sceneGeometryIndices[arrayIndex] = nullptr;
            sceneGeometryAttributeSize[arrayIndex] = 0;
            sceneGeometryIndicesSize[arrayIndex] = 0;
        }
    }
}

void VoxelEngine::collectInstanceTransforms()
{
    auto &scene = Scene::Get();
    auto &geometryInstanceIdMap = scene.geometryInstanceIdMap;
    auto &instanceTransformMatrices = scene.instanceTransformMatrices;

    geometryInstanceIdMap.clear();
    instanceTransformMatrices.clear();

    unsigned int globalWidth = chunkConfig.getGlobalWidth();

    for (unsigned int objectId = Assets::BlockManager::Get().getInstancedObjectIdBegin(); objectId < Assets::BlockManager::Get().getInstancedObjectIdEnd(); ++objectId)
    {
        unsigned int blockId = Assets::BlockManager::Get().objectIdToBlockId(objectId);

        for (unsigned int globalX = 0; globalX < chunkConfig.getGlobalWidth(); ++globalX)
        {
            for (unsigned int globalY = 0; globalY < chunkConfig.getGlobalHeight(); ++globalY)
            {
                for (unsigned int globalZ = 0; globalZ < chunkConfig.getGlobalDepth(); ++globalZ)
                {
                    auto val = getVoxelAtGlobal(globalX, globalY, globalZ);
                    // Check if this is a light base matching the current light block
                    bool isLightBase = Assets::BlockManager::Get().hasLightBase(blockId) && (val.id == Assets::BlockManager::Get().getLightBaseBlockId(blockId));
                    if (val.id == blockId || isLightBase)
                    {
                        unsigned int instanceId = PositionToInstanceId(Assets::BlockManager::Get().getNumUninstancedBlockTypes(), objectId, globalX, globalY, globalZ, globalWidth);

                        geometryInstanceIdMap[objectId].insert(instanceId);

                        std::array<float, 12> transform = {1.0f, 0.0f, 0.0f, (float)globalX,
                                                           0.0f, 1.0f, 0.0f, (float)globalY,
                                                           0.0f, 0.0f, 1.0f, (float)globalZ};

                        instanceTransformMatrices[instanceId] = transform;
                    }
                }
            }
        }
    }
}

unsigned int VoxelEngine::countLightTriangles()
{
    auto &scene = Scene::Get();
    auto &sceneGeometryIndicesSize = scene.m_instancedGeometryIndicesSize;
    auto &geometryInstanceIdMap = scene.geometryInstanceIdMap;

    unsigned int totalNumTriLights = 0;

    for (unsigned int objectId = Assets::BlockManager::Get().getInstancedObjectIdBegin(); objectId < Assets::BlockManager::Get().getInstancedObjectIdEnd(); ++objectId)
    {
        unsigned int arrayIndex = objectId - Assets::BlockManager::Get().getInstancedObjectIdBegin();
        unsigned int blockId = Assets::BlockManager::Get().objectIdToBlockId(objectId);

        if (Assets::BlockManager::Get().isEmissive(blockId))
        {
            unsigned int numInstances = geometryInstanceIdMap[objectId].size();
            unsigned int numTriPerInstance = sceneGeometryIndicesSize[arrayIndex] / 3;
            totalNumTriLights += numTriPerInstance * numInstances;
        }
    }

    return totalNumTriLights;
}

void VoxelEngine::generateInstanceLights(unsigned int totalNumTriLights)
{
    auto &scene = Scene::Get();

    // Early return if no lights to generate
    if (totalNumTriLights == 0)
    {
        scene.instanceLightMapping.clear();
        return;
    }

    auto &sceneGeometryAttributes = scene.m_instancedGeometryAttributes;
    auto &sceneGeometryIndices = scene.m_instancedGeometryIndices;
    auto &sceneGeometryIndicesSize = scene.m_instancedGeometryIndicesSize;
    auto &geometryInstanceIdMap = scene.geometryInstanceIdMap;
    auto &instanceTransformMatrices = scene.instanceTransformMatrices;

    // Dynamic resize if needed
    if (totalNumTriLights > scene.m_maxLightCapacity)
    {
        // Free old buffer if it exists
        if (scene.m_lights != nullptr)
        {
            cudaFree(scene.m_lights);
            scene.m_lights = nullptr;
        }

        // Allocate new buffer with extra capacity
        scene.m_maxLightCapacity = totalNumTriLights + 100; // Add some headroom
        cudaMalloc((void **)&scene.m_lights, scene.m_maxLightCapacity * sizeof(LightInfo));
    }
    else if (scene.m_lights == nullptr && totalNumTriLights > 0)
    {
        // First time allocation
        scene.m_maxLightCapacity = totalNumTriLights + 100;
        cudaMalloc((void **)&scene.m_lights, scene.m_maxLightCapacity * sizeof(LightInfo));
    }

    // Generate light info
    unsigned int currentGlobalOffset = 0;
    scene.instanceLightMapping.clear();

    for (unsigned int objectId = Assets::BlockManager::Get().getInstancedObjectIdBegin(); objectId < Assets::BlockManager::Get().getInstancedObjectIdEnd(); ++objectId)
    {
        unsigned int arrayIndex = objectId - Assets::BlockManager::Get().getInstancedObjectIdBegin();
        unsigned int blockId = Assets::BlockManager::Get().objectIdToBlockId(objectId);

        if (Assets::BlockManager::Get().isEmissive(blockId))
        {
            unsigned int numInstances = geometryInstanceIdMap[objectId].size();
            unsigned int numTriPerInstance = sceneGeometryIndicesSize[arrayIndex] / 3;

            std::vector<std::array<float, 12>> transforms;
            unsigned int lightOffset = currentGlobalOffset;
            for (unsigned int instanceId : geometryInstanceIdMap[objectId])
            {
                transforms.push_back(instanceTransformMatrices[instanceId]);
                scene.instanceLightMapping.push_back(InstanceLightMapping{instanceId, lightOffset, numTriPerInstance});
                lightOffset += numTriPerInstance;
            }

            // Get emissive radiance from the material associated with this block
            Float3 radiance = Assets::MaterialManager::Get().getEmissiveRadianceForBlock(blockId);

            float *d_transforms = nullptr;
            size_t transformsSizeInBytes = numInstances * 12 * sizeof(float);
            cudaMalloc((void **)&d_transforms, transformsSizeInBytes);
            cudaMemcpy(d_transforms, &(transforms[0][0]), transformsSizeInBytes, cudaMemcpyHostToDevice);

            launchGenerateLightInfos(sceneGeometryAttributes[arrayIndex], sceneGeometryIndices[arrayIndex], sceneGeometryIndicesSize[arrayIndex], d_transforms, numInstances, radiance, scene.m_lights, currentGlobalOffset, scene.m_maxLightCapacity);

            cudaFree(d_transforms);
            currentGlobalOffset += numTriPerInstance * numInstances;
        }
    }
}

void VoxelEngine::uploadInstanceLightMapping()
{
    auto &scene = Scene::Get();

    // Upload instance light mapping
    scene.instanceLightMappingSize = scene.instanceLightMapping.size();

    // Only allocate and copy if we have mappings
    if (scene.instanceLightMappingSize > 0)
    {
        size_t mappingSizeInBytes = scene.instanceLightMappingSize * sizeof(InstanceLightMapping);
        cudaMalloc((void **)&scene.d_instanceLightMapping, mappingSizeInBytes);
        cudaMemcpy(scene.d_instanceLightMapping, scene.instanceLightMapping.data(), mappingSizeInBytes, cudaMemcpyHostToDevice);
    }
}

void VoxelEngine::buildLightAliasTable(unsigned int totalNumTriLights)
{
    auto &scene = Scene::Get();

    if (totalNumTriLights > 0)
    {
        // Build alias table. If we have an existing table and the size is different
        if (scene.lightAliasTable.initialized() && scene.lightAliasTable.size() != totalNumTriLights)
        {
            scene.lightAliasTable = AliasTable(); // trigger the destructor and constructor
        }
        buildAliasTable(scene.m_lights, totalNumTriLights, scene.lightAliasTable, scene.accumulatedLocalLightLuminance);
    }
    else
    {
        // No lights - reset alias table
        scene.lightAliasTable = AliasTable();
        scene.accumulatedLocalLightLuminance = 0.0f;
    }

    cudaMemcpy(scene.d_lightAliasTable, &scene.lightAliasTable, sizeof(AliasTable), cudaMemcpyHostToDevice);
}

void VoxelEngine::updateLight()
{
    auto &scene = Scene::Get();

    if (!scene.m_lightsNeedUpdate)
    {
        return;
    }

    // Save previous light count for ReSTIR
    scene.m_prevNumLights = scene.m_currentNumLights;

    // Clear any existing light mapping data
    if (scene.d_instanceLightMapping != nullptr)
    {
        cudaFree(scene.d_instanceLightMapping);
        scene.d_instanceLightMapping = nullptr;
        scene.instanceLightMappingSize = 0;
    }

    // Count total light triangles
    unsigned int totalNumTriLights = countLightTriangles();

    scene.accumulatedLocalLightLuminance = 0.0f;

    // Generate instance lights (handles dynamic resizing internally)
    generateInstanceLights(totalNumTriLights);

    // Upload light mapping
    uploadInstanceLightMapping();

    // Build light alias table
    buildLightAliasTable(totalNumTriLights);

    // Update current light count
    scene.m_currentNumLights = totalNumTriLights;

    scene.m_lightsJustUpdated = true;
    scene.m_lightsNeedUpdate = false;

    cudaDeviceSynchronize();
}

// Multi-chunk version with chunk-specific face tracking
void VoxelEngine::updateUninstancedMeshes(const std::vector<Voxel *> &d_dataChunks)
{
    auto &scene = Scene::Get();

    // Process each chunk in parallel for each object type
    for (unsigned int chunkIndex = 0; chunkIndex < chunkConfig.getTotalChunks(); ++chunkIndex)
    {
        for (int objectId = Assets::BlockManager::Get().getUninstancedObjectIdBegin(); objectId < Assets::BlockManager::Get().getUninstancedObjectIdEnd(); ++objectId)
        {
            int blockId = Assets::BlockManager::Get().objectIdToBlockId(objectId);

            // Reset chunk-specific geometry data
            scene.getChunkGeometryAttributeSize(chunkIndex, objectId) = 0;
            scene.getChunkGeometryIndicesSize(chunkIndex, objectId) = 0;

            // Reset chunk-specific face tracking data
            currentFaceCount[chunkIndex][objectId] = 0;
            maxFaceCount[chunkIndex][objectId] = 0;

            // Generate mesh for this specific chunk and object
            generateMesh(
                scene.getChunkGeometryAttributes(chunkIndex, objectId),
                scene.getChunkGeometryIndices(chunkIndex, objectId),
                faceLocation[chunkIndex][objectId],
                scene.getChunkGeometryAttributeSize(chunkIndex, objectId),
                scene.getChunkGeometryIndicesSize(chunkIndex, objectId),
                currentFaceCount[chunkIndex][objectId],
                maxFaceCount[chunkIndex][objectId],
                voxelChunks[chunkIndex],
                d_dataChunks[chunkIndex],
                blockId);
        }
    }
}

void VoxelEngine::init()
{
#ifndef OFFLINE_MODE
    // Input handling
    auto &inputHandler = InputHandler::Get();
    inputHandler.setMouseButtonCallbackFunc(MouseButtonCallback);
#endif

    auto &scene = Scene::Get();

    scene.uninstancedGeometryCount = Assets::BlockManager::Get().getNumUninstancedBlockTypes();
    scene.instancedGeometryCount = Assets::BlockManager::Get().getNumInstancedBlockTypes();

    // Initialize chunk-based geometry buffers for uninstanced objects
    scene.initChunkGeometry(chunkConfig.getTotalChunks(), Assets::BlockManager::Get().getNumUninstancedBlockTypes());

    // Initialize instanced geometry buffers
    scene.initInstancedGeometry(Assets::BlockManager::Get().getNumInstancedBlockTypes());

    // Initialize chunk-specific face tracking buffers
    unsigned int numChunks = chunkConfig.getTotalChunks();
    unsigned int numObjects = Assets::BlockManager::Get().getNumUninstancedBlockTypes();

    faceLocation.resize(numChunks);
    currentFaceCount.resize(numChunks);
    maxFaceCount.resize(numChunks);
    freeFaces.resize(numChunks);

    for (unsigned int chunkIndex = 0; chunkIndex < numChunks; ++chunkIndex)
    {
        faceLocation[chunkIndex].resize(numObjects);
        currentFaceCount[chunkIndex].resize(numObjects);
        maxFaceCount[chunkIndex].resize(numObjects);
        freeFaces[chunkIndex].resize(numObjects);
    }

    // Init multiple voxel chunks
    voxelChunks.resize(chunkConfig.getTotalChunks());

    // Initialize all chunks
    std::vector<Voxel *> d_dataChunks(chunkConfig.getTotalChunks());
    for (unsigned int i = 0; i < chunkConfig.getTotalChunks(); ++i)
    {
        initVoxelsMultiChunk(voxelChunks[i], &d_dataChunks[i], i, chunkConfig);
    }

    // Create geometry mesh based on the voxel grid
    updateUninstancedMeshes(d_dataChunks);

    // Free device data for all chunks
    for (auto &d_data : d_dataChunks)
    {
        freeDeviceVoxelData(d_data);
    }

    // Init instanced meshes
    initInstanceGeometry();
    collectInstanceTransforms();

    scene.m_lightsNeedUpdate = true;
    updateLight();

    // Initialize entities
    initEntities();
}

void VoxelEngine::reload()
{
    auto &scene = Scene::Get();

    // Upload voxel data from disk load for all chunks
    std::vector<Voxel *> d_dataChunks(chunkConfig.getTotalChunks());
    for (unsigned int chunkIndex = 0; chunkIndex < chunkConfig.getTotalChunks(); ++chunkIndex)
    {
        size_t totalVoxels = VoxelChunk::width * VoxelChunk::width * VoxelChunk::width;
        cudaMalloc(&d_dataChunks[chunkIndex], totalVoxels * sizeof(Voxel));
        cudaMemcpy(d_dataChunks[chunkIndex], voxelChunks[chunkIndex].data, totalVoxels * sizeof(Voxel), cudaMemcpyHostToDevice);
    }

    // Generate uninstanced meshes for all chunks
    updateUninstancedMeshes(d_dataChunks);

    // Free device data for all chunks
    for (auto &d_data : d_dataChunks)
    {
        freeDeviceVoxelData(d_data);
    }

    collectInstanceTransforms();

    scene.m_lightsNeedUpdate = true;
    updateLight();

    scene.needSceneUpdate = true;
    scene.needSceneReloadUpdate = true;
}

void VoxelEngine::update()
{
    // Reset center block info
    centerBlockInfo.hasValidBlock = false;
    centerBlockInfo.blockId = 0;
    centerBlockInfo.blockName = "Empty";
    centerBlockInfo.position = Int3(-1, -1, -1);

    // Use unified time management from GlobalSettings
    float deltaTime = GlobalSettings::GetDeltaTime();

    // Update entities
    auto &scene = Scene::Get();
    for (int i = 0; i < scene.getEntityCount(); ++i)
    {
        Entity *entity = scene.getEntity(i);
        if (entity)
        {
            entity->update(deltaTime);
        }
    }

    cudaDeviceSynchronize();

    // Perform ray traversal to find target block
    RayTraversalResult rayResult = performRayTraversal();

    // Calculate and store edge highlighting for hit voxel
    if (rayResult.hitSurface)
    {
        // Edge highlighting calculation (kept inline as it's view-dependent)
        float half = 0.5f;
        Float3 voxelCorners[8] = {
            Float3(rayResult.hitX - half, rayResult.hitY - half, rayResult.hitZ - half),
            Float3(rayResult.hitX + half, rayResult.hitY - half, rayResult.hitZ - half),
            Float3(rayResult.hitX + half, rayResult.hitY + half, rayResult.hitZ - half),
            Float3(rayResult.hitX - half, rayResult.hitY + half, rayResult.hitZ - half),
            Float3(rayResult.hitX - half, rayResult.hitY - half, rayResult.hitZ + half),
            Float3(rayResult.hitX + half, rayResult.hitY - half, rayResult.hitZ + half),
            Float3(rayResult.hitX + half, rayResult.hitY + half, rayResult.hitZ + half),
            Float3(rayResult.hitX - half, rayResult.hitY + half, rayResult.hitZ + half)};

        // Store face corners for edge highlighting (simplified - could be extracted to function)
        Float3 corners[4] = {voxelCorners[0], voxelCorners[3], voxelCorners[2], voxelCorners[1]};

        Scene::Get().edgeToHighlight[0] = corners[0];
        Scene::Get().edgeToHighlight[1] = corners[1];
        Scene::Get().edgeToHighlight[2] = corners[2];
        Scene::Get().edgeToHighlight[3] = corners[3];
    }

    // Handle mouse input for block placement/deletion
    if (leftMouseButtonClicked)
    {
        leftMouseButtonClicked = false;

#ifndef OFFLINE_MODE
        int blockId = InputHandler::Get().currentSelectedBlockId;
#else
        // Test sequence: place light block (16) → remove (0) → place light block (16)
        static int clickCount = 0;
        int testSequence[] = {16, 0, 16}; // First light, delete, second light
        int blockId = testSequence[clickCount % 3];
        clickCount++;
        auto &debugCamera = RenderCamera::Get().camera;
        std::cout << "CAMERA RAY DEBUG: Camera pos=(" << debugCamera.pos.x << "," << debugCamera.pos.y << "," << debugCamera.pos.z << ")" << std::endl;
        std::cout << "CAMERA RAY DEBUG: Camera dir=(" << debugCamera.dir.x << "," << debugCamera.dir.y << "," << debugCamera.dir.z << ")" << std::endl;
        if (rayResult.hitSurface)
        {
            std::cout << "CAMERA RAY DEBUG: Hit block at (" << rayResult.hitX << "," << rayResult.hitY << "," << rayResult.hitZ << ")" << std::endl;
        }
        if (rayResult.hasSpaceToCreate)
        {
            std::cout << "CAMERA RAY DEBUG: Will place light block at (" << rayResult.createPos.x << "," << rayResult.createPos.y << "," << rayResult.createPos.z << ")" << std::endl;
        }
#endif

        if (blockId == 0) // Delete block
        {
            if (rayResult.hitSurface)
            {
                deleteBlock(rayResult.deletePos, rayResult.deleteBlockId);
            }
        }
        else // Create block
        {
            if (rayResult.hasSpaceToCreate && rayResult.hitSurface)
            {
                addBlock(rayResult.createPos, blockId);
            }
        }
    }

    updateLight();
}

// Refactored functions for better maintainability

void VoxelEngine::initEntities()
{
    auto &scene = Scene::Get();

    // Clear any existing entities
    scene.clearEntities();
    // Create a hardcoded Minecraft character entity
    // Position it in front of the current camera for guaranteed visibility
    EntityTransform transform;

    // Get current camera position and direction
    auto &camera = RenderCamera::Get().camera;
    Float3 cameraPos = camera.pos;
    Float3 cameraDir = camera.dir;

    // Check if camera values are valid (not NaN, not zero vector)
    bool validCameraPos = !isnan(cameraPos.x) && !isnan(cameraPos.y) && !isnan(cameraPos.z);
    bool validCameraDir = !isnan(cameraDir.x) && !isnan(cameraDir.y) && !isnan(cameraDir.z) &&
                          (cameraDir.x != 0.0f || cameraDir.y != 0.0f || cameraDir.z != 0.0f);

    // Use character position from scene config if available
    // Try to load from scene config file first
    std::string sceneConfigFile = "data/scene/scene_export.yaml";
    SceneConfig sceneConfig;
    bool configLoaded = SceneConfigParser::LoadFromFile(sceneConfigFile, sceneConfig);

    if (configLoaded)
    {
        // Use character position from config
        transform.position = sceneConfig.character.position;
        transform.rotation = sceneConfig.character.rotation;
        transform.scale = sceneConfig.character.scale;
    }
    else
    {
        // Fallback to default position
        transform.position = Float3(16.0f, 10.0f, 16.0f);
        transform.rotation = Float3(0.0f, 0.0f, 0.0f);
        transform.scale = Float3(1.0f, 1.0f, 1.0f);
        std::cout << "Failed to load scene config file" << std::endl;
    }

    // Create Character instead of Entity for controllable character
    auto minecraftCharacter = std::make_unique<Character>(transform);

#ifndef OFFLINE_MODE
    // Set up the character with the input handler for control
    auto &inputHandler = InputHandler::Get();
    Character *characterPtr = minecraftCharacter.get();
    inputHandler.setCharacter(characterPtr);
#endif

    // Add character to scene (Character inherits from Entity)
    scene.addEntity(std::move(minecraftCharacter));
}

// Refactored functions for better maintainability

VoxelEngine::RayTraversalResult VoxelEngine::performRayTraversal()
{
    RayTraversalResult result;

    auto &camera = RenderCamera::Get().camera;
    Ray ray{camera.pos, camera.dir};

    // Normalize the ray direction
    float len = std::sqrt(ray.dir.x * ray.dir.x + ray.dir.y * ray.dir.y + ray.dir.z * ray.dir.z);
    if (len <= 1e-8f)
    {
        return result; // Degenerate direction vector
    }

    ray.dir.x /= len;
    ray.dir.y /= len;
    ray.dir.z /= len;

    // Find the starting voxel indices
    int x = static_cast<int>(std::floor(ray.orig.x));
    int y = static_cast<int>(std::floor(ray.orig.y));
    int z = static_cast<int>(std::floor(ray.orig.z));

    // Determine step direction along each axis
    int stepX = (ray.dir.x > 0.0f) ? 1 : -1;
    int stepY = (ray.dir.y > 0.0f) ? 1 : -1;
    int stepZ = (ray.dir.z > 0.0f) ? 1 : -1;

    // Compute tDeltaX, tDeltaY, tDeltaZ
    float tDeltaX = (std::fabs(ray.dir.x) < 1e-8f) ? FLT_MAX : (1.0f / std::fabs(ray.dir.x));
    float tDeltaY = (std::fabs(ray.dir.y) < 1e-8f) ? FLT_MAX : (1.0f / std::fabs(ray.dir.y));
    float tDeltaZ = (std::fabs(ray.dir.z) < 1e-8f) ? FLT_MAX : (1.0f / std::fabs(ray.dir.z));

    float nextBoundaryX = (stepX > 0) ? (float)(x + 1) : (float)(x);
    float nextBoundaryY = (stepY > 0) ? (float)(y + 1) : (float)(y);
    float nextBoundaryZ = (stepZ > 0) ? (float)(z + 1) : (float)(z);

    float tMaxX = (std::fabs(ray.dir.x) < 1e-8f) ? FLT_MAX : (nextBoundaryX - ray.orig.x) / ray.dir.x;
    float tMaxY = (std::fabs(ray.dir.y) < 1e-8f) ? FLT_MAX : (nextBoundaryY - ray.orig.y) / ray.dir.y;
    float tMaxZ = (std::fabs(ray.dir.z) < 1e-8f) ? FLT_MAX : (nextBoundaryZ - ray.orig.z) / ray.dir.z;

    int iterationCount = 0;
    int maxIteration = 1000;
    int hitAxis = -1;

    // Traverse the voxel grid
    while (iterationCount++ < maxIteration)
    {
        // Check if the voxel is within bounds
        if (x < 0 || x >= (int)chunkConfig.getGlobalWidth() ||
            y < 0 || y >= (int)chunkConfig.getGlobalHeight() ||
            z < 0 || z >= (int)chunkConfig.getGlobalDepth())
        {
            break;
        }

        Voxel voxel = getVoxelAtGlobal(x, y, z);

        if (voxel.id == 0)
        {
            // Empty voxel: we can potentially place something here
            result.hasSpaceToCreate = true;
            result.createPos = Int3(x, y, z);
        }
        else
        {
            // Solid voxel hit
            result.hitSurface = true;
            result.hitX = x;
            result.hitY = y;
            result.hitZ = z;
            result.deletePos = Int3(x, y, z);
            result.deleteBlockId = voxel.id;

            // Store center block information for GUI
            centerBlockInfo.hasValidBlock = true;
            centerBlockInfo.blockId = voxel.id;
            centerBlockInfo.position = Int3(x, y, z);

            const std::string *blockName = Assets::BlockManager::Get().getBlockName(voxel.id);
            if (blockName)
            {
                centerBlockInfo.blockName = *blockName;
            }
            else
            {
                centerBlockInfo.blockName = "Unknown";
            }

            break;
        }

        // Move to the next voxel
        if (tMaxX < tMaxY)
        {
            if (tMaxX < tMaxZ)
            {
                x += stepX;
                tMaxX += tDeltaX;
                hitAxis = 0;
            }
            else
            {
                z += stepZ;
                tMaxZ += tDeltaZ;
                hitAxis = 2;
            }
        }
        else
        {
            if (tMaxY < tMaxZ)
            {
                y += stepY;
                tMaxY += tDeltaY;
                hitAxis = 1;
            }
            else
            {
                z += stepZ;
                tMaxZ += tDeltaZ;
                hitAxis = 2;
            }
        }
    }

    return result;
}

void VoxelEngine::deleteBlock(const Int3 &pos, int blockId)
{
    if (Assets::BlockManager::Get().isInstancedBlockType(blockId))
    {
        deleteInstancedBlock(pos, blockId);
    }
    else if (Assets::BlockManager::Get().isUninstancedBlockType(blockId))
    {
        deleteUninstancedBlock(pos, blockId);
    }
}

void VoxelEngine::addBlock(const Int3 &pos, int blockId)
{
    if (Assets::BlockManager::Get().isInstancedBlockType(blockId))
    {
        addInstancedBlock(pos, blockId);
    }
    else if (Assets::BlockManager::Get().isUninstancedBlockType(blockId))
    {
        addUninstancedBlock(pos, blockId);
    }
}

void VoxelEngine::deleteInstancedBlock(const Int3 &pos, int blockId)
{
    setVoxelAtGlobal(pos.x, pos.y, pos.z, 0);

    auto &scene = Scene::Get();
    std::unordered_map<int, std::set<int>> &geometryInstanceIdMap = scene.geometryInstanceIdMap;
    int objectId = Assets::BlockManager::Get().blockIdToObjectId(blockId);

    unsigned int instanceId = PositionToInstanceId(Assets::BlockManager::Get().getNumUninstancedBlockTypes(),
                                                   objectId, pos.x, pos.y, pos.z, chunkConfig.getGlobalWidth());
    geometryInstanceIdMap[objectId].erase(instanceId);

    updateSceneForInstancedBlock(objectId, instanceId);

    if (Assets::BlockManager::Get().isEmissive(blockId))
    {
        scene.m_lightsNeedUpdate = true;
    }

    // If we're deleting a light block, also remove its base
    if (Assets::BlockManager::Get().hasLightBase(blockId))
    {
        unsigned int baseBlockId = Assets::BlockManager::Get().getLightBaseBlockId(blockId);
        int baseObjectIdx = Assets::BlockManager::Get().blockIdToObjectId(baseBlockId);

        unsigned int baseInstanceId = PositionToInstanceId(Assets::BlockManager::Get().getNumUninstancedBlockTypes(),
                                                           baseObjectIdx, pos.x, pos.y, pos.z, chunkConfig.getGlobalWidth());
        geometryInstanceIdMap[baseObjectIdx].erase(baseInstanceId);

        updateSceneForInstancedBlock(baseObjectIdx, baseInstanceId);
    }
}

void VoxelEngine::deleteUninstancedBlock(const Int3 &pos, int blockId)
{
    auto &scene = Scene::Get();
    unsigned int newVal = 0;
    int objectId = Assets::BlockManager::Get().blockIdToObjectId(blockId);

    // Determine which chunk this voxel belongs to
    unsigned int chunkX, chunkY, chunkZ, localX, localY, localZ;
    globalToChunkCoords(pos.x, pos.y, pos.z, chunkX, chunkY, chunkZ, localX, localY, localZ);
    unsigned int chunkIndex = getChunkIndex(chunkX, chunkY, chunkZ);

    updateSingleVoxelGlobal(
        pos.x, pos.y, pos.z,
        newVal,
        voxelChunks,
        chunkConfig,
        *scene.getChunkGeometryAttributes(chunkIndex, objectId),
        *scene.getChunkGeometryIndices(chunkIndex, objectId),
        faceLocation[chunkIndex][objectId],
        scene.getChunkGeometryAttributeSize(chunkIndex, objectId),
        scene.getChunkGeometryIndicesSize(chunkIndex, objectId),
        currentFaceCount[chunkIndex][objectId],
        maxFaceCount[chunkIndex][objectId],
        freeFaces[chunkIndex][objectId]);

    updateSceneForUninstancedBlock(objectId, chunkIndex);
}

void VoxelEngine::addInstancedBlock(const Int3 &pos, int blockId)
{
    // Adding instanced block to scene

    setVoxelAtGlobal(pos.x, pos.y, pos.z, blockId);

    auto &scene = Scene::Get();
    std::unordered_map<int, std::set<int>> &geometryInstanceIdMap = scene.geometryInstanceIdMap;
    std::unordered_map<int, std::array<float, 12>> &instanceTransformMatrices = scene.instanceTransformMatrices;
    int objectId = Assets::BlockManager::Get().blockIdToObjectId(blockId);

    unsigned int instanceId = PositionToInstanceId(Assets::BlockManager::Get().getNumUninstancedBlockTypes(),
                                                   objectId, pos.x, pos.y, pos.z, chunkConfig.getGlobalWidth());
    geometryInstanceIdMap[objectId].insert(instanceId);

    std::array<float, 12> transform = {1.0f, 0.0f, 0.0f, (float)pos.x,
                                       0.0f, 1.0f, 0.0f, (float)pos.y,
                                       0.0f, 0.0f, 1.0f, (float)pos.z};
    instanceTransformMatrices[instanceId] = transform;

    updateSceneForInstancedBlock(objectId, instanceId);

    if (Assets::BlockManager::Get().isEmissive(blockId))
    {
        scene.m_lightsNeedUpdate = true;
    }

    // If this is a light block, also place its base
    if (Assets::BlockManager::Get().hasLightBase(blockId))
    {
        unsigned int baseBlockId = Assets::BlockManager::Get().getLightBaseBlockId(blockId);
        int baseObjectIdx = Assets::BlockManager::Get().blockIdToObjectId(baseBlockId);

        unsigned int baseInstanceId = PositionToInstanceId(Assets::BlockManager::Get().getNumUninstancedBlockTypes(),
                                                           baseObjectIdx, pos.x, pos.y, pos.z, chunkConfig.getGlobalWidth());
        geometryInstanceIdMap[baseObjectIdx].insert(baseInstanceId);

        instanceTransformMatrices[baseInstanceId] = transform;

        updateSceneForInstancedBlock(baseObjectIdx, baseInstanceId);
    }
}

void VoxelEngine::addUninstancedBlock(const Int3 &pos, int blockId)
{
    auto &scene = Scene::Get();
    unsigned int newVal = blockId;
    int objectId = Assets::BlockManager::Get().blockIdToObjectId(blockId);

    // Determine which chunk this voxel belongs to
    unsigned int chunkX, chunkY, chunkZ, localX, localY, localZ;
    globalToChunkCoords(pos.x, pos.y, pos.z, chunkX, chunkY, chunkZ, localX, localY, localZ);
    unsigned int chunkIndex = getChunkIndex(chunkX, chunkY, chunkZ);

    updateSingleVoxelGlobal(
        pos.x, pos.y, pos.z,
        newVal,
        voxelChunks,
        chunkConfig,
        *scene.getChunkGeometryAttributes(chunkIndex, objectId),
        *scene.getChunkGeometryIndices(chunkIndex, objectId),
        faceLocation[chunkIndex][objectId],
        scene.getChunkGeometryAttributeSize(chunkIndex, objectId),
        scene.getChunkGeometryIndicesSize(chunkIndex, objectId),
        currentFaceCount[chunkIndex][objectId],
        maxFaceCount[chunkIndex][objectId],
        freeFaces[chunkIndex][objectId]);

    updateSceneForUninstancedBlock(objectId, chunkIndex);
}

void VoxelEngine::updateSceneForInstancedBlock(int objectId, unsigned int instanceId)
{
    auto &scene = Scene::Get();
    scene.needSceneUpdate = true;
    scene.sceneUpdateObjectId.push_back(objectId);
    scene.sceneUpdateInstanceId.push_back(instanceId);
    scene.sceneUpdateChunkId.push_back(-1);
}

void VoxelEngine::updateSceneForUninstancedBlock(int objectId, unsigned int chunkIndex)
{
    auto &scene = Scene::Get();
    scene.needSceneUpdate = true;
    scene.sceneUpdateObjectId.push_back(objectId);
    scene.sceneUpdateInstanceId.push_back(-1);
    scene.sceneUpdateChunkId.push_back(chunkIndex);
}