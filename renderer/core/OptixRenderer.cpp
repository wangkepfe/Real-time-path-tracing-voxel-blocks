#include "core/Backend.h"
#include "core/OfflineBackend.h"
#include "util/DebugUtils.h"
#include "OptixRenderer.h"
#include "core/Scene.h"
#include "core/BufferManager.h"
#include "util/TextureUtils.h"
#include "core/GlobalSettings.h"
#include "sky/Sky.h"
#include "core/RenderCamera.h"
#ifndef OFFLINE_MODE
#include "core/InputHandler.h"
#endif

#include "util/BufferUtils.h"

// Helper function to get CUDA stream from either backend
CUstream getCurrentCudaStream()
{
    if (GlobalSettings::IsOfflineMode())
    {
        return OfflineBackend::Get().getCudaStream();
    }
    else
    {
        return Backend::Get().getCudaStream();
    }
}

#ifdef _WIN32
// The cfgmgr32 header is necessary for interrogating driver information in the registry.
#include <cfgmgr32.h>
// For convenience the library is also linked in automatically using the #pragma command.
#pragma comment(lib, "Cfgmgr32.lib")
#else
#include <dlfcn.h>
#endif

#include <mutex>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string.h>
#include <vector>
#include <filesystem>

#include "voxelengine/Block.h"
#include "voxelengine/VoxelEngine.h"

namespace
{

    std::string ReadPtx(std::string const &filename)
    {
        // std::filesystem::path cwd = std::filesystem::current_path();
        //  std::cout << "The current directory is " << cwd.string() << std::endl;

        std::ifstream inputPtx(filename);

        if (!inputPtx)
        {
            std::cerr << "ERROR: ReadPtx() Failed to open file " << filename << '\n';
            return std::string();
        }

        std::stringstream ptx;

        ptx << inputPtx.rdbuf();

        if (inputPtx.fail())
        {
            std::cerr << "ERROR: ReadPtx() Failed to read file " << filename << '\n';
            return std::string();
        }

        return ptx.str();
    }

} // namespace

static constexpr bool g_useGeometrySquareLight = false;

class OptixLogger
{
public:
    OptixLogger() : m_stream(std::cout) {}
    OptixLogger(std::ostream &s)
        : m_stream(s)
    {
    }

    static void callback(unsigned int level, const char *tag, const char *message, void *cbdata)
    {
        OptixLogger *self = static_cast<OptixLogger *>(cbdata);
        self->callback(level, tag, message);
    }

    // Need this detour because m_mutex is not static.
    void callback(unsigned int /*level*/, const char *tag, const char *message)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stream << tag << ":" << ((message) ? message : "(no message)") << "\n";
    }

private:
    std::mutex m_mutex;     // Mutex that protects m_stream.
    std::ostream &m_stream; // Needs m_mutex.
};

static OptixLogger g_logger = {};

void OptixRenderer::clear()
{
    auto &backend = Backend::Get();
    CUDA_CHECK(cudaStreamSynchronize(backend.getCudaStream()));

    CUDA_CHECK(cudaFree((void *)m_systemParameter.materialParameters));
    CUDA_CHECK(cudaFree((void *)m_d_systemParameter));

    for (size_t i = 0; i < m_geometries.size(); ++i)
    {
        CUDA_CHECK(cudaFree((void *)m_geometries[i].indices));
        CUDA_CHECK(cudaFree((void *)m_geometries[i].attributes));
        CUDA_CHECK(cudaFree((void *)m_geometries[i].gas));
    }
    CUDA_CHECK(cudaFree((void *)m_d_ias[0]));
    CUDA_CHECK(cudaFree((void *)m_d_ias[1]));

    CUDA_CHECK(cudaFree((void *)m_d_sbtRecordRaygeneration));
    CUDA_CHECK(cudaFree((void *)m_d_sbtRecordMiss));
    CUDA_CHECK(cudaFree((void *)m_d_sbtRecordCallables));

    if (m_d_sbtRecordGeometryInstanceData) {
        CUDA_CHECK(cudaFree((void *)m_d_sbtRecordGeometryInstanceData));
    }

    OPTIX_CHECK(m_api.optixPipelineDestroy(m_pipeline));
    OPTIX_CHECK(m_api.optixDeviceContextDestroy(m_context));
}

void OptixRenderer::render()
{
    auto &backend = Backend::Get();
    auto &scene = Scene::Get();
    auto &bufferManager = BufferManager::Get();

    int &iterationIndex = GlobalSettings::Get().iterationIndex;
    m_systemParameter.iterationIndex = iterationIndex++;

    CUDA_CHECK(cudaStreamSynchronize(backend.getCudaStream()));

    // CAMERA HISTORY UPDATE: Store PREVIOUS frame camera state before current frame processing
    // This should happen BEFORE InputHandler updates the camera for the current frame
    // The history camera should contain the camera state from the PREVIOUS frame
    // TODO: Move this to happen before InputHandler::update() is called
    
    // UNIFIED CAMERA MANAGEMENT: Camera is updated ONLY by InputHandler
    // OptixRenderer no longer calls camera.update() - this prevents position corruption
    // The camera matrices are already updated by the active CameraController
#ifndef OFFLINE_MODE
    // Online mode: Camera is managed by InputHandler's CameraControllers
    // No need to call camera.update() here - it's handled in InputHandler::update()
#else
    // Offline mode: For now, still use traditional camera update
    // TODO: Integrate unified camera management with offline mode
    RenderCamera::Get().camera.update();
#endif

    // SYSTEM PARAMETERS: Read-only access to cameras for shader/denoiser
    // Camera values are READ here, never modified
    const Camera& currentCameraForShader = RenderCamera::Get().camera;
    const Camera& historyCameraForShader = RenderCamera::Get().historyCamera;
    
    m_systemParameter.camera = currentCameraForShader;
    m_systemParameter.prevCamera = historyCameraForShader;
    
    m_systemParameter.timeInSecond = backend.getTimer().getTimeInSecond();

    const auto &skyModel = SkyModel::Get();
    m_systemParameter.sunDir = skyModel.getSunDir();
    m_systemParameter.accumulatedSkyLuminance = skyModel.getAccumulatedSkyLuminance();
    m_systemParameter.accumulatedSunLuminance = skyModel.getAccumulatedSunLuminance();
    m_systemParameter.accumulatedLocalLightLuminance = scene.accumulatedLocalLightLuminance;

    // Handle light-related parameters safely to prevent null pointer access
    m_systemParameter.lights = scene.m_lights;
    m_systemParameter.numLights = scene.m_numLights;
    m_systemParameter.lightAliasTable = scene.d_lightAliasTable;
    m_systemParameter.instanceLightMapping = scene.d_instanceLightMapping;
    m_systemParameter.numInstancedLightMesh = scene.numInstancedLightMesh;

    // Update animated entities and rebuild their BLAS if needed
    updateAnimatedEntities(backend.getCudaStream(), m_systemParameter.timeInSecond);

    BufferSetFloat4(bufferManager.GetBufferDim(UIBuffer), bufferManager.GetBuffer2D(UIBuffer), Float4(0.0f));

    CUDA_CHECK(cudaMemcpyAsync((void *)m_d_systemParameter, &m_systemParameter, sizeof(SystemParameter), cudaMemcpyHostToDevice, backend.getCudaStream()));

    // Debug: Check for errors before OptixLaunch
    CUDA_CHECK(cudaPeekAtLastError());

    // Only launch OptixLaunch if we have a valid top object (scene is not empty)
    if (m_systemParameter.topObject != 0) {
        OPTIX_CHECK(m_api.optixLaunch(m_pipeline, backend.getCudaStream(), (CUdeviceptr)m_d_systemParameter, sizeof(SystemParameter), &m_sbt, m_width, m_height, 1));
        
        // Check immediately after OptixLaunch with device sync
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            std::cout << "ERROR: OptixLaunch kernels failed with CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(backend.getCudaStream()));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

    BufferCopyFloat4(bufferManager.GetBufferDim(GeoNormalThinfilmBuffer), bufferManager.GetBuffer2D(GeoNormalThinfilmBuffer), bufferManager.GetBuffer2D(PrevGeoNormalThinfilmBuffer));
    BufferCopyFloat4(bufferManager.GetBufferDim(AlbedoBuffer), bufferManager.GetBuffer2D(AlbedoBuffer), bufferManager.GetBuffer2D(PrevAlbedoBuffer));
    BufferCopyFloat4(bufferManager.GetBufferDim(MaterialParameterBuffer), bufferManager.GetBuffer2D(MaterialParameterBuffer), bufferManager.GetBuffer2D(PrevMaterialParameterBuffer));
}

void OptixRenderer::updateAnimatedEntities(CUstream cudaStream, float currentTime)
{
    static int optixUpdateCount = 0;
    optixUpdateCount++;

    auto &scene = Scene::Get();
    constexpr int numTypesOfRays = 2;
    bool needBVHRebuild = false;

    // Calculate delta time for animation updates
    static float lastTime = -1.0f;
    float deltaTime;

    if (GlobalSettings::IsOfflineMode())
    {
        // In offline mode, use fixed timestep for consistent animation
        static int frameCounter = 0;
        frameCounter++;

        const float targetFPS = 30.0f; // 30 FPS for smooth animation
        deltaTime = 1.0f / targetFPS;

        // Fixed timestep for consistent offline animation
    }
    else
    {
        // Real-time mode - use actual time differences
        if (lastTime < 0.0f)
        {
            deltaTime = 1.0f / 60.0f; // Default for first frame
        }
        else
        {
            deltaTime = currentTime - lastTime;
        }
        lastTime = currentTime;
    }

    // Animation update system for OptiX renderer

    // Update each animated entity
    for (size_t entityIndex = 0; entityIndex < scene.getEntityCount(); ++entityIndex)
    {
        Entity *entity = scene.getEntity(entityIndex);
        if (entity && entity->getAttributeSize() > 0 && entity->getIndicesSize() > 0)
        {
            // Processing animated entity

            // Update entity animation
            entity->update(deltaTime);

            // Check if this entity has animations that require geometry updates
            if (entity->hasAnimation())
            {
                // SPARSE MAPPING: Look up entity geometry in sparse mapping
                auto entityGeometryIt = m_entityGeometryMap.find(entityIndex);
                if (entityGeometryIt == m_entityGeometryMap.end()) {
                    printf("ENTITY UPDATE ERROR: Entity geometry not found in sparse mapping (entityIndex=%zu)\n", entityIndex);
                    continue;
                }
                
                size_t geometryIndex = entityGeometryIt->second;
                if (geometryIndex < m_geometries.size())
                {
                    GeometryData &geometry = m_geometries[geometryIndex];

                    // Validate entity data before BLAS update
                    if (!entity->getAttributes() || !entity->getIndices() || entity->getAttributeSize() == 0 || geometry.gas == 0)
                    {
                        continue;
                    }

                    // Update the BLAS with new animated vertices (much faster than recreation)
                    OptixTraversableHandle blasHandle = Scene::UpdateGeometry(
                        m_api, m_context, cudaStream,
                        geometry,
                        entity->getAttributes(),
                        entity->getIndices(),
                        entity->getAttributeSize(),
                        entity->getIndicesSize());

                    // Update the instance with the new BLAS handle
                    unsigned int targetInstanceId = EntityConstants::ENTITY_INSTANCE_ID_OFFSET + static_cast<unsigned int>(entityIndex);
                    for (auto &instance : m_instances)
                    {
                        if (instance.instanceId == targetInstanceId)
                        {
                            instance.traversableHandle = blasHandle;

                            // Update transform matrix in case entity moved
                            float transformMatrix[12];
                            entity->getTransform().getTransformMatrix(transformMatrix);
                            memcpy(instance.transform, transformMatrix, sizeof(float) * 12);

                            needBVHRebuild = true;
                            break;
                        }
                    }

                    // Update SBT records for the animated entity
                    // Calculate SBT offset for this entity
                    unsigned int entitySbtOffset = (scene.numChunks * scene.uninstancedGeometryCount + scene.instancedGeometryCount + static_cast<unsigned int>(entityIndex)) * numTypesOfRays;
                    if (entitySbtOffset + numTypesOfRays - 1 < m_sbtRecordGeometryInstanceData.size())
                    {
                        for (unsigned int rayType = 0; rayType < numTypesOfRays; ++rayType)
                        {
                            unsigned int sbtIndex = entitySbtOffset + rayType;
                            m_sbtRecordGeometryInstanceData[sbtIndex].data.indices = (Int3 *)geometry.indices;
                            m_sbtRecordGeometryInstanceData[sbtIndex].data.attributes = (VertexAttributes *)geometry.attributes;
                            
                        }
                    }
                }
            }
        }
    }

    // Rebuild top-level BVH if any animated entities were updated
    if (needBVHRebuild)
    {
        CUDA_CHECK(cudaStreamSynchronize(cudaStream));

        m_systemParameter.prevTopObject = m_systemParameter.topObject;

        int nextIndex = (m_currentIasIdx + 1) % 2;
        if (m_d_ias[nextIndex] != 0)
        {
            CUDA_CHECK(cudaFree((void *)m_d_ias[nextIndex]));
        }

        // Handle case where there are no instances to prevent CUDA errors
        if (m_instances.empty()) {
            // Set to null handle for empty scene
            m_systemParameter.topObject = 0;
            // Swap the buffer index even for empty scenes
            m_currentIasIdx = nextIndex;
        }
        else {
            // Build the new BVH in the inactive slot
            CUdeviceptr d_instances;
            const size_t instancesSizeInBytes = sizeof(OptixInstance) * m_instances.size();
            CUDA_CHECK(cudaMalloc((void **)&d_instances, instancesSizeInBytes));
            CUDA_CHECK(cudaMemcpyAsync((void *)d_instances, m_instances.data(), instancesSizeInBytes,
                                       cudaMemcpyHostToDevice, cudaStream));

            OptixBuildInput instanceInput = {};
            instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            instanceInput.instanceArray.instances = d_instances;
            instanceInput.instanceArray.numInstances = (unsigned int)m_instances.size();

            OptixAccelBuildOptions accelBuildOptions = {};
            accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
            accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

            OptixAccelBufferSizes iasBufferSizes = {};
            OPTIX_CHECK(m_api.optixAccelComputeMemoryUsage(m_context, &accelBuildOptions, &instanceInput, 1, &iasBufferSizes));

            CUDA_CHECK(cudaMalloc((void **)&m_d_ias[nextIndex], iasBufferSizes.outputSizeInBytes));

            CUdeviceptr d_tmp;
            CUDA_CHECK(cudaMalloc((void **)&d_tmp, iasBufferSizes.tempSizeInBytes));

            // Build the acceleration structure into the new buffer
            OPTIX_CHECK(m_api.optixAccelBuild(m_context, cudaStream,
                                              &accelBuildOptions, &instanceInput, 1,
                                              d_tmp, iasBufferSizes.tempSizeInBytes,
                                              m_d_ias[nextIndex], iasBufferSizes.outputSizeInBytes,
                                              &m_systemParameter.topObject, nullptr, 0));
            

            // Clean up temporary allocations
            CUDA_CHECK(cudaFree((void *)d_tmp));
            CUDA_CHECK(cudaFree((void *)d_instances));

            // Swap the buffer index
            m_currentIasIdx = nextIndex;
        }

        // Upload updated SBT records only if we have instances
        if (m_d_sbtRecordGeometryInstanceData && !m_sbtRecordGeometryInstanceData.empty()) {
            CUDA_CHECK(cudaMemcpyAsync((void *)m_d_sbtRecordGeometryInstanceData, m_sbtRecordGeometryInstanceData.data(),
                                       sizeof(SbtRecordGeometryInstanceData) * m_sbtRecordGeometryInstanceData.size(),
                                       cudaMemcpyHostToDevice, cudaStream));
        }
    }
}

#ifdef _WIN32
// Code based on helper function in optix_stubs.h
static void *optixLoadWindowsDll(void)
{
    const char *optixDllName = "nvoptix.dll";
    void *handle = NULL;

    // Get the size of the path first, then allocate
    unsigned int size = GetSystemDirectoryA(NULL, 0);
    if (size == 0)
    {
        // Couldn't get the system path size, so bail
        return NULL;
    }

    size_t pathSize = size + 1 + strlen(optixDllName);
    char *systemPath = (char *)malloc(pathSize);

    if (GetSystemDirectoryA(systemPath, size) != size - 1)
    {
        // Something went wrong
        free(systemPath);
        return NULL;
    }

    strcat(systemPath, "\\");
    strcat(systemPath, optixDllName);

    handle = LoadLibraryA(systemPath);

    free(systemPath);

    if (handle)
    {
        return handle;
    }

    // If we didn't find it, go looking in the register store.  Since nvoptix.dll doesn't
    // have its own registry entry, we are going to look for the OpenGL driver which lives
    // next to nvoptix.dll. 0 (null) will be returned if any errors occured.

    static const char *deviceInstanceIdentifiersGUID = "{4d36e968-e325-11ce-bfc1-08002be10318}";
    const ULONG flags = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT;
    ULONG deviceListSize = 0;

    if (CM_Get_Device_ID_List_SizeA(&deviceListSize, deviceInstanceIdentifiersGUID, flags) != CR_SUCCESS)
    {
        return NULL;
    }

    char *deviceNames = (char *)malloc(deviceListSize);

    if (CM_Get_Device_ID_ListA(deviceInstanceIdentifiersGUID, deviceNames, deviceListSize, flags))
    {
        free(deviceNames);
        return NULL;
    }

    DEVINST devID = 0;

    // Continue to the next device if errors are encountered.
    for (char *deviceName = deviceNames; *deviceName; deviceName += strlen(deviceName) + 1)
    {
        if (CM_Locate_DevNodeA(&devID, deviceName, CM_LOCATE_DEVNODE_NORMAL) != CR_SUCCESS)
        {
            continue;
        }

        HKEY regKey = 0;
        if (CM_Open_DevNode_Key(devID, KEY_QUERY_VALUE, 0, RegDisposition_OpenExisting, &regKey, CM_REGISTRY_SOFTWARE) != CR_SUCCESS)
        {
            continue;
        }

        const char *valueName = "OpenGLDriverName";
        DWORD valueSize = 0;

        LSTATUS ret = RegQueryValueExA(regKey, valueName, NULL, NULL, NULL, &valueSize);
        if (ret != ERROR_SUCCESS)
        {
            RegCloseKey(regKey);
            continue;
        }

        char *regValue = (char *)malloc(valueSize);
        ret = RegQueryValueExA(regKey, valueName, NULL, NULL, (LPBYTE)regValue, &valueSize);
        if (ret != ERROR_SUCCESS)
        {
            free(regValue);
            RegCloseKey(regKey);
            continue;
        }

        // Strip the OpenGL driver dll name from the string then create a new string with
        // the path and the nvoptix.dll name
        for (int i = valueSize - 1; i >= 0 && regValue[i] != '\\'; --i)
        {
            regValue[i] = '\0';
        }

        size_t newPathSize = strlen(regValue) + strlen(optixDllName) + 1;
        char *dllPath = (char *)malloc(newPathSize);
        strcpy(dllPath, regValue);
        strcat(dllPath, optixDllName);

        free(regValue);
        RegCloseKey(regKey);

        handle = LoadLibraryA((LPCSTR)dllPath);
        free(dllPath);

        if (handle)
        {
            break;
        }
    }

    free(deviceNames);

    return handle;
}
#endif

void OptixRenderer::update()
{
    auto &scene = Scene::Get();

    constexpr int numTypesOfRays = 2;
    const int numObjects = scene.uninstancedGeometryCount + scene.instancedGeometryCount;

    if (!scene.needSceneUpdate)
    {
        return;
    }

    if (scene.needSceneReloadUpdate)
    {
        m_instances.resize(scene.uninstancedGeometryCount);
        instanceIds.clear();

        // Uninstanced - recreate geometry for all chunks
        m_instances.clear(); // Clear all instances and rebuild them
        unsigned int instanceIndex = 0;
        unsigned int baseSbtRecords = 0; // Track SBT records for entity offset calculation

        for (unsigned int chunkIndex = 0; chunkIndex < scene.numChunks; ++chunkIndex)
        {
            for (unsigned int objectId = GetUninstancedObjectIdBegin(); objectId < GetUninstancedObjectIdEnd(); ++objectId)
            {
                unsigned int blockId = ObjectIdToBlockId(objectId);
                unsigned int geometryIndex = chunkIndex * scene.uninstancedGeometryCount + objectId;

                assert(geometryIndex < m_geometries.size());
                if (scene.getChunkGeometryAttributeSize(chunkIndex, objectId) > 0 &&
                    scene.getChunkGeometryIndicesSize(chunkIndex, objectId) > 0)
                {
                    GeometryData &geometry = m_geometries[geometryIndex];
                    CUDA_CHECK(cudaFree((void *)geometry.gas));

                    OptixTraversableHandle blasHandle = Scene::CreateGeometry(
                        m_api, m_context, Backend::Get().getCudaStream(),
                        geometry,
                        *scene.getChunkGeometryAttributes(chunkIndex, objectId),
                        *scene.getChunkGeometryIndices(chunkIndex, objectId),
                        scene.getChunkGeometryAttributeSize(chunkIndex, objectId),
                        scene.getChunkGeometryIndicesSize(chunkIndex, objectId));

                    // Calculate chunk offset
                    auto &voxelEngine = VoxelEngine::Get();
                    auto &chunkConfig = voxelEngine.chunkConfig;
                    unsigned int chunkX = chunkIndex % chunkConfig.chunksX;
                    unsigned int chunkZ = (chunkIndex / chunkConfig.chunksX) % chunkConfig.chunksZ;
                    unsigned int chunkY = chunkIndex / (chunkConfig.chunksX * chunkConfig.chunksZ);

                    OptixInstance instance = {};
                    const float transformMatrix[12] =
                        {
                            1.0f, 0.0f, 0.0f, (float)(chunkX * VoxelChunk::width),
                            0.0f, 1.0f, 0.0f, (float)(chunkY * VoxelChunk::width),
                            0.0f, 0.0f, 1.0f, (float)(chunkZ * VoxelChunk::width)};
                    memcpy(instance.transform, transformMatrix, sizeof(float) * 12);
                    instance.instanceId = geometryIndex;
                    instance.visibilityMask = IsTransparentBlockType(blockId) ? 1 : 255;
                    instance.sbtOffset = objectId * numTypesOfRays;
                    instance.flags = OPTIX_INSTANCE_FLAG_NONE;
                    instance.traversableHandle = blasHandle;
                    m_instances.push_back(instance);

                    // Count this geometry for entity SBT offset calculation
                    baseSbtRecords++;

                    // Shader binding table record hit group geometry
                    unsigned int sbtIndex = objectId * numTypesOfRays;
                    assert(sbtIndex + numTypesOfRays - 1 < m_sbtRecordGeometryInstanceData.size());

                    for (unsigned int rayType = 0; rayType < numTypesOfRays; ++rayType)
                    {
                        m_sbtRecordGeometryInstanceData[sbtIndex + rayType].data.indices = (Int3 *)geometry.indices;
                        m_sbtRecordGeometryInstanceData[sbtIndex + rayType].data.attributes = (VertexAttributes *)geometry.attributes;
                        
                    }
                }
            }
        }
        // Upload SBT only if we have instances
        if (m_d_sbtRecordGeometryInstanceData && !m_sbtRecordGeometryInstanceData.empty()) {
            CUDA_CHECK(cudaMemcpyAsync((void *)m_d_sbtRecordGeometryInstanceData, m_sbtRecordGeometryInstanceData.data(), sizeof(SbtRecordGeometryInstanceData) * m_sbtRecordGeometryInstanceData.size(), cudaMemcpyHostToDevice, Backend::Get().getCudaStream()));
        }

        // Instanced
        for (unsigned int objectId = GetInstancedObjectIdBegin(); objectId < GetInstancedObjectIdEnd(); ++objectId)
        {
            // Only create instances if BLAS handle is valid
            if (objectIdxToBlasHandleMap[objectId] != 0) {
                for (int instanceId : scene.geometryInstanceIdMap[objectId])
                {
                    OptixInstance instance = {};
                    memcpy(instance.transform, scene.instanceTransformMatrices[instanceId].data(), sizeof(float) * 12);
                    instance.instanceId = instanceId;
                    instance.visibilityMask = 255;
                    instance.sbtOffset = objectId * numTypesOfRays;
                    instance.flags = OPTIX_INSTANCE_FLAG_NONE;
                    instance.traversableHandle = objectIdxToBlasHandleMap[objectId];
                    m_instances.push_back(instance);
                    instanceIds.insert(instanceId);
                }
            }
        }

        // Add instanced geometry records to the base count
        baseSbtRecords += scene.instancedGeometryCount;

        // Entities - add them to instances during reload
        // NOTE: Entities should keep their original SBT offsets from init() since SBT records are static
        for (size_t entityIndex = 0; entityIndex < scene.getEntityCount(); ++entityIndex)
        {
            Entity *entity = scene.getEntity(entityIndex);
            if (entity && entity->getAttributeSize() > 0 && entity->getIndicesSize() > 0)
            {
                // Calculate the correct geometry index for entities
                unsigned int geometryIndex = scene.numChunks * scene.uninstancedGeometryCount + scene.instancedGeometryCount + static_cast<unsigned int>(entityIndex);

                OptixInstance instance = {};
                float transformMatrix[12];
                entity->getTransform().getTransformMatrix(transformMatrix);
                memcpy(instance.transform, transformMatrix, sizeof(float) * 12);

                // Use consistent entity instance ID
                instance.instanceId = EntityConstants::ENTITY_INSTANCE_ID_OFFSET + static_cast<unsigned int>(entityIndex);
                instance.visibilityMask = 255;

                // Calculate SBT offset using cached base + entity-specific offset
                unsigned int entitySbtOffset = baseSbtRecords + static_cast<unsigned int>(entityIndex);
                instance.sbtOffset = entitySbtOffset * numTypesOfRays;
                instance.flags = OPTIX_INSTANCE_FLAG_NONE;
                instance.traversableHandle = m_geometries[geometryIndex].gas; // Use the gas handle directly since we don't have a map for entities
                m_instances.push_back(instance);
            }
        }
    }
    else
    {
        for (int i = 0; i < scene.sceneUpdateObjectId.size(); ++i)
        {
            auto objectId = scene.sceneUpdateObjectId[i];
            unsigned int blockId = ObjectIdToBlockId(objectId);

            // Uninstanced
            if (IsUninstancedBlockType(blockId))
            {
                // SPARSE GEOMETRY UPDATE: Use sparse mapping system for dynamic updates
                unsigned int chunkIndex = scene.sceneUpdateChunkId[i];
                ChunkObjectKey chunkKey = {chunkIndex, objectId};
                
                
                // Check if this geometry has non-zero size (valid for update/creation)
                if (scene.getChunkGeometryAttributeSize(chunkIndex, objectId) > 0 &&
                    scene.getChunkGeometryIndicesSize(chunkIndex, objectId) > 0)
                {
                    // Check if geometry already exists in sparse mapping
                    auto chunkGeometryIt = m_chunkGeometryMap.find(chunkKey);
                    
                    if (chunkGeometryIt != m_chunkGeometryMap.end())
                    {
                        // CASE 1: Update existing geometry
                        size_t geometryIndex = chunkGeometryIt->second;
                        if (geometryIndex < m_geometries.size()) {
                            printf("SPARSE UPDATE: Updating existing chunk geometry (chunk=%u, object=%u, mapped_index=%zu)\n", 
                                   chunkIndex, objectId, geometryIndex);
                            
                            GeometryData &geometry = m_geometries[geometryIndex];
                            CUDA_CHECK(cudaFree((void *)geometry.gas));

                            OptixTraversableHandle blasHandle = Scene::CreateGeometry(
                                m_api, m_context, Backend::Get().getCudaStream(),
                                geometry,
                                *scene.getChunkGeometryAttributes(chunkIndex, objectId),
                                *scene.getChunkGeometryIndices(chunkIndex, objectId),
                                scene.getChunkGeometryAttributeSize(chunkIndex, objectId),
                                scene.getChunkGeometryIndicesSize(chunkIndex, objectId));

                            // Find and update the corresponding instance using the OLD instanceId system
                            unsigned int expectedInstanceId = chunkIndex * scene.uninstancedGeometryCount + objectId;
                            for (auto &instance : m_instances)
                            {
                                if (instance.instanceId == expectedInstanceId)
                                {
                                    instance.traversableHandle = blasHandle;
                                    printf("SPARSE UPDATE: Updated instance traversable handle (instanceId=%u)\n", expectedInstanceId);
                                    printf("SPARSE UPDATE: Instance transform: [%.1f %.1f %.1f %.1f] [%.1f %.1f %.1f %.1f] [%.1f %.1f %.1f %.1f]\n",
                                           instance.transform[0], instance.transform[1], instance.transform[2], instance.transform[3],
                                           instance.transform[4], instance.transform[5], instance.transform[6], instance.transform[7], 
                                           instance.transform[8], instance.transform[9], instance.transform[10], instance.transform[11]);
                                    printf("SPARSE UPDATE: Instance visibility mask: %u, flags: %u\n", instance.visibilityMask, instance.flags);
                                    
                                    // Update SBT records using the instance's sbtOffset
                                    unsigned int sbtIndex = instance.sbtOffset;
                                    if (sbtIndex + numTypesOfRays - 1 < m_sbtRecordGeometryInstanceData.size()) {
                                        // MATERIAL FIX: Get correct material for this block type
                                        unsigned int blockType = ObjectIdToBlockId(objectId);
                                        unsigned int materialIndex = MaterialManager::Get().getMaterialIndexForBlock(blockType);
                                        
                                        for (unsigned int rayType = 0; rayType < numTypesOfRays; ++rayType)
                                        {
                                            m_sbtRecordGeometryInstanceData[sbtIndex + rayType].data.indices = (Int3 *)geometry.indices;
                                            m_sbtRecordGeometryInstanceData[sbtIndex + rayType].data.attributes = (VertexAttributes *)geometry.attributes;
                                            m_sbtRecordGeometryInstanceData[sbtIndex + rayType].data.materialIndex = materialIndex;
                                        }
                                        printf("SPARSE UPDATE: Updated SBT records (sbtIndex=%u, materialIndex=%u)\n", sbtIndex, materialIndex);
                                        printf("SPARSE UPDATE: SBT geometry pointers: d_indices=%p, d_attributes=%p\n", 
                                               m_sbtRecordGeometryInstanceData[sbtIndex].data.indices,
                                               m_sbtRecordGeometryInstanceData[sbtIndex].data.attributes);
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    else
                    {
                        // CASE 2: Create new geometry (this is a newly placed block)
                        printf("SPARSE UPDATE: Creating new chunk geometry (chunk=%u, object=%u)\n", chunkIndex, objectId);
                        
                        GeometryData geometry = {};
                        OptixTraversableHandle blasHandle = Scene::CreateGeometry(
                            m_api, m_context, Backend::Get().getCudaStream(),
                            geometry,
                            *scene.getChunkGeometryAttributes(chunkIndex, objectId),
                            *scene.getChunkGeometryIndices(chunkIndex, objectId),
                            scene.getChunkGeometryAttributeSize(chunkIndex, objectId),
                            scene.getChunkGeometryIndicesSize(chunkIndex, objectId));
                        
                        // Register in sparse mapping
                        size_t geometryIndex = m_geometries.size();
                        m_geometries.push_back(geometry);
                        m_chunkGeometryMap[chunkKey] = geometryIndex;
                        printf("SPARSE UPDATE: Registered new chunk geometry (chunk=%u, object=%u) -> index=%zu\n", 
                               chunkIndex, objectId, geometryIndex);
                        
                        // Create new instance for the new geometry
                        auto &voxelEngine = VoxelEngine::Get();
                        auto &chunkConfig = voxelEngine.chunkConfig;
                        unsigned int chunkX = chunkIndex % chunkConfig.chunksX;
                        unsigned int chunkZ = (chunkIndex / chunkConfig.chunksX) % chunkConfig.chunksZ;
                        unsigned int chunkY = chunkIndex / (chunkConfig.chunksX * chunkConfig.chunksZ);
                        
                        OptixInstance instance = {};
                        const float transformMatrix[12] =
                            {
                                1.0f, 0.0f, 0.0f, (float)(chunkX * VoxelChunk::width),
                                0.0f, 1.0f, 0.0f, (float)(chunkY * VoxelChunk::width),
                                0.0f, 0.0f, 1.0f, (float)(chunkZ * VoxelChunk::width)};
                        memcpy(instance.transform, transformMatrix, sizeof(float) * 12);
                        instance.instanceId = chunkIndex * scene.uninstancedGeometryCount + objectId;
                        instance.visibilityMask = IsTransparentBlockType(blockId) ? 1 : 255;
                        instance.sbtOffset = m_sbtRecordGeometryInstanceData.size(); // Use next available SBT slot
                        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
                        instance.traversableHandle = blasHandle;
                        m_instances.push_back(instance);
                        
                        printf("SPARSE UPDATE: Created new instance (instanceId=%u, sbtOffset=%u)\n", 
                               instance.instanceId, instance.sbtOffset);
                        
                        // Create new SBT records for the new geometry
                        // MATERIAL FIX: Get correct material for this block type (outside the loop)
                        unsigned int blockType = ObjectIdToBlockId(objectId);
                        unsigned int materialIndex = MaterialManager::Get().getMaterialIndexForBlock(blockType);
                        
                        for (unsigned int rayType = 0; rayType < numTypesOfRays; ++rayType)
                        {
                            SbtRecordGeometryInstanceData sbtRecord;
                            if (rayType == 0)
                            {
                                memcpy(sbtRecord.header, m_sbtRecordHitRadiance.header, OPTIX_SBT_RECORD_HEADER_SIZE);
                            }
                            else
                            {
                                memcpy(sbtRecord.header, m_sbtRecordHitShadow.header, OPTIX_SBT_RECORD_HEADER_SIZE);
                            }
                            
                            sbtRecord.data.indices = (Int3 *)geometry.indices;
                            sbtRecord.data.attributes = (VertexAttributes *)geometry.attributes;
                            sbtRecord.data.materialIndex = materialIndex;
                            
                            m_sbtRecordGeometryInstanceData.push_back(sbtRecord);
                        }
                        printf("SPARSE UPDATE: Added SBT records for new geometry (materialIndex=%u)\n", materialIndex);
                    }
                }
                else
                {
                    // CASE 3: Remove geometry (block was deleted - size is now 0)
                    auto chunkGeometryIt = m_chunkGeometryMap.find(chunkKey);
                    if (chunkGeometryIt != m_chunkGeometryMap.end())
                    {
                        printf("SPARSE UPDATE: Removing chunk geometry (chunk=%u, object=%u)\n", chunkIndex, objectId);
                        
                        // Find and remove the corresponding instance
                        unsigned int expectedInstanceId = chunkIndex * scene.uninstancedGeometryCount + objectId;
                        auto instanceIt = std::find_if(m_instances.begin(), m_instances.end(), 
                            [expectedInstanceId](const OptixInstance& inst) { 
                                return inst.instanceId == expectedInstanceId; 
                            });
                        
                        if (instanceIt != m_instances.end()) {
                            printf("SPARSE UPDATE: Removed instance (instanceId=%u)\n", expectedInstanceId);
                            m_instances.erase(instanceIt);
                        }
                        
                        // Note: We don't remove from m_geometries or m_chunkGeometryMap to avoid invalidating indices
                        // Instead, we just mark the geometry as invalid by setting gas = 0
                        size_t geometryIndex = chunkGeometryIt->second;
                        if (geometryIndex < m_geometries.size()) {
                            GeometryData &geometry = m_geometries[geometryIndex];
                            if (geometry.gas != 0) {
                                CUDA_CHECK(cudaFree((void *)geometry.gas));
                                geometry.gas = 0;
                                geometry.indices = nullptr;
                                geometry.attributes = nullptr;
                                geometry.numIndices = 0;
                                geometry.numAttributes = 0;
                                printf("SPARSE UPDATE: Invalidated geometry at index %zu\n", geometryIndex);
                            }
                        }
                        
                        // Remove from mapping
                        m_chunkGeometryMap.erase(chunkGeometryIt);
                    }
                }
            }
            // Instanced
            else if (IsInstancedBlockType(blockId))
            {
                auto instanceId = scene.sceneUpdateInstanceId[i];
                bool hasInstance = instanceIds.count(instanceId);

                if (hasInstance)
                {
                    instanceIds.erase(instanceId);
                    int idxToRemove = -1;
                    for (int j = 0; j < m_instances.size(); ++j)
                    {
                        if (m_instances[j].instanceId == instanceId)
                        {
                            idxToRemove = j;
                        }
                    }
                    assert(idxToRemove != -1);
                    m_instances.erase(m_instances.begin() + idxToRemove);
                }
                else
                {
                    // Only create instances if BLAS handle is valid
                    if (objectIdxToBlasHandleMap[objectId] != 0) {
                        instanceIds.insert(instanceId);

                        OptixInstance instance = {};
                        memcpy(instance.transform, scene.instanceTransformMatrices[instanceId].data(), sizeof(float) * 12);
                        instance.instanceId = instanceId;
                        instance.visibilityMask = 255;
                        instance.sbtOffset = objectId * numTypesOfRays;
                        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
                        instance.traversableHandle = objectIdxToBlasHandleMap[objectId];

                        m_instances.push_back(instance);
                    }
                }
            }
        }
        // Upload SBT only if we have instances
        if (m_d_sbtRecordGeometryInstanceData && !m_sbtRecordGeometryInstanceData.empty()) {
            CUDA_CHECK(cudaMemcpyAsync((void *)m_d_sbtRecordGeometryInstanceData, m_sbtRecordGeometryInstanceData.data(), sizeof(SbtRecordGeometryInstanceData) * m_sbtRecordGeometryInstanceData.size(), cudaMemcpyHostToDevice, Backend::Get().getCudaStream()));
        }
    }

    // Rebuild BVH
    {
        CUstream cudaStream = getCurrentCudaStream();
        CUDA_CHECK(cudaStreamSynchronize(cudaStream));

        m_systemParameter.prevTopObject = m_systemParameter.topObject;

        int nextIndex = (m_currentIasIdx + 1) % 2;
        if (m_d_ias[nextIndex] != 0)
        {
            CUDA_CHECK(cudaFree((void *)m_d_ias[nextIndex]));
            m_d_ias[nextIndex] = 0;
        }

        // Handle case where there are no instances to prevent CUDA errors
        if (m_instances.empty()) {
            // Set to null handle for empty scene
            m_systemParameter.topObject = 0;
            // Swap the buffer index even for empty scenes
            m_currentIasIdx = nextIndex;
        }
        else {
            // Build the new BVH in the inactive slot.
            CUdeviceptr d_instances;
            const size_t instancesSizeInBytes = sizeof(OptixInstance) * m_instances.size();
            CUDA_CHECK(cudaMalloc((void **)&d_instances, instancesSizeInBytes));
            CUDA_CHECK(cudaMemcpyAsync((void *)d_instances, m_instances.data(), instancesSizeInBytes,
                                       cudaMemcpyHostToDevice, cudaStream));

            OptixBuildInput instanceInput = {};
            instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            instanceInput.instanceArray.instances = d_instances;
            instanceInput.instanceArray.numInstances = (unsigned int)m_instances.size();

            OptixAccelBuildOptions accelBuildOptions = {};
            accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
            accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

            OptixAccelBufferSizes iasBufferSizes = {};
            OPTIX_CHECK(m_api.optixAccelComputeMemoryUsage(m_context, &accelBuildOptions, &instanceInput, 1, &iasBufferSizes));

            CUDA_CHECK(cudaMalloc((void **)&m_d_ias[nextIndex], iasBufferSizes.outputSizeInBytes));

            CUdeviceptr d_tmp;
            CUDA_CHECK(cudaMalloc((void **)&d_tmp, iasBufferSizes.tempSizeInBytes));

            // Build the acceleration structure into the new buffer.
            OPTIX_CHECK(m_api.optixAccelBuild(m_context, cudaStream,
                                              &accelBuildOptions, &instanceInput, 1,
                                              d_tmp, iasBufferSizes.tempSizeInBytes,
                                              m_d_ias[nextIndex], iasBufferSizes.outputSizeInBytes,
                                              &m_systemParameter.topObject, nullptr, 0));

            // Synchronize and clean up temporary allocations.
            CUDA_CHECK(cudaStreamSynchronize(cudaStream));
            CUDA_CHECK(cudaFree((void *)d_tmp));
            CUDA_CHECK(cudaFree((void *)d_instances));

            // Swap the buffer index.
            m_currentIasIdx = nextIndex;
        }
    }

    scene.sceneUpdateObjectId.clear();
    scene.sceneUpdateInstanceId.clear();
    scene.sceneUpdateChunkId.clear();

    scene.needSceneUpdate = false;
    scene.needSceneReloadUpdate = false;
}

void OptixRenderer::init()
{
    Scene &scene = Scene::Get();
    constexpr int numTypesOfRays = 2;
    const int numObjects = scene.uninstancedGeometryCount + scene.instancedGeometryCount;

    {
        m_systemParameter.topObject = 0;
        m_systemParameter.prevTopObject = 0;
        m_systemParameter.materialParameters = nullptr;

        const auto &bufferManager = BufferManager::Get();
        const auto &skyModel = SkyModel::Get();

        m_systemParameter.illuminationBuffer = bufferManager.GetBuffer2D(IlluminationBuffer);

        // Gbuffers
        m_systemParameter.normalRoughnessBuffer = bufferManager.GetBuffer2D(NormalRoughnessBuffer);
        m_systemParameter.depthBuffer = bufferManager.GetBuffer2D(DepthBuffer);
        m_systemParameter.albedoBuffer = bufferManager.GetBuffer2D(AlbedoBuffer);
        m_systemParameter.materialBuffer = bufferManager.GetBuffer2D(MaterialBuffer);
        m_systemParameter.geoNormalThinfilmBuffer = bufferManager.GetBuffer2D(GeoNormalThinfilmBuffer);
        m_systemParameter.materialParameterBuffer = bufferManager.GetBuffer2D(MaterialParameterBuffer);

        // Previous frame Gbuffers
        m_systemParameter.prevNormalRoughnessBuffer = bufferManager.GetBuffer2D(PrevNormalRoughnessBuffer);
        m_systemParameter.prevDepthBuffer = bufferManager.GetBuffer2D(PrevDepthBuffer);
        m_systemParameter.prevAlbedoBuffer = bufferManager.GetBuffer2D(PrevAlbedoBuffer);
        m_systemParameter.prevMaterialBuffer = bufferManager.GetBuffer2D(PrevMaterialBuffer);
        m_systemParameter.prevGeoNormalThinfilmBuffer = bufferManager.GetBuffer2D(PrevGeoNormalThinfilmBuffer);
        m_systemParameter.prevMaterialParameterBuffer = bufferManager.GetBuffer2D(PrevMaterialParameterBuffer);

        m_systemParameter.motionVectorBuffer = bufferManager.GetBuffer2D(MotionVectorBuffer);

        m_systemParameter.UIBuffer = bufferManager.GetBuffer2D(UIBuffer);

        m_systemParameter.randGen = d_randGen;

        m_systemParameter.skyBuffer = bufferManager.GetBuffer2D(SkyBuffer);
        m_systemParameter.sunBuffer = bufferManager.GetBuffer2D(SunBuffer);
        m_systemParameter.skyAliasTable = skyModel.getSkyAliasTable();
        m_systemParameter.sunAliasTable = skyModel.getSunAliasTable();
        m_systemParameter.accumulatedSkyLuminance = skyModel.getAccumulatedSkyLuminance();
        m_systemParameter.accumulatedSunLuminance = skyModel.getAccumulatedSunLuminance();
        m_systemParameter.skyRes = skyModel.getSkyRes();
        m_systemParameter.sunRes = skyModel.getSunRes();

        m_systemParameter.edgeToHighlight = Scene::Get().edgeToHighlight;

        m_systemParameter.iterationIndex = 0;

        m_systemParameter.reservoirBlockRowPitch = bufferManager.reservoirBlockRowPitch;
        m_systemParameter.reservoirArrayPitch = bufferManager.reservoirArrayPitch;
        m_systemParameter.reservoirBuffer = bufferManager.reservoirBuffer;

        m_systemParameter.neighborOffsetBuffer = bufferManager.neighborOffsetBuffer;

        m_d_ias[0] = 0;
        m_d_ias[1] = 0;

        m_pipeline = nullptr;

        m_d_systemParameter = nullptr;

        m_d_sbtRecordRaygeneration = 0;
        m_d_sbtRecordMiss = 0;

        m_d_sbtRecordCallables = 0;

        m_d_sbtRecordGeometryInstanceData = nullptr;

        auto &camera = RenderCamera::Get().camera;
        camera.init(m_width, m_height);
    }

    // Create function table
    {
        void *handle = optixLoadWindowsDll();
        if (!handle)
        {
            throw std::runtime_error("OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND");
        }
        void *symbol = reinterpret_cast<void *>(GetProcAddress((HMODULE)handle, "optixQueryFunctionTable"));
        if (!symbol)
        {
            throw std::runtime_error("OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND");
        }
        OptixQueryFunctionTable_t *optixQueryFunctionTable = reinterpret_cast<OptixQueryFunctionTable_t *>(symbol);

        OptixResult res = optixQueryFunctionTable(OPTIX_ABI_VERSION, 0, 0, 0, &m_api, sizeof(OptixFunctionTable));

        if (res != OPTIX_SUCCESS)
        {
            throw std::runtime_error("optixQueryFunctionTable failed.");
        }
    }

    // Create context
    {
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &OptixLogger::callback;
        options.logCallbackData = &g_logger;
        options.logCallbackLevel = 3; // Keep at warning level to suppress the disk cache messages.

        auto &backend = Backend::Get();
        auto res = m_api.optixDeviceContextCreate(backend.getCudaContext(), &options, &m_context);
        if (res != OPTIX_SUCCESS)
        {
            throw std::runtime_error("ERROR: initOptiX() optixDeviceContextCreate() failed");
        }
    }

    // DYNAMIC MATERIAL SYSTEM: Initialize MaterialManager instead of hardcoded materials
    {
        MaterialManager& materialManager = MaterialManager::Get();
        materialManager.init();
        
        // Print material information for debugging
        materialManager.printMaterialInfo();
    }

    assert((sizeof(SbtRecordHeader) % OPTIX_SBT_RECORD_ALIGNMENT) == 0);
    assert((sizeof(SbtRecordGeometryInstanceData) % OPTIX_SBT_RECORD_ALIGNMENT) == 0);

    // Create uninstanced geometry BLAS for all chunks and track instances
    // Also build sbtOffset mapping for proper SBT indexing across chunks
    unsigned int currentSbtOffset = 0;

    for (unsigned int chunkIndex = 0; chunkIndex < scene.numChunks; ++chunkIndex)
    {
        for (unsigned int objectId = GetUninstancedObjectIdBegin(); objectId < GetUninstancedObjectIdEnd(); ++objectId)
        {
            unsigned int blockId = ObjectIdToBlockId(objectId);

            // Only create geometry if this chunk has data for this object
            if (scene.getChunkGeometryAttributeSize(chunkIndex, objectId) > 0 &&
                scene.getChunkGeometryIndicesSize(chunkIndex, objectId) > 0)
            {
                // Create BLAS for the geometry
                GeometryData geometry = {};
                OptixTraversableHandle blasHandle = Scene::CreateGeometry(
                    m_api, m_context, Backend::Get().getCudaStream(),
                    geometry,
                    *scene.getChunkGeometryAttributes(chunkIndex, objectId),
                    *scene.getChunkGeometryIndices(chunkIndex, objectId),
                    scene.getChunkGeometryAttributeSize(chunkIndex, objectId),
                    scene.getChunkGeometryIndicesSize(chunkIndex, objectId));
                
                // SPARSE MAPPING: Register this chunk geometry in the mapping system
                size_t geometryIndex = m_geometries.size();
                m_geometries.push_back(geometry);
                m_chunkGeometryMap[{chunkIndex, objectId}] = geometryIndex;
                printf("SPARSE MAPPING: Registered chunk geometry (chunk=%u, object=%u) -> index=%zu\n", 
                       chunkIndex, objectId, geometryIndex);

                // Calculate chunk offset for world coordinates
                auto &voxelEngine = VoxelEngine::Get();
                auto &chunkConfig = voxelEngine.chunkConfig;
                unsigned int chunkX = chunkIndex % chunkConfig.chunksX;
                unsigned int chunkZ = (chunkIndex / chunkConfig.chunksX) % chunkConfig.chunksZ;
                unsigned int chunkY = chunkIndex / (chunkConfig.chunksX * chunkConfig.chunksZ);

                // Create an instance for the geometry with chunk offset
                OptixInstance instance = {};
                const float transformMatrix[12] =
                    {
                        1.0f, 0.0f, 0.0f, (float)(chunkX * VoxelChunk::width),
                        0.0f, 1.0f, 0.0f, (float)(chunkY * VoxelChunk::width),
                        0.0f, 0.0f, 1.0f, (float)(chunkZ * VoxelChunk::width)};
                memcpy(instance.transform, transformMatrix, sizeof(float) * 12);
                instance.instanceId = chunkIndex * scene.uninstancedGeometryCount + objectId;
                instance.visibilityMask = IsTransparentBlockType(blockId) ? 1 : 255;
                instance.sbtOffset = currentSbtOffset; // Use sequential sbtOffset to match SBT creation
                instance.flags = OPTIX_INSTANCE_FLAG_NONE;
                instance.traversableHandle = blasHandle;
                m_instances.push_back(instance);

                // Advance sbtOffset for next geometry instance
                currentSbtOffset += numTypesOfRays;
            }
        }
    }

    // Create instanced geometry BLAS
    for (unsigned int objectId = GetInstancedObjectIdBegin(); objectId < GetInstancedObjectIdEnd(); ++objectId)
    {
        // Convert objectId to array index (0-based)
        unsigned int arrayIndex = objectId - GetInstancedObjectIdBegin();

        // Create BLAS for instanced geometry using the loaded geometry data
        if (scene.m_instancedGeometryAttributeSize[arrayIndex] > 0 && scene.m_instancedGeometryIndicesSize[arrayIndex] > 0)
        {
            GeometryData geometry = {};
            OptixTraversableHandle blasHandle = Scene::CreateGeometry(
                m_api, m_context, Backend::Get().getCudaStream(),
                geometry,
                scene.m_instancedGeometryAttributes[arrayIndex],
                scene.m_instancedGeometryIndices[arrayIndex],
                scene.m_instancedGeometryAttributeSize[arrayIndex],
                scene.m_instancedGeometryIndicesSize[arrayIndex]);

            // SPARSE MAPPING: Register this instanced geometry in the mapping system
            size_t geometryIndex = m_geometries.size();
            m_geometries.push_back(geometry);
            m_instancedGeometryMap[objectId] = geometryIndex;
            printf("SPARSE MAPPING: Registered instanced geometry (objectId=%u) -> index=%zu, triangles=%zu\n", 
                   objectId, geometryIndex, geometry.numIndices / 3);
            objectIdxToBlasHandleMap[objectId] = blasHandle;
        }
        else
        {
            // Create empty geometry if no data available
            GeometryData geometry = {};
            geometry.indices = nullptr;
            geometry.attributes = nullptr;
            geometry.numIndices = 0;
            geometry.numAttributes = 0;
            geometry.gas = 0;
            printf("GEOMETRY CREATION DEBUG: Adding empty instanced geometry (objectId=%u, numIndices=%zu, numTriangles=%zu)\n", 
                   objectId, geometry.numIndices, geometry.numIndices / 3);
            m_geometries.push_back(geometry);
            objectIdxToBlasHandleMap[objectId] = 0;
        }

        // Create instances for this object type only if BLAS handle is valid
        if (objectIdxToBlasHandleMap[objectId] != 0) {
            for (int instanceId : scene.geometryInstanceIdMap[objectId])
            {
                OptixInstance instance = {};
                memcpy(instance.transform, scene.instanceTransformMatrices[instanceId].data(), sizeof(float) * 12);
                instance.instanceId = instanceId;
                instance.visibilityMask = 255;
                instance.sbtOffset = currentSbtOffset; // Use sequential sbtOffset continuing from uninstanced geometry
                instance.flags = OPTIX_INSTANCE_FLAG_NONE;
                instance.traversableHandle = objectIdxToBlasHandleMap[objectId];
                m_instances.push_back(instance);
                instanceIds.insert(instanceId);
            }
        }

        // Advance sbtOffset for next instanced geometry type
        currentSbtOffset += numTypesOfRays;
    }

    // Create entity geometry BLAS
    for (size_t entityIndex = 0; entityIndex < scene.getEntityCount(); ++entityIndex)
    {
        Entity *entity = scene.getEntity(entityIndex);
        if (entity && entity->getAttributeSize() > 0 && entity->getIndicesSize() > 0)
        {

            GeometryData geometry = {};
            OptixTraversableHandle blasHandle = Scene::CreateGeometry(
                m_api, m_context, Backend::Get().getCudaStream(),
                geometry,
                entity->getAttributes(),
                entity->getIndices(),
                entity->getAttributeSize(),
                entity->getIndicesSize(),
                true); // Allow updates for animated entities

            // SPARSE MAPPING: Register this entity geometry in the mapping system
            size_t geometryIndex = m_geometries.size();
            m_geometries.push_back(geometry);
            m_entityGeometryMap[entityIndex] = geometryIndex;
            printf("SPARSE MAPPING: Registered entity geometry (entityIndex=%zu) -> index=%zu, triangles=%zu\n", 
                   entityIndex, geometryIndex, geometry.numIndices / 3);

            // Create an instance for the entity
            OptixInstance instance = {};
            float transformMatrix[12];
            entity->getTransform().getTransformMatrix(transformMatrix);
            memcpy(instance.transform, transformMatrix, sizeof(float) * 12);

            // Use a unique instance ID for entities to avoid conflicts with block instances
            instance.instanceId = EntityConstants::ENTITY_INSTANCE_ID_OFFSET + static_cast<unsigned int>(entityIndex);
            instance.visibilityMask = 255;
            instance.sbtOffset = currentSbtOffset;
            instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            instance.traversableHandle = blasHandle;
            m_instances.push_back(instance);

            // Advance sbtOffset for next entity
            currentSbtOffset += numTypesOfRays;
        }
    }

    // Build BVH
    {
        // Handle case where there are no instances to prevent CUDA errors
        if (m_instances.empty()) {
            // Set to null handle for empty scene
            m_systemParameter.topObject = 0;
            // Don't build IAS for empty scene
        }
        else {
            CUdeviceptr d_instances;

            const size_t instancesSizeInBytes = sizeof(OptixInstance) * m_instances.size();

            CUDA_CHECK(cudaMalloc((void **)&d_instances, instancesSizeInBytes));
            CUDA_CHECK(cudaMemcpyAsync((void *)d_instances, m_instances.data(), instancesSizeInBytes, cudaMemcpyHostToDevice, Backend::Get().getCudaStream()));

            OptixBuildInput instanceInput = {};

            instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            instanceInput.instanceArray.instances = d_instances;
            instanceInput.instanceArray.numInstances = (unsigned int)m_instances.size();

            OptixAccelBuildOptions accelBuildOptions = {};

            accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
            accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

            OptixAccelBufferSizes iasBufferSizes = {};

            OPTIX_CHECK(m_api.optixAccelComputeMemoryUsage(m_context, &accelBuildOptions, &instanceInput, 1, &iasBufferSizes));

            CUDA_CHECK(cudaMalloc((void **)&m_d_ias[0], iasBufferSizes.outputSizeInBytes));

            CUdeviceptr d_tmp;

            CUDA_CHECK(cudaMalloc((void **)&d_tmp, iasBufferSizes.tempSizeInBytes));

            auto &backend = Backend::Get();
            OPTIX_CHECK(m_api.optixAccelBuild(m_context, backend.getCudaStream(),
                                              &accelBuildOptions, &instanceInput, 1,
                                              d_tmp, iasBufferSizes.tempSizeInBytes,
                                              m_d_ias[0], iasBufferSizes.outputSizeInBytes,
                                              &m_systemParameter.topObject, nullptr, 0));

            CUDA_CHECK(cudaStreamSynchronize(backend.getCudaStream()));

            CUDA_CHECK(cudaFree((void *)d_tmp));
            CUDA_CHECK(cudaFree((void *)d_instances)); // Don't need the instances anymore.
        }
    }

    // Options
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = 0;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOptions.numPayloadValues = 2;   // I need two to encode a 64-bit pointer to the per ray payload structure.
    pipelineCompileOptions.numAttributeValues = 2; // The minimum is two, for the barycentrics.
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "sysParam";

    OptixProgramGroupOptions programGroupOptions = {};

    // Shaders:
    std::vector<OptixModule> moduleList;
    std::vector<OptixProgramGroup> programGroups;
    std::vector<OptixProgramGroup> programGroupCallables;

    // Raygen
    {
        std::string ptxRaygeneration = ReadPtx("ptx/RayGen.ptx");
        OptixModule moduleRaygeneration;
        OPTIX_CHECK(m_api.optixModuleCreate(m_context, &moduleCompileOptions, &pipelineCompileOptions, ptxRaygeneration.c_str(), ptxRaygeneration.size(), nullptr, nullptr, &moduleRaygeneration));

        OptixProgramGroupDesc programGroupDescRaygeneration = {};
        programGroupDescRaygeneration.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        programGroupDescRaygeneration.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
        programGroupDescRaygeneration.raygen.module = moduleRaygeneration;
        programGroupDescRaygeneration.raygen.entryFunctionName = "__raygen__pathtracer";

        OptixProgramGroup programGroupRaygeneration;
        OPTIX_CHECK(m_api.optixProgramGroupCreate(m_context, &programGroupDescRaygeneration, 1, &programGroupOptions, nullptr, nullptr, &programGroupRaygeneration));
        programGroups.push_back(programGroupRaygeneration);

        moduleList.push_back(moduleRaygeneration);
    }

    // Miss
    {
        std::string ptxMiss = ReadPtx("ptx/Miss.ptx");
        OptixModule moduleMiss;
        OPTIX_CHECK(m_api.optixModuleCreate(m_context, &moduleCompileOptions, &pipelineCompileOptions, ptxMiss.c_str(), ptxMiss.size(), nullptr, nullptr, &moduleMiss));

        OptixProgramGroupDesc programGroupDescMissRadiance = {};
        programGroupDescMissRadiance.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        programGroupDescMissRadiance.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
        programGroupDescMissRadiance.miss.module = moduleMiss;
        programGroupDescMissRadiance.miss.entryFunctionName = "__miss__radiance";

        OptixProgramGroup programGroupMissRadiance;
        OPTIX_CHECK(m_api.optixProgramGroupCreate(m_context, &programGroupDescMissRadiance, 1, &programGroupOptions, nullptr, nullptr, &programGroupMissRadiance));
        programGroups.push_back(programGroupMissRadiance);

        OptixProgramGroupDesc programGroupDescMissBsdf = {};
        programGroupDescMissBsdf.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        programGroupDescMissBsdf.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
        programGroupDescMissBsdf.miss.module = moduleMiss;
        programGroupDescMissBsdf.miss.entryFunctionName = "__miss__bsdf_light";

        OptixProgramGroup programGroupMissBsdf;
        OPTIX_CHECK(m_api.optixProgramGroupCreate(m_context, &programGroupDescMissBsdf, 1, &programGroupOptions, nullptr, nullptr, &programGroupMissBsdf));
        programGroups.push_back(programGroupMissBsdf);

        OptixProgramGroupDesc programGroupDescMissVisibility = {};
        programGroupDescMissVisibility.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        programGroupDescMissVisibility.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
        programGroupDescMissVisibility.miss.module = moduleMiss;
        programGroupDescMissVisibility.miss.entryFunctionName = "__miss__visibility";

        OptixProgramGroup programGroupMissVisibility;
        OPTIX_CHECK(m_api.optixProgramGroupCreate(m_context, &programGroupDescMissVisibility, 1, &programGroupOptions, nullptr, nullptr, &programGroupMissVisibility));
        programGroups.push_back(programGroupMissVisibility);

        moduleList.push_back(moduleMiss);
    }

    // Closest hit
    {
        std::string ptxClosesthit = ReadPtx("ptx/ClosestHit.ptx");
        OptixModule moduleClosesthit;
        OPTIX_CHECK(m_api.optixModuleCreate(m_context, &moduleCompileOptions, &pipelineCompileOptions, ptxClosesthit.c_str(), ptxClosesthit.size(), nullptr, nullptr, &moduleClosesthit));

        OptixProgramGroupDesc programGroupDescHitRadiance = {};
        programGroupDescHitRadiance.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        programGroupDescHitRadiance.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
        programGroupDescHitRadiance.hitgroup.moduleCH = moduleClosesthit;
        programGroupDescHitRadiance.hitgroup.entryFunctionNameCH = "__closesthit__radiance";

        OptixProgramGroup programGroupHitRadiance;
        OPTIX_CHECK(m_api.optixProgramGroupCreate(m_context, &programGroupDescHitRadiance, 1, &programGroupOptions, nullptr, nullptr, &programGroupHitRadiance));
        programGroups.push_back(programGroupHitRadiance);

        OptixProgramGroupDesc programGroupDescHitShadow = {};
        programGroupDescHitShadow.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        programGroupDescHitShadow.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
        programGroupDescHitShadow.hitgroup.moduleCH = moduleClosesthit;
        programGroupDescHitShadow.hitgroup.entryFunctionNameCH = "__closesthit__bsdf_light";

        OptixProgramGroup programGroupHitShadow;
        OPTIX_CHECK(m_api.optixProgramGroupCreate(m_context, &programGroupDescHitShadow, 1, &programGroupOptions, nullptr, nullptr, &programGroupHitShadow));
        programGroups.push_back(programGroupHitShadow);

        moduleList.push_back(moduleClosesthit);
    }

    // Direct callables
    {
        // std::string ptxBsdf = ReadPtx("ptx/Bsdf.ptx");
        // OptixModule moduleBsdf;
        // OPTIX_CHECK(m_api.optixModuleCreate(m_context, &moduleCompileOptions, &pipelineCompileOptions, ptxBsdf.c_str(), ptxBsdf.size(), nullptr, nullptr, &moduleBsdf));

        // std::vector<OptixProgramGroupDesc> programGroupDescCallables;

        // OptixProgramGroupDesc pgd = {};

        // pgd.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        // pgd.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

        // pgd.callables.moduleDC = moduleBsdf;

        // pgd.callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_specular_reflection"; // 0
        // programGroupDescCallables.push_back(pgd);
        // pgd.callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_specular_reflection_transmission"; // 1
        // programGroupDescCallables.push_back(pgd);
        // pgd.callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_diffuse_reflection"; // 2
        // programGroupDescCallables.push_back(pgd);
        // pgd.callables.entryFunctionNameDC = "__direct_callable__eval_bsdf_diffuse_reflection"; // 3
        // programGroupDescCallables.push_back(pgd);
        // pgd.callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_microfacet_reflection"; // 4
        // programGroupDescCallables.push_back(pgd);
        // pgd.callables.entryFunctionNameDC = "__direct_callable__eval_bsdf_microfacet_reflection"; // 5
        // programGroupDescCallables.push_back(pgd);
        // pgd.callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_diffuse_reflection_transmission_thinfilm"; // 6
        // programGroupDescCallables.push_back(pgd);
        // pgd.callables.entryFunctionNameDC = "__direct_callable__eval_bsdf_diffuse_reflection_transmission_thinfilm"; // 7
        // programGroupDescCallables.push_back(pgd);
        // pgd.callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_microfacet_reflection_metal"; // 8
        // programGroupDescCallables.push_back(pgd);
        // pgd.callables.entryFunctionNameDC = "__direct_callable__eval_bsdf_microfacet_reflection_metal"; // 9
        // programGroupDescCallables.push_back(pgd);

        // programGroupCallables.resize(programGroupDescCallables.size());
        // OPTIX_CHECK(m_api.optixProgramGroupCreate(m_context, programGroupDescCallables.data(), programGroupDescCallables.size(), &programGroupOptions, nullptr, nullptr, programGroupCallables.data()));
        // programGroups.insert(programGroups.end(), programGroupCallables.begin(), programGroupCallables.end());
        // moduleList.push_back(moduleBsdf);
    }

    // Pipeline
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    {
        pipelineLinkOptions.maxTraceDepth = 2;
        OPTIX_CHECK(m_api.optixPipelineCreate(m_context, &pipelineCompileOptions, &pipelineLinkOptions, programGroups.data(), (unsigned int)programGroups.size(), nullptr, nullptr, &m_pipeline));
    }

    // Stack size
    {
        OptixStackSizes stackSizesPipeline = {};
        for (size_t i = 0; i < programGroups.size(); ++i)
        {
            OptixStackSizes stackSizes;

            OPTIX_CHECK(m_api.optixProgramGroupGetStackSize(programGroups[i], &stackSizes, m_pipeline));

            stackSizesPipeline.cssRG = std::max(stackSizesPipeline.cssRG, stackSizes.cssRG);
            stackSizesPipeline.cssMS = std::max(stackSizesPipeline.cssMS, stackSizes.cssMS);
            stackSizesPipeline.cssCH = std::max(stackSizesPipeline.cssCH, stackSizes.cssCH);
            stackSizesPipeline.cssAH = std::max(stackSizesPipeline.cssAH, stackSizes.cssAH);
            stackSizesPipeline.cssIS = std::max(stackSizesPipeline.cssIS, stackSizes.cssIS);
            stackSizesPipeline.cssCC = std::max(stackSizesPipeline.cssCC, stackSizes.cssCC);
            stackSizesPipeline.dssDC = std::max(stackSizesPipeline.dssDC, stackSizes.dssDC);
        }

        const unsigned int cssCCTree = stackSizesPipeline.cssCC;
        const unsigned int cssCHOrMSPlusCCTree = std::max(stackSizesPipeline.cssCH, stackSizesPipeline.cssMS) + cssCCTree;
        const unsigned int directCallableStackSizeFromTraversal = stackSizesPipeline.dssDC;
        const unsigned int directCallableStackSizeFromState = stackSizesPipeline.dssDC;
        const unsigned int continuationStackSize = stackSizesPipeline.cssRG + cssCCTree + cssCHOrMSPlusCCTree * (std::max(1u, pipelineLinkOptions.maxTraceDepth) - 1u) +
                                                   std::min(1u, pipelineLinkOptions.maxTraceDepth) * std::max(cssCHOrMSPlusCCTree, stackSizesPipeline.cssAH + stackSizesPipeline.cssIS);
        const unsigned int maxTraversableGraphDepth = 2;

        OPTIX_CHECK(m_api.optixPipelineSetStackSize(m_pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState, continuationStackSize, maxTraversableGraphDepth));
    }

    // Set up Shader Binding Table (SBT)

    // Raygeneration group
    {
        SbtRecordHeader sbtRecordRaygeneration;
        OPTIX_CHECK(m_api.optixSbtRecordPackHeader(programGroups[0], &sbtRecordRaygeneration));
        CUDA_CHECK(cudaMalloc((void **)&m_d_sbtRecordRaygeneration, sizeof(SbtRecordHeader)));
        CUDA_CHECK(cudaMemcpyAsync((void *)m_d_sbtRecordRaygeneration, &sbtRecordRaygeneration, sizeof(SbtRecordHeader), cudaMemcpyHostToDevice, Backend::Get().getCudaStream()));
    }

    // Miss group
    {
        constexpr unsigned int numMissShaders = 3;
        std::vector<SbtRecordHeader> sbtRecordMiss(numMissShaders);

        OPTIX_CHECK(m_api.optixSbtRecordPackHeader(programGroups[1], &sbtRecordMiss[0]));
        OPTIX_CHECK(m_api.optixSbtRecordPackHeader(programGroups[2], &sbtRecordMiss[1]));
        OPTIX_CHECK(m_api.optixSbtRecordPackHeader(programGroups[3], &sbtRecordMiss[2]));

        CUDA_CHECK(cudaMalloc((void **)&m_d_sbtRecordMiss, sizeof(SbtRecordHeader) * numMissShaders));
        CUDA_CHECK(cudaMemcpyAsync((void *)m_d_sbtRecordMiss, sbtRecordMiss.data(), sizeof(SbtRecordHeader) * numMissShaders, cudaMemcpyHostToDevice, Backend::Get().getCudaStream()));
    }

    // Hit group
    {
        OPTIX_CHECK(m_api.optixSbtRecordPackHeader(programGroups[4], &m_sbtRecordHitRadiance));
        OPTIX_CHECK(m_api.optixSbtRecordPackHeader(programGroups[5], &m_sbtRecordHitShadow));

        // Calculate total instances needed for all chunks
        unsigned int totalInstances = 0;
        for (unsigned int chunkIndex = 0; chunkIndex < scene.numChunks; ++chunkIndex)
        {
            for (unsigned int objectId = GetUninstancedObjectIdBegin(); objectId < GetUninstancedObjectIdEnd(); ++objectId)
            {
                if (scene.getChunkGeometryAttributeSize(chunkIndex, objectId) > 0 &&
                    scene.getChunkGeometryIndicesSize(chunkIndex, objectId) > 0)
                {
                    totalInstances++;
                }
            }
        }
        // Add space for instanced objects
        totalInstances += scene.instancedGeometryCount;
        // Add space for entities
        totalInstances += static_cast<unsigned int>(scene.getEntityCount());

        m_sbtRecordGeometryInstanceData.resize(totalInstances * numTypesOfRays);

        // Initialize SBT records for all geometry instances
        unsigned int sbtRecordIndex = 0;

        // SPARSE GEOMETRY MAPPING: Setup SBT records using sparse mapping system
        // Process chunk geometries that actually exist in the mapping
        for (const auto& entry : m_chunkGeometryMap)
        {
            unsigned int chunkIndex = entry.first.chunkIndex;
            unsigned int objectId = entry.first.objectId;
            size_t geometryIndex = entry.second;
            
            // SPARSE SBT VALIDATION: Validate mapped geometry exists and is valid
            if (geometryIndex >= m_geometries.size()) {
                printf("SPARSE SBT ERROR: Mapped geometry index out of bounds (chunk=%u, object=%u, index=%zu, size=%zu)\n", 
                       chunkIndex, objectId, geometryIndex, m_geometries.size());
                continue;
            }
            
            const GeometryData &geometry = m_geometries[geometryIndex];
            
            // Skip invalid geometry to prevent crashes
            if (geometry.indices == nullptr || geometry.attributes == nullptr) {
                printf("SPARSE SBT SKIP: Null geometry pointers (chunk=%u, object=%u)\n", chunkIndex, objectId);
                continue;
            }
            
            // Skip empty geometry
            if (geometry.numIndices == 0 || geometry.numAttributes == 0) {
                printf("SPARSE SBT SKIP: Empty geometry (chunk=%u, object=%u, indices=%zu, attributes=%zu)\n", 
                       chunkIndex, objectId, geometry.numIndices, geometry.numAttributes);
                continue;
            }
            
            // Skip geometry without valid GAS
            if (geometry.gas == 0) {
                printf("SPARSE SBT SKIP: Invalid GAS (chunk=%u, object=%u)\n", chunkIndex, objectId);
                continue;
            }
            
            printf("SPARSE SBT SUCCESS: Adding chunk geometry (chunk=%u, object=%u, mapped_index=%zu, triangles=%zu)\n", 
                   chunkIndex, objectId, geometryIndex, geometry.numIndices / 3);

            for (unsigned int rayType = 0; rayType < numTypesOfRays; ++rayType)
            {
                if (rayType == 0)
                {
                    memcpy(m_sbtRecordGeometryInstanceData[sbtRecordIndex].header, m_sbtRecordHitRadiance.header, OPTIX_SBT_RECORD_HEADER_SIZE);
                }
                else
                {
                    memcpy(m_sbtRecordGeometryInstanceData[sbtRecordIndex].header, m_sbtRecordHitShadow.header, OPTIX_SBT_RECORD_HEADER_SIZE);
                }

                m_sbtRecordGeometryInstanceData[sbtRecordIndex].data.indices = (Int3 *)geometry.indices;
                m_sbtRecordGeometryInstanceData[sbtRecordIndex].data.attributes = (VertexAttributes *)geometry.attributes;
                
                // DYNAMIC MATERIALS: Use MaterialManager to get correct material for block type
                unsigned int blockType = ObjectIdToBlockId(objectId);
                unsigned int materialIndex = MaterialManager::Get().getMaterialIndexForBlock(blockType);
                m_sbtRecordGeometryInstanceData[sbtRecordIndex].data.materialIndex = materialIndex;
                sbtRecordIndex++;
            }
        }

        // SPARSE GEOMETRY MAPPING: Setup SBT records for instanced geometry using sparse mapping
        // Process instanced geometries that actually exist in the mapping
        for (const auto& entry : m_instancedGeometryMap)
        {
            unsigned int objectId = entry.first;
            size_t geometryIndex = entry.second;
            
            // SPARSE SBT VALIDATION: Validate mapped instanced geometry exists and is valid
            if (geometryIndex >= m_geometries.size()) {
                printf("SPARSE SBT ERROR: Mapped instanced geometry index out of bounds (objectId=%u, index=%zu, size=%zu)\n", 
                       objectId, geometryIndex, m_geometries.size());
                continue;
            }
            
            const GeometryData &geometry = m_geometries[geometryIndex];
            
            printf("SPARSE SBT DEBUG: Reading instanced geometry (objectId=%u, mapped_index=%zu): numIndices=%zu, numAttributes=%zu, indices=%p, attributes=%p\n",
                   objectId, geometryIndex, geometry.numIndices, geometry.numAttributes, geometry.indices, geometry.attributes);
            
            // Skip invalid instanced geometry
            if (geometry.indices == nullptr || geometry.attributes == nullptr) {
                printf("SPARSE SBT SKIP: Null instanced geometry pointers (objectId=%u)\n", objectId);
                continue;
            }
            
            // Skip empty instanced geometry
            if (geometry.numIndices == 0 || geometry.numAttributes == 0) {
                printf("SPARSE SBT SKIP: Empty instanced geometry (objectId=%u, indices=%zu, attributes=%zu)\n", 
                       objectId, geometry.numIndices, geometry.numAttributes);
                continue;
            }
            
            // Skip geometry without valid GAS
            if (geometry.gas == 0) {
                printf("SPARSE SBT SKIP: Invalid instanced GAS (objectId=%u)\n", objectId);
                continue;
            }
            
            printf("SPARSE SBT SUCCESS: Adding instanced geometry (objectId=%u, mapped_index=%zu, triangles=%zu)\n", 
                   objectId, geometryIndex, geometry.numIndices / 3);

            for (unsigned int rayType = 0; rayType < numTypesOfRays; ++rayType)
            {
                if (rayType == 0)
                {
                    memcpy(m_sbtRecordGeometryInstanceData[sbtRecordIndex].header, m_sbtRecordHitRadiance.header, OPTIX_SBT_RECORD_HEADER_SIZE);
                }
                else
                {
                    memcpy(m_sbtRecordGeometryInstanceData[sbtRecordIndex].header, m_sbtRecordHitShadow.header, OPTIX_SBT_RECORD_HEADER_SIZE);
                }

                m_sbtRecordGeometryInstanceData[sbtRecordIndex].data.indices = (Int3 *)geometry.indices;
                m_sbtRecordGeometryInstanceData[sbtRecordIndex].data.attributes = (VertexAttributes *)geometry.attributes;
                
                // DYNAMIC MATERIALS: Use MaterialManager to get correct material for block type
                unsigned int blockType = ObjectIdToBlockId(objectId);
                unsigned int materialIndex = MaterialManager::Get().getMaterialIndexForBlock(blockType);
                m_sbtRecordGeometryInstanceData[sbtRecordIndex].data.materialIndex = materialIndex;
                sbtRecordIndex++;
            }
        }

        // SPARSE GEOMETRY MAPPING: Setup SBT records for entities using sparse mapping
        // Process entity geometries that actually exist in the mapping
        for (const auto& entry : m_entityGeometryMap)
        {
            size_t entityIndex = entry.first;
            size_t geometryIndex = entry.second;
            
            // Get entity for material information
            Entity *entity = scene.getEntity(entityIndex);
            if (!entity) {
                printf("SPARSE SBT ERROR: Entity not found (entityIndex=%zu)\n", entityIndex);
                continue;
            }
            
            // SPARSE SBT VALIDATION: Validate mapped entity geometry exists and is valid
            if (geometryIndex >= m_geometries.size()) {
                printf("SPARSE SBT ERROR: Mapped entity geometry index out of bounds (entityIndex=%zu, index=%zu, size=%zu)\n", 
                       entityIndex, geometryIndex, m_geometries.size());
                continue;
            }
            
            const GeometryData &geometry = m_geometries[geometryIndex];
            
            // Skip invalid entity geometry
            if (geometry.indices == nullptr || geometry.attributes == nullptr) {
                printf("SPARSE SBT SKIP: Null entity geometry pointers (entityIndex=%zu)\n", entityIndex);
                continue;
            }
            
            // Skip empty entity geometry
            if (geometry.numIndices == 0 || geometry.numAttributes == 0) {
                printf("SPARSE SBT SKIP: Empty entity geometry (entityIndex=%zu, indices=%zu, attributes=%zu)\n", 
                       entityIndex, geometry.numIndices, geometry.numAttributes);
                continue;
            }
            
            // Skip geometry without valid GAS
            if (geometry.gas == 0) {
                printf("SPARSE SBT SKIP: Invalid entity GAS (entityIndex=%zu)\n", entityIndex);
                continue;
            }
            
            printf("SPARSE SBT SUCCESS: Adding entity geometry (entityIndex=%zu, mapped_index=%zu, triangles=%zu)\n", 
                   entityIndex, geometryIndex, geometry.numIndices / 3);

            for (unsigned int rayType = 0; rayType < numTypesOfRays; ++rayType)
            {
                if (rayType == 0)
                {
                    memcpy(m_sbtRecordGeometryInstanceData[sbtRecordIndex].header, m_sbtRecordHitRadiance.header, OPTIX_SBT_RECORD_HEADER_SIZE);
                }
                else
                {
                    memcpy(m_sbtRecordGeometryInstanceData[sbtRecordIndex].header, m_sbtRecordHitShadow.header, OPTIX_SBT_RECORD_HEADER_SIZE);
                }

                m_sbtRecordGeometryInstanceData[sbtRecordIndex].data.indices = (Int3 *)geometry.indices;
                m_sbtRecordGeometryInstanceData[sbtRecordIndex].data.attributes = (VertexAttributes *)geometry.attributes;
                
                // DYNAMIC MATERIALS: Use MaterialManager to get correct material for entity type
                unsigned int entityMaterialIndex = MaterialManager::Get().getMaterialIndexForEntity(static_cast<unsigned int>(entity->getType()));
                m_sbtRecordGeometryInstanceData[sbtRecordIndex].data.materialIndex = entityMaterialIndex;

                sbtRecordIndex++;
            }
        }

        // Only allocate and copy SBT data if we have instances
        if (totalInstances > 0) {
            CUDA_CHECK(cudaMalloc((void **)&m_d_sbtRecordGeometryInstanceData, sizeof(SbtRecordGeometryInstanceData) * totalInstances * numTypesOfRays));
            CUDA_CHECK(cudaMemcpyAsync((void *)m_d_sbtRecordGeometryInstanceData, m_sbtRecordGeometryInstanceData.data(), sizeof(SbtRecordGeometryInstanceData) * totalInstances * numTypesOfRays, cudaMemcpyHostToDevice, Backend::Get().getCudaStream()));
        }
        else {
            m_d_sbtRecordGeometryInstanceData = nullptr;
        }
    }

    // Direct callables
    // std::vector<SbtRecordHeader> sbtRecordCallables(programGroupCallables.size());
    // {
    //     for (size_t i = 0; i < programGroupCallables.size(); ++i)
    //     {
    //         OPTIX_CHECK(m_api.optixSbtRecordPackHeader(programGroupCallables[i], &sbtRecordCallables[i]));
    //     }

    //     CUDA_CHECK(cudaMalloc((void **)&m_d_sbtRecordCallables, sizeof(SbtRecordHeader) * sbtRecordCallables.size()));
    //     CUDA_CHECK(cudaMemcpyAsync((void *)m_d_sbtRecordCallables, sbtRecordCallables.data(), sizeof(SbtRecordHeader) * sbtRecordCallables.size(), cudaMemcpyHostToDevice, Backend::Get().getCudaStream()));
    // }

    // Setup the OptixShaderBindingTable
    {
        m_sbt.raygenRecord = m_d_sbtRecordRaygeneration;

        m_sbt.exceptionRecord = 0;

        m_sbt.missRecordBase = m_d_sbtRecordMiss;
        m_sbt.missRecordStrideInBytes = (unsigned int)sizeof(SbtRecordHeader);
        m_sbt.missRecordCount = numTypesOfRays;

        m_sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(m_d_sbtRecordGeometryInstanceData);
        m_sbt.hitgroupRecordStrideInBytes = (unsigned int)sizeof(SbtRecordGeometryInstanceData);
        m_sbt.hitgroupRecordCount = m_sbtRecordGeometryInstanceData.size();

        // m_sbt.callablesRecordBase = m_d_sbtRecordCallables;
        // m_sbt.callablesRecordStrideInBytes = (unsigned int)sizeof(SbtRecordHeader);
        // m_sbt.callablesRecordCount = (unsigned int)sbtRecordCallables.size();
    }

    // Setup "sysParam" data.
    {
        // DYNAMIC MATERIALS: Use MaterialManager's GPU materials instead of hardcoded array
        MaterialManager& materialManager = MaterialManager::Get();
        m_systemParameter.materialParameters = materialManager.getDeviceMaterialParameters();
        m_systemParameter.numMaterialParameters = materialManager.getMaterialCount();

        CUDA_CHECK(cudaMalloc((void **)&m_d_systemParameter, sizeof(SystemParameter)));
        CUDA_CHECK(cudaMemcpyAsync((void *)m_d_systemParameter, &m_systemParameter, sizeof(SystemParameter), cudaMemcpyHostToDevice, Backend::Get().getCudaStream()));
    }

    // Destroy modules
    for (auto &module : moduleList)
    {
        OPTIX_CHECK(m_api.optixModuleDestroy(module));
    }
}