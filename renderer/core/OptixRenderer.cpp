#include "core/Backend.h"
#include "core/OfflineBackend.h"
#include "util/DebugUtils.h"
#include "util/FileUtils.h"
#include "OptixRenderer.h"
#include "core/Scene.h"
#include "core/BufferManager.h"
#include "assets/TextureManager.h"
#include "core/GlobalSettings.h"
#include "sky/Sky.h"
#include "core/RenderCamera.h"
#ifndef OFFLINE_MODE
#include "core/InputHandler.h"
#endif

#include "util/BufferUtils.h"

// New asset management system
#include "assets/AssetRegistry.h"
#include "assets/MaterialManager.h"
#include "assets/ModelManager.h"
#include "assets/BlockManager.h"

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

#include "voxelengine/BlockType.h"
#include "voxelengine/VoxelEngine.h"

namespace
{

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

} // namespace

// Helper functions for geometry index calculations
unsigned int OptixRenderer::getUninstancedGeometryIndex(unsigned int chunkIndex, unsigned int objectId) const
{
    const auto &scene = Scene::Get();
    return chunkIndex * scene.uninstancedGeometryCount + objectId;
}

unsigned int OptixRenderer::getInstancedGeometryIndex(unsigned int objectId) const
{
    const auto &scene = Scene::Get();
    return scene.numChunks * scene.uninstancedGeometryCount + (objectId - Assets::BlockManager::Get().getInstancedObjectIdBegin());
}

unsigned int OptixRenderer::getEntityGeometryIndex(unsigned int entityIndex) const
{
    const auto &scene = Scene::Get();
    return scene.numChunks * scene.uninstancedGeometryCount + scene.instancedGeometryCount + entityIndex;
}

// Helper function to update SBT record geometry data for all ray types
void OptixRenderer::updateSbtRecordGeometryData(unsigned int geometryIndex, const GeometryData &geometry)
{
    unsigned int sbtBaseIndex = geometryIndex * NUM_RAY_TYPES;

    // Bounds check
    assert(sbtBaseIndex + NUM_RAY_TYPES - 1 < m_sbtRecordGeometryInstanceData.size());

    for (unsigned int rayType = 0; rayType < NUM_RAY_TYPES; ++rayType)
    {
        unsigned int sbtIndex = sbtBaseIndex + rayType;
        m_sbtRecordGeometryInstanceData[sbtIndex].data.indices = (Int3 *)geometry.indices;
        m_sbtRecordGeometryInstanceData[sbtIndex].data.attributes = (VertexAttributes *)geometry.attributes;
    }
}

// Helper function to initialize an SBT record with headers, geometry data, and material index
void OptixRenderer::initializeSbtRecord(unsigned int sbtRecordIndex, unsigned int geometryIndex, unsigned int materialIndex)
{
    if (sbtRecordIndex + NUM_RAY_TYPES - 1 >= m_sbtRecordGeometryInstanceData.size() ||
        geometryIndex >= m_geometries.size())
    {
        return;
    }

    for (unsigned int rayType = 0; rayType < NUM_RAY_TYPES; ++rayType)
    {
        unsigned int currentSbtIndex = sbtRecordIndex + rayType;

        // Set header based on ray type
        if (rayType == 0)
        {
            memcpy(m_sbtRecordGeometryInstanceData[currentSbtIndex].header, m_sbtRecordHitRadiance.header, OPTIX_SBT_RECORD_HEADER_SIZE);
        }
        else
        {
            memcpy(m_sbtRecordGeometryInstanceData[currentSbtIndex].header, m_sbtRecordHitShadow.header, OPTIX_SBT_RECORD_HEADER_SIZE);
        }

        // Set geometry data
        m_sbtRecordGeometryInstanceData[currentSbtIndex].data.indices = (Int3 *)m_geometries[geometryIndex].indices;
        m_sbtRecordGeometryInstanceData[currentSbtIndex].data.attributes = (VertexAttributes *)m_geometries[geometryIndex].attributes;
        m_sbtRecordGeometryInstanceData[currentSbtIndex].data.materialIndex = materialIndex;
    }
}

void OptixRenderer::clear()
{
    auto &backend = Backend::Get();
    CUDA_CHECK(cudaStreamSynchronize(backend.getCudaStream()));

    // Material memory is now managed by MaterialManager, don't free here
    if (m_d_systemParameter)
    {
        CUDA_CHECK(cudaFree((void *)m_d_systemParameter));
    }

    for (size_t i = 0; i < m_geometries.size(); ++i)
    {
        if (m_geometries[i].indices)
        {
            cudaError_t err = cudaFree((void *)m_geometries[i].indices);
            if (err != cudaSuccess && err != cudaErrorInvalidValue)
            {
                CUDA_CHECK(err);
            }
        }
        if (m_geometries[i].attributes)
        {
            cudaError_t err = cudaFree((void *)m_geometries[i].attributes);
            if (err != cudaSuccess && err != cudaErrorInvalidValue)
            {
                CUDA_CHECK(err);
            }
        }
        if (m_geometries[i].gas)
        {
            cudaError_t err = cudaFree((void *)m_geometries[i].gas);
            if (err != cudaSuccess && err != cudaErrorInvalidValue)
            {
                CUDA_CHECK(err);
            }
        }
    }
    if (m_d_ias[0])
    {
        CUDA_CHECK(cudaFree((void *)m_d_ias[0]));
    }
    if (m_d_ias[1])
    {
        CUDA_CHECK(cudaFree((void *)m_d_ias[1]));
    }

    if (m_d_sbtRecordRaygeneration)
    {
        CUDA_CHECK(cudaFree((void *)m_d_sbtRecordRaygeneration));
    }
    if (m_d_sbtRecordMiss)
    {
        CUDA_CHECK(cudaFree((void *)m_d_sbtRecordMiss));
    }
    if (m_d_sbtRecordCallables)
    {
        CUDA_CHECK(cudaFree((void *)m_d_sbtRecordCallables));
    }

    if (m_d_sbtRecordGeometryInstanceData)
    {
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

#ifndef OFFLINE_MODE
#else
    RenderCamera::Get().camera.update();
#endif

    const Camera &currentCameraForShader = RenderCamera::Get().camera;
    const Camera &historyCameraForShader = RenderCamera::Get().historyCamera;

    m_systemParameter.camera = currentCameraForShader;
    m_systemParameter.prevCamera = historyCameraForShader;

    m_systemParameter.timeInSecond = backend.getTimer().getTime() / 1000.0f; // Convert ms to seconds

    const auto &skyModel = SkyModel::Get();
    m_systemParameter.sunDir = skyModel.getSunDir();
    m_systemParameter.accumulatedSkyLuminance = skyModel.getAccumulatedSkyLuminance();
    m_systemParameter.accumulatedSunLuminance = skyModel.getAccumulatedSunLuminance();
    m_systemParameter.accumulatedLocalLightLuminance = scene.accumulatedLocalLightLuminance;
    m_systemParameter.lights = scene.m_lights;
    m_systemParameter.numLights = scene.m_currentNumLights;
    m_systemParameter.prevNumLights = scene.m_prevNumLights;
    m_systemParameter.lightAliasTable = scene.d_lightAliasTable;
    m_systemParameter.instanceLightMapping = scene.d_instanceLightMapping;
    m_systemParameter.prevLightIdToCurrentId = scene.d_prevLightIdToCurrentId;
    m_systemParameter.lightsStateDirty = scene.m_lightsJustUpdated;
    m_systemParameter.instanceLightMappingSize = scene.instanceLightMappingSize;
    
    // Reset the flag after reading it
    if (scene.m_lightsJustUpdated)
    {
        scene.m_lightsJustUpdated = false;
    }

    updateAnimatedEntities(backend.getCudaStream(), backend.getTimer().getTime() / 1000.0f, backend.getTimer().getDeltaTime());

    BufferSetFloat4(bufferManager.GetBufferDim(UIBuffer), bufferManager.GetBuffer2D(UIBuffer), Float4(0.0f));

    m_systemParameter.prevTopObject = m_systemParameter.topObject;
    int nextIndex = (m_currentIasIdx + 1) % 2;
    buildInstanceAccelerationStructure(backend.getCudaStream(), nextIndex);
    m_currentIasIdx = nextIndex;

    CUDA_CHECK(cudaMemcpyAsync((void *)m_d_systemParameter, &m_systemParameter, sizeof(SystemParameter), cudaMemcpyHostToDevice, backend.getCudaStream()));

    OPTIX_CHECK(m_api.optixLaunch(m_pipeline, backend.getCudaStream(), (CUdeviceptr)m_d_systemParameter, sizeof(SystemParameter), &m_sbt, m_width, m_height, 1));

    CUDA_CHECK(cudaStreamSynchronize(backend.getCudaStream()));

    BufferCopyFloat4(bufferManager.GetBufferDim(GeoNormalThinfilmBuffer), bufferManager.GetBuffer2D(GeoNormalThinfilmBuffer), bufferManager.GetBuffer2D(PrevGeoNormalThinfilmBuffer));
    BufferCopyFloat4(bufferManager.GetBufferDim(AlbedoBuffer), bufferManager.GetBuffer2D(AlbedoBuffer), bufferManager.GetBuffer2D(PrevAlbedoBuffer));
    BufferCopyFloat4(bufferManager.GetBufferDim(MaterialParameterBuffer), bufferManager.GetBuffer2D(MaterialParameterBuffer), bufferManager.GetBuffer2D(PrevMaterialParameterBuffer));

    // Clear the lights dirty flag after rendering one frame with it. This ensures ReSTIR only remaps light IDs for one frame after lights change
    if (scene.m_lightsNeedUpdate)
    {
        scene.m_lightsNeedUpdate = false;
    }
}

void OptixRenderer::updateAnimatedEntities(CUstream cudaStream, float currentTime, float deltaTime)
{
    auto &scene = Scene::Get();

    // Delta time is now passed as parameter from Backend's Timer

    // Update each animated entity
    for (size_t entityIndex = 0; entityIndex < scene.getEntityCount(); ++entityIndex)
    {
        Entity *entity = scene.getEntity(entityIndex);
        if (entity && entity->getAttributeSize() > 0 && entity->getIndicesSize() > 0)
        {
            // Update entity animation
            entity->update(deltaTime);

            // Check if this entity has animations that require geometry updates
            if (entity->hasAnimation())
            {
                // Calculate the correct geometry index for entities
                unsigned int geometryIndex = getEntityGeometryIndex(entityIndex);

                if (geometryIndex < m_geometries.size())
                {
                    GeometryData &geometry = m_geometries[geometryIndex];

                    // Validate entity data before BLAS update
                    assert(!(!entity->getAttributes() || !entity->getIndices() || entity->getAttributeSize() == 0 || geometry.gas == 0));

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

                            break;
                        }
                    }

                    // Update SBT records for the animated entity
                    updateSbtRecordGeometryData(getEntityGeometryIndex(entityIndex), geometry);
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpyAsync((void *)m_d_sbtRecordGeometryInstanceData, m_sbtRecordGeometryInstanceData.data(),
                               sizeof(SbtRecordGeometryInstanceData) * m_sbtRecordGeometryInstanceData.size(),
                               cudaMemcpyHostToDevice, cudaStream));
}

void OptixRenderer::createGasAndOptixInstanceForUninstancedObject()
{
    auto &scene = Scene::Get();

    assert(Assets::BlockManager::Get().getUninstancedObjectIdBegin() == 0);

    // Ensure m_geometries is properly sized for all uninstanced geometries
    size_t requiredSize = scene.numChunks * scene.uninstancedGeometryCount;
    if (m_geometries.size() < requiredSize)
    {
        m_geometries.resize(requiredSize);
    }

    for (unsigned int chunkIndex = 0; chunkIndex < scene.numChunks; ++chunkIndex)
    {
        for (unsigned int objectId = Assets::BlockManager::Get().getUninstancedObjectIdBegin(); objectId < Assets::BlockManager::Get().getUninstancedObjectIdEnd(); ++objectId)
        {
            auto &scene = Scene::Get();
            unsigned int blockId = Assets::BlockManager::Get().objectIdToBlockId(objectId);
            unsigned int geometryIndex = getUninstancedGeometryIndex(chunkIndex, objectId);

            assert(scene.getChunkGeometryAttributeSize(chunkIndex, objectId) > 0);
            assert(scene.getChunkGeometryIndicesSize(chunkIndex, objectId) > 0);

            GeometryData &geometry = m_geometries[geometryIndex];
            if (geometry.gas != 0)
            {
                CUDA_CHECK(cudaFree((void *)geometry.gas));
                geometry.gas = 0;
            }

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
            instance.visibilityMask = Assets::BlockManager::Get().isTransparentBlockType(blockId) ? 1 : 255;
            instance.sbtOffset = geometryIndex * NUM_RAY_TYPES;
            instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            instance.traversableHandle = blasHandle;
            m_instances.push_back(instance);
        }
    }
}

void OptixRenderer::updateGasAndOptixInstanceForUninstancedObject(unsigned int chunkIndex, unsigned int objectId)
{
    auto &scene = Scene::Get();
    unsigned int geometryIndex = getUninstancedGeometryIndex(chunkIndex, objectId);

    assert(geometryIndex < m_geometries.size());
    assert(scene.getChunkGeometryAttributeSize(chunkIndex, objectId) > 0);
    assert(scene.getChunkGeometryIndicesSize(chunkIndex, objectId) > 0);

    GeometryData &geometry = m_geometries[geometryIndex];
    if (geometry.gas != 0)
    {
        CUDA_CHECK(cudaFree((void *)geometry.gas));
        geometry.gas = 0;
    }

    OptixTraversableHandle blasHandle = Scene::CreateGeometry(
        m_api, m_context, Backend::Get().getCudaStream(),
        geometry,
        *scene.getChunkGeometryAttributes(chunkIndex, objectId),
        *scene.getChunkGeometryIndices(chunkIndex, objectId),
        scene.getChunkGeometryAttributeSize(chunkIndex, objectId),
        scene.getChunkGeometryIndicesSize(chunkIndex, objectId));

    // Find and update the corresponding instance, and get its sbtOffset
    unsigned int sbtIndex = 0;
    bool instanceFound = false;
    for (auto &instance : m_instances)
    {
        if (instance.instanceId == geometryIndex)
        {
            instance.traversableHandle = blasHandle;
            sbtIndex = instance.sbtOffset;
            instanceFound = true;
            break;
        }
    }

    // Update shader binding table record using the correct SBT index
    if (instanceFound)
    {
        // Update geometry data using the sbtIndex directly since it's already calculated
        if (sbtIndex + NUM_RAY_TYPES - 1 < m_sbtRecordGeometryInstanceData.size())
        {
            for (unsigned int rayType = 0; rayType < NUM_RAY_TYPES; ++rayType)
            {
                m_sbtRecordGeometryInstanceData[sbtIndex + rayType].data.indices = (Int3 *)geometry.indices;
                m_sbtRecordGeometryInstanceData[sbtIndex + rayType].data.attributes = (VertexAttributes *)geometry.attributes;
            }
        }
    }
}

void OptixRenderer::createBlasForInstancedObjects()
{
    auto &scene = Scene::Get();

    for (unsigned int objectId = Assets::BlockManager::Get().getInstancedObjectIdBegin(); objectId < Assets::BlockManager::Get().getInstancedObjectIdEnd(); ++objectId)
    {
        // Convert objectId to array index (0-based)
        unsigned int arrayIndex = objectId - Assets::BlockManager::Get().getInstancedObjectIdBegin();

        // ALL instanced blocks should have geometry loaded at initialization
        // This ensures we can dynamically add instances without creating BLAS on the fly
        assert(!(scene.m_instancedGeometryAttributeSize[arrayIndex] == 0 || scene.m_instancedGeometryIndicesSize[arrayIndex] == 0));

        GeometryData geometry = {};
        OptixTraversableHandle blasHandle = Scene::CreateGeometry(
            m_api, m_context, Backend::Get().getCudaStream(),
            geometry,
            scene.m_instancedGeometryAttributes[arrayIndex],
            scene.m_instancedGeometryIndices[arrayIndex],
            scene.m_instancedGeometryAttributeSize[arrayIndex],
            scene.m_instancedGeometryIndicesSize[arrayIndex]);

        m_geometries.push_back(geometry);
        m_objectIdxToBlasHandleMap[objectId] = blasHandle;
    }
}

void OptixRenderer::createOptixInstanceForInstancedObject()
{
    auto &scene = Scene::Get();

    for (unsigned int objectId = Assets::BlockManager::Get().getInstancedObjectIdBegin(); objectId < Assets::BlockManager::Get().getInstancedObjectIdEnd(); ++objectId)
    {
        unsigned int geometryIndex = getInstancedGeometryIndex(objectId);
        for (int instanceId : scene.geometryInstanceIdMap[objectId])
        {
            OptixInstance instance = {};
            memcpy(instance.transform, scene.instanceTransformMatrices[instanceId].data(), sizeof(float) * 12);
            instance.instanceId = instanceId;
            instance.visibilityMask = 255;
            instance.sbtOffset = geometryIndex * NUM_RAY_TYPES;
            instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            instance.traversableHandle = m_objectIdxToBlasHandleMap[objectId];
            m_instances.push_back(instance);
            m_instanceIds.insert(instanceId);
        }
    }
}

void OptixRenderer::createBlasForEntities()
{
    auto &scene = Scene::Get();

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

            m_geometries.push_back(geometry);
        }
    }
}

void OptixRenderer::createOptixInstanceForEntity()
{
    auto &scene = Scene::Get();

    // Entities - add them to instances during reload
    for (size_t entityIndex = 0; entityIndex < scene.getEntityCount(); ++entityIndex)
    {
        unsigned int geometryIndex = getEntityGeometryIndex(entityIndex);
        Entity *entity = scene.getEntity(entityIndex);
        if (entity)
        {
            OptixInstance instance = {};
            float transformMatrix[12];
            entity->getTransform().getTransformMatrix(transformMatrix);
            memcpy(instance.transform, transformMatrix, sizeof(float) * 12);

            // Use consistent entity instance ID
            instance.instanceId = EntityConstants::ENTITY_INSTANCE_ID_OFFSET + static_cast<unsigned int>(entityIndex);
            instance.visibilityMask = 255;

            // Calculate SBT offset using cached base + entity-specific offset
            instance.sbtOffset = geometryIndex * NUM_RAY_TYPES;
            instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            instance.traversableHandle = m_geometries[geometryIndex].gas;
            m_instances.push_back(instance);
        }
    }
}

void OptixRenderer::updateInstancedObjectInstance(unsigned int instanceId, unsigned int objectId)
{
    Scene &scene = Scene::Get();
    bool hasInstance = m_instanceIds.count(instanceId);

    if (hasInstance)
    {
        // Remove existing instance - object was deleted from scene
        m_instanceIds.erase(instanceId);

        // Find and remove the corresponding OptixInstance from the vector
        int idxToRemove = -1;
        for (int j = 0; j < m_instances.size(); ++j)
        {
            if (m_instances[j].instanceId == instanceId)
            {
                idxToRemove = j;
                break;
            }
        }
        assert(idxToRemove != -1);
        m_instances.erase(m_instances.begin() + idxToRemove);
    }
    else
    {
        // Create new instance - object was added to scene

        // BLAS should already exist for all instanced objects (created at initialization)
        assert(!(m_objectIdxToBlasHandleMap.find(objectId) == m_objectIdxToBlasHandleMap.end()));

        m_instanceIds.insert(instanceId);

        // Create OptixInstance with transformation matrix from scene
        OptixInstance instance = {};
        memcpy(instance.transform, scene.instanceTransformMatrices[instanceId].data(), sizeof(float) * 12);
        instance.instanceId = instanceId;
        instance.visibilityMask = 255;                                            // Visible to all ray types
        instance.sbtOffset = getInstancedGeometryIndex(objectId) * NUM_RAY_TYPES; // SBT offset for material/shader binding
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = m_objectIdxToBlasHandleMap[objectId]; // Link to geometry BLAS

        m_instances.push_back(instance);
    }
}

void OptixRenderer::update()
{
    auto &scene = Scene::Get();

    if (!scene.needSceneUpdate)
    {
        return;
    }

    if (scene.needSceneReloadUpdate)
    {
        // We will recreate all optix instances
        m_instances.resize(scene.uninstancedGeometryCount);
        m_instanceIds.clear();

        // Uninstanced
        m_instances.clear(); // Clear all instances and rebuild them
        unsigned int instanceIndex = 0;

        createGasAndOptixInstanceForUninstancedObject();

        // Update SBT records with new geometry pointers for uninstanced objects
        for (unsigned int chunkIndex = 0; chunkIndex < scene.numChunks; ++chunkIndex)
        {
            for (unsigned int objectId = Assets::BlockManager::Get().getUninstancedObjectIdBegin(); objectId < Assets::BlockManager::Get().getUninstancedObjectIdEnd(); ++objectId)
            {
                unsigned int geometryIndex = getUninstancedGeometryIndex(chunkIndex, objectId);
                if (geometryIndex < m_geometries.size())
                {
                    updateSbtRecordGeometryData(geometryIndex, m_geometries[geometryIndex]);
                }
            }
        }

        // Upload SBT
        CUDA_CHECK(cudaMemcpyAsync((void *)m_d_sbtRecordGeometryInstanceData, m_sbtRecordGeometryInstanceData.data(), sizeof(SbtRecordGeometryInstanceData) * m_sbtRecordGeometryInstanceData.size(), cudaMemcpyHostToDevice, Backend::Get().getCudaStream()));

        // Instanced
        createOptixInstanceForInstancedObject();

        // Entities
        createOptixInstanceForEntity();
    }
    else
    {
        for (int i = 0; i < scene.sceneUpdateObjectId.size(); ++i)
        {
            auto objectId = scene.sceneUpdateObjectId[i];
            unsigned int blockId = Assets::BlockManager::Get().objectIdToBlockId(objectId);

            // Uninstanced
            if (Assets::BlockManager::Get().isUninstancedBlockType(blockId))
            {
                unsigned int chunkIndex = scene.sceneUpdateChunkId[i];
                updateGasAndOptixInstanceForUninstancedObject(chunkIndex, objectId);
                CUDA_CHECK(cudaMemcpyAsync((void *)m_d_sbtRecordGeometryInstanceData, m_sbtRecordGeometryInstanceData.data(), sizeof(SbtRecordGeometryInstanceData) * m_sbtRecordGeometryInstanceData.size(), cudaMemcpyHostToDevice, Backend::Get().getCudaStream()));
            }
            // Instanced
            else if (Assets::BlockManager::Get().isInstancedBlockType(blockId))
            {
                auto instanceId = scene.sceneUpdateInstanceId[i];
                updateInstancedObjectInstance(instanceId, objectId);
            }
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

    // Initialize asset management system
    {
        // Initialize the new asset management system
        Assets::AssetRegistry::Get().loadFromYAML();

        // Initialize material manager (TextureManager is already initialized in main.cpp)
        Assets::MaterialManager::Get().initialize();

        // ModelManager is now initialized earlier in main to avoid duplication
        // Assets::ModelManager::Get().initialize();

        // Materials are now managed by MaterialManager
        // No need to create them here - they're already in GPU memory
    }

    assert((sizeof(SbtRecordHeader) % OPTIX_SBT_RECORD_ALIGNMENT) == 0);
    assert((sizeof(SbtRecordGeometryInstanceData) % OPTIX_SBT_RECORD_ALIGNMENT) == 0);

    createGasAndOptixInstanceForUninstancedObject();
    createBlasForInstancedObjects();
    createOptixInstanceForInstancedObject();
    createBlasForEntities();
    createOptixInstanceForEntity();
    buildInstanceAccelerationStructure(Backend::Get().getCudaStream(), 0);

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
        std::string ptxRaygeneration = ReadFileToString("ptx/RayGen.ptx");
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
        std::string ptxMiss = ReadFileToString("ptx/Miss.ptx");
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
        std::string ptxClosesthit = ReadFileToString("ptx/ClosestHit.ptx");
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
        // std::string ptxBsdf = ReadFileToString("ptx/Bsdf.ptx");
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

        unsigned int totalInstances = scene.uninstancedGeometryCount * scene.numChunks + scene.instancedGeometryCount + static_cast<unsigned int>(scene.getEntityCount());
        m_sbtRecordGeometryInstanceData.resize(totalInstances * NUM_RAY_TYPES);

        // Setup SBT records for chunk-based uninstanced geometry
        for (unsigned int chunkIndex = 0; chunkIndex < scene.numChunks; ++chunkIndex)
        {
            for (unsigned int objectId = Assets::BlockManager::Get().getUninstancedObjectIdBegin(); objectId < Assets::BlockManager::Get().getUninstancedObjectIdEnd(); ++objectId)
            {
                unsigned int geometryIndex = getUninstancedGeometryIndex(chunkIndex, objectId);
                unsigned int sbtBaseIndex = geometryIndex * NUM_RAY_TYPES;

                // Get material index from MaterialManager for proper mapping
                unsigned int materialIndex = Assets::MaterialManager::Get().getMaterialIndexForObjectId(objectId);
                initializeSbtRecord(sbtBaseIndex, geometryIndex, materialIndex);
            }
        }

        // Setup SBT records for instanced geometry
        for (unsigned int objectId = Assets::BlockManager::Get().getInstancedObjectIdBegin(); objectId < Assets::BlockManager::Get().getInstancedObjectIdEnd(); ++objectId)
        {
            // Calculate the correct geometry index for instanced objects
            // Instanced geometries are stored after all chunk-based geometries
            unsigned int geometryIndex = getInstancedGeometryIndex(objectId);
            unsigned int sbtBaseIndex = geometryIndex * NUM_RAY_TYPES;

            // Get material index from MaterialManager for proper mapping
            unsigned int materialIndex = Assets::MaterialManager::Get().getMaterialIndexForObjectId(objectId);
            int blockType = Assets::BlockManager::Get().objectIdToBlockId(objectId);

            initializeSbtRecord(sbtBaseIndex, geometryIndex, materialIndex);
        }

        // Setup SBT records for entities
        for (size_t entityIndex = 0; entityIndex < scene.getEntityCount(); ++entityIndex)
        {
            Entity *entity = scene.getEntity(entityIndex);
            if (entity)
            {
                // Calculate the correct geometry index for entities
                // Entities are stored after all chunk-based and instanced geometries
                unsigned int geometryIndex = getEntityGeometryIndex(entityIndex);
                unsigned int sbtBaseIndex = geometryIndex * NUM_RAY_TYPES;

                // Get material index from MaterialManager for this entity type
                unsigned int entityMaterialIndex = Assets::MaterialManager::Get().getMaterialIndexForEntity(static_cast<unsigned int>(entity->getType()));
                initializeSbtRecord(sbtBaseIndex, geometryIndex, entityMaterialIndex);
            }
        }

        CUDA_CHECK(cudaMalloc((void **)&m_d_sbtRecordGeometryInstanceData, sizeof(SbtRecordGeometryInstanceData) * totalInstances * NUM_RAY_TYPES));
        CUDA_CHECK(cudaMemcpyAsync((void *)m_d_sbtRecordGeometryInstanceData, m_sbtRecordGeometryInstanceData.data(), sizeof(SbtRecordGeometryInstanceData) * totalInstances * NUM_RAY_TYPES, cudaMemcpyHostToDevice, Backend::Get().getCudaStream()));
    }

    // Setup the OptixShaderBindingTable
    {
        m_sbt.raygenRecord = m_d_sbtRecordRaygeneration;

        m_sbt.exceptionRecord = 0;

        m_sbt.missRecordBase = m_d_sbtRecordMiss;
        m_sbt.missRecordStrideInBytes = (unsigned int)sizeof(SbtRecordHeader);
        m_sbt.missRecordCount = NUM_RAY_TYPES;

        m_sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(m_d_sbtRecordGeometryInstanceData);
        m_sbt.hitgroupRecordStrideInBytes = (unsigned int)sizeof(SbtRecordGeometryInstanceData);
        m_sbt.hitgroupRecordCount = m_sbtRecordGeometryInstanceData.size();

        // m_sbt.callablesRecordBase = m_d_sbtRecordCallables;
        // m_sbt.callablesRecordStrideInBytes = (unsigned int)sizeof(SbtRecordHeader);
        // m_sbt.callablesRecordCount = (unsigned int)sbtRecordCallables.size();
    }

    // Setup "sysParam" data.
    {
        // Materials are now managed by MaterialManager
        m_systemParameter.materialParameters = Assets::MaterialManager::Get().getGPUMaterialsPointer();

        CUDA_CHECK(cudaMalloc((void **)&m_d_systemParameter, sizeof(SystemParameter)));
        CUDA_CHECK(cudaMemcpyAsync((void *)m_d_systemParameter, &m_systemParameter, sizeof(SystemParameter), cudaMemcpyHostToDevice, Backend::Get().getCudaStream()));
    }

    // Destroy modules
    for (auto &module : moduleList)
    {
        OPTIX_CHECK(m_api.optixModuleDestroy(module));
    }
}

// Helper function to build Instance Acceleration Structure (IAS)
void OptixRenderer::buildInstanceAccelerationStructure(CUstream cudaStream, int targetIasIndex)
{
    // Build Instance Acceleration Structure for the given target index

    // Check if we have instances to build
    if (m_instances.empty())
    {
        // No instances to build
        return;
    }

    // Upload instances to GPU
    CUdeviceptr d_instances = 0;
    const size_t instancesSizeInBytes = sizeof(OptixInstance) * m_instances.size();
    // Allocate memory for instances

    cudaError_t cudaErr = cudaMalloc((void **)&d_instances, instancesSizeInBytes);
    if (cudaErr != cudaSuccess)
    {
        std::cerr << "IAS BUILD ERROR: Failed to allocate d_instances: " << cudaGetErrorString(cudaErr) << std::endl;
        return;
    }
    // Successfully allocated instances buffer

    if (d_instances == 0)
    {
        std::cerr << "IAS BUILD ERROR: d_instances is NULL after allocation!" << std::endl;
        return;
    }

    cudaErr = cudaMemcpyAsync((void *)d_instances, m_instances.data(), instancesSizeInBytes,
                              cudaMemcpyHostToDevice, cudaStream);
    if (cudaErr != cudaSuccess)
    {
        std::cerr << "IAS BUILD ERROR: Failed to copy instances: " << cudaGetErrorString(cudaErr) << std::endl;
        cudaFree((void *)d_instances);
        return;
    }

    // Sync to ensure copy is complete
    cudaDeviceSynchronize();
    // Instances copied to GPU

    // Setup build input
    OptixBuildInput instanceInput = {};
    instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instanceInput.instanceArray.instances = d_instances;
    instanceInput.instanceArray.numInstances = (unsigned int)m_instances.size();

    // Setup build options
    OptixAccelBuildOptions accelBuildOptions = {};
    accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Compute memory requirements
    OptixAccelBufferSizes iasBufferSizes = {};
    OptixResult optixRes = m_api.optixAccelComputeMemoryUsage(m_context, &accelBuildOptions, &instanceInput, 1, &iasBufferSizes);
    if (optixRes != OPTIX_SUCCESS)
    {
        std::cerr << "IAS BUILD ERROR: optixAccelComputeMemoryUsage failed with " << optixRes << std::endl;
        cudaFree((void *)d_instances);
        return;
    }

    // Computed memory requirements for IAS

    // Only reallocate if buffer doesn't exist or required size is larger than current
    if (m_d_ias[targetIasIndex] == 0 || iasBufferSizes.outputSizeInBytes > m_iasBufferSizes[targetIasIndex])
    {
        // Need to allocate/reallocate IAS buffer

        // Free existing buffer if needed
        if (m_d_ias[targetIasIndex] != 0)
        {
            // Free existing buffer
            CUDA_CHECK(cudaFree((void *)m_d_ias[targetIasIndex]));
        }

        // Allocate new buffer
        // Allocate new IAS buffer
        cudaErr = cudaMalloc((void **)&m_d_ias[targetIasIndex], iasBufferSizes.outputSizeInBytes);
        if (cudaErr != cudaSuccess)
        {
            std::cerr << "IAS BUILD ERROR: Failed to allocate IAS buffer: " << cudaGetErrorString(cudaErr) << std::endl;
            cudaFree((void *)d_instances);
            return;
        }
        m_iasBufferSizes[targetIasIndex] = iasBufferSizes.outputSizeInBytes;
        // IAS buffer allocated successfully
    }

    if (m_d_ias[targetIasIndex] == 0)
    {
        std::cerr << "IAS BUILD ERROR: m_d_ias[" << targetIasIndex << "] is NULL!" << std::endl;
        cudaFree((void *)d_instances);
        return;
    }

    CUdeviceptr d_tmp = 0;
    // Allocate temporary buffer for build
    cudaErr = cudaMalloc((void **)&d_tmp, iasBufferSizes.tempSizeInBytes);
    if (cudaErr != cudaSuccess)
    {
        std::cerr << "IAS BUILD ERROR: Failed to allocate temp buffer: " << cudaGetErrorString(cudaErr) << std::endl;
        cudaFree((void *)d_instances);
        return;
    }
    // Temp buffer allocated

    if (d_tmp == 0)
    {
        std::cerr << "IAS BUILD ERROR: d_tmp is NULL after allocation!" << std::endl;
        cudaFree((void *)d_instances);
        return;
    }

    // Add sync before build
    cudaDeviceSynchronize();

    // Build the acceleration structure
    // Build the IAS

    if (m_d_ias[targetIasIndex] == 0)
    {
        std::cerr << "ERROR: outputBuffer is 0" << std::endl;
    }

    optixRes = m_api.optixAccelBuild(m_context, cudaStream,
                                     &accelBuildOptions, &instanceInput, 1,
                                     d_tmp, iasBufferSizes.tempSizeInBytes,
                                     m_d_ias[targetIasIndex], iasBufferSizes.outputSizeInBytes,
                                     &m_systemParameter.topObject, nullptr, 0);

    if (optixRes != OPTIX_SUCCESS)
    {
        std::cerr << "IAS BUILD ERROR: optixAccelBuild failed with " << optixRes << std::endl;
    }
    else
    {
        // IAS build succeeded
    }

    // Synchronize and cleanup
    cudaErr = cudaStreamSynchronize(cudaStream);
    if (cudaErr != cudaSuccess)
    {
        std::cerr << "IAS BUILD ERROR: cudaStreamSynchronize failed: " << cudaGetErrorString(cudaErr) << std::endl;
    }

    cudaErr = cudaFree((void *)d_tmp);
    if (cudaErr != cudaSuccess)
    {
        std::cerr << "IAS BUILD ERROR: Failed to free temp buffer: " << cudaGetErrorString(cudaErr) << std::endl;
    }

    cudaErr = cudaFree((void *)d_instances);
    if (cudaErr != cudaSuccess)
    {
        std::cerr << "IAS BUILD ERROR: Failed to free instances buffer: " << cudaGetErrorString(cudaErr) << std::endl;
    }
}