#include "core/Backend.h"
#include "util/DebugUtils.h"
#include "OptixRenderer.h"
#include "core/Scene.h"
#include "core/BufferManager.h"
#include "util/TextureUtils.h"
#include "core/GlobalSettings.h"
#include "sky/Sky.h"
#include "core/RenderCamera.h"

#include "util/BufferUtils.h"

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
    CUDA_CHECK(cudaFree((void *)m_d_ias));

    CUDA_CHECK(cudaFree((void *)m_d_sbtRecordRaygeneration));
    CUDA_CHECK(cudaFree((void *)m_d_sbtRecordMiss));
    CUDA_CHECK(cudaFree((void *)m_d_sbtRecordCallables));

    CUDA_CHECK(cudaFree((void *)m_d_sbtRecordGeometryInstanceData));

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

    RenderCamera::Get().camera.update();

    constexpr int samplePerIteration = 1;

    m_systemParameter.camera = RenderCamera::Get().camera;
    m_systemParameter.prevCamera = RenderCamera::Get().historyCamera;
    m_systemParameter.accumulationCounter = backend.getAccumulationCounter();
    m_systemParameter.samplePerIteration = samplePerIteration;
    m_systemParameter.timeInSecond = backend.getTimer().getTimeInSecond();

    const auto &skyModel = SkyModel::Get();
    m_systemParameter.sunDir = skyModel.getSunDir();
    m_systemParameter.accumulatedSkyLuminance = skyModel.getAccumulatedSkyLuminance();
    m_systemParameter.accumulatedSunLuminance = skyModel.getAccumulatedSunLuminance();

    m_systemParameter.lights = scene.m_lights;
    m_systemParameter.lightAliasTable = scene.d_lightAliasTable;
    m_systemParameter.instanceLightMapping = scene.d_instanceLightMapping;
    m_systemParameter.numInstancedLightMesh = scene.numInstancedLightMesh;

    BufferSetFloat4(bufferManager.GetBufferDim(UiBuffer), bufferManager.GetBuffer2D(UiBuffer), Float4(0.0f));

    for (int sampleIndex = 0; sampleIndex < samplePerIteration; ++sampleIndex)
    {
        m_systemParameter.sampleIndex = sampleIndex;
        CUDA_CHECK(cudaMemcpyAsync((void *)m_d_systemParameter, &m_systemParameter, sizeof(SystemParameter), cudaMemcpyHostToDevice, backend.getCudaStream()));
        OPTIX_CHECK(m_api.optixLaunch(m_pipeline, backend.getCudaStream(), (CUdeviceptr)m_d_systemParameter, sizeof(SystemParameter), &m_sbt, m_width, m_height, 1));
    }

    CUDA_CHECK(cudaStreamSynchronize(backend.getCudaStream()));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

    BufferCopyFloat4(bufferManager.GetBufferDim(GeoNormalThinfilmBuffer), bufferManager.GetBuffer2D(GeoNormalThinfilmBuffer), bufferManager.GetBuffer2D(PrevGeoNormalThinfilmBuffer));
    BufferCopyFloat4(bufferManager.GetBufferDim(AlbedoBuffer), bufferManager.GetBuffer2D(AlbedoBuffer), bufferManager.GetBuffer2D(PrevAlbedoBuffer));

    RenderCamera::Get().historyCamera = RenderCamera::Get().camera;
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

        // Uninstanced
        for (int objectId = 0; objectId < scene.uninstancedGeometryCount; ++objectId)
        {
            auto blockId = objectId + 1;

            GeometryData &geometry = m_geometries[objectId];
            CUDA_CHECK(cudaFree((void *)geometry.gas));
            OptixTraversableHandle blasHandle = Scene::CreateGeometry(m_api, m_context, Backend::Get().getCudaStream(), geometry, scene.m_geometryAttibutes[objectId], scene.m_geometryIndices[objectId], scene.m_geometryAttibuteSize[objectId], scene.m_geometryIndicesSize[objectId]);

            OptixInstance &instance = m_instances[objectId];
            const float transformMatrix[12] =
                {
                    1.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 1.0f, 0.0f};
            memcpy(instance.transform, transformMatrix, sizeof(float) * 12);
            instance.instanceId = objectId;
            instance.visibilityMask = (blockId == BlockTypeWater) ? 1 : 255;
            instance.sbtOffset = objectId * numTypesOfRays;
            instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            instance.traversableHandle = blasHandle;

            // Shader binding table record hit group geometry
            m_sbtRecordGeometryInstanceData[objectId * 2].data.indices = (Int3 *)m_geometries[objectId].indices;
            m_sbtRecordGeometryInstanceData[objectId * 2].data.attributes = (VertexAttributes *)m_geometries[objectId].attributes;
            m_sbtRecordGeometryInstanceData[objectId * 2 + 1].data.indices = (Int3 *)m_geometries[objectId].indices;
            m_sbtRecordGeometryInstanceData[objectId * 2 + 1].data.attributes = (VertexAttributes *)m_geometries[objectId].attributes;
        }
        // Upload SBT
        CUDA_CHECK(cudaMemcpyAsync((void *)m_d_sbtRecordGeometryInstanceData, m_sbtRecordGeometryInstanceData.data(), sizeof(SbtRecordGeometryInstanceData) * numTypesOfRays * numObjects, cudaMemcpyHostToDevice, Backend::Get().getCudaStream()));

        // Instanced
        for (int objectId = scene.uninstancedGeometryCount; objectId < scene.uninstancedGeometryCount + scene.instancedGeometryCount; ++objectId)
        {
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
    else
    {
        for (int i = 0; i < scene.sceneUpdateObjectId.size(); ++i)
        {
            auto objectId = scene.sceneUpdateObjectId[i];
            auto blockId = objectId + 1;

            // Uninstanced
            if (blockId < BlockTypeWater)
            {
                GeometryData &geometry = m_geometries[objectId];
                CUDA_CHECK(cudaFree((void *)geometry.gas));
                OptixTraversableHandle blasHandle = Scene::CreateGeometry(m_api, m_context, Backend::Get().getCudaStream(), geometry, scene.m_geometryAttibutes[objectId], scene.m_geometryIndices[objectId], scene.m_geometryAttibuteSize[objectId], scene.m_geometryIndicesSize[objectId]);

                OptixInstance &instance = m_instances[objectId];
                const float transformMatrix[12] =
                    {
                        1.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 1.0f, 0.0f};
                memcpy(instance.transform, transformMatrix, sizeof(float) * 12);
                instance.instanceId = objectId;
                instance.visibilityMask = 255;
                instance.sbtOffset = objectId * numTypesOfRays;
                instance.flags = OPTIX_INSTANCE_FLAG_NONE;
                instance.traversableHandle = blasHandle;

                // Shader binding table record hit group geometry
                m_sbtRecordGeometryInstanceData[objectId * 2].data.indices = (Int3 *)m_geometries[objectId].indices;
                m_sbtRecordGeometryInstanceData[objectId * 2].data.attributes = (VertexAttributes *)m_geometries[objectId].attributes;
                m_sbtRecordGeometryInstanceData[objectId * 2 + 1].data.indices = (Int3 *)m_geometries[objectId].indices;
                m_sbtRecordGeometryInstanceData[objectId * 2 + 1].data.attributes = (VertexAttributes *)m_geometries[objectId].attributes;
            }
            // Instanced
            else if (blockId > BlockTypeWater)
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
        // Upload SBT
        CUDA_CHECK(cudaMemcpyAsync((void *)m_d_sbtRecordGeometryInstanceData, m_sbtRecordGeometryInstanceData.data(), sizeof(SbtRecordGeometryInstanceData) * numTypesOfRays * numObjects, cudaMemcpyHostToDevice, Backend::Get().getCudaStream()));
    }

    // Rebuild BVH
    {
        CUDA_CHECK(cudaStreamSynchronize(Backend::Get().getCudaStream()));
        CUDA_CHECK(cudaFree((void *)m_d_ias));

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

        CUDA_CHECK(cudaMalloc((void **)&m_d_ias, iasBufferSizes.outputSizeInBytes));

        CUdeviceptr d_tmp;

        CUDA_CHECK(cudaMalloc((void **)&d_tmp, iasBufferSizes.tempSizeInBytes));

        auto &backend = Backend::Get();
        OPTIX_CHECK(m_api.optixAccelBuild(m_context, backend.getCudaStream(),
                                          &accelBuildOptions, &instanceInput, 1,
                                          d_tmp, iasBufferSizes.tempSizeInBytes,
                                          m_d_ias, iasBufferSizes.outputSizeInBytes,
                                          &m_systemParameter.topObject, nullptr, 0));

        CUDA_CHECK(cudaStreamSynchronize(backend.getCudaStream()));

        CUDA_CHECK(cudaFree((void *)d_tmp));
        CUDA_CHECK(cudaFree((void *)d_instances)); // Don't need the instances anymore.
    }

    scene.sceneUpdateObjectId.clear();
    scene.sceneUpdateInstanceId.clear();

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
        m_systemParameter.outputBuffer = 0;
        m_systemParameter.materialParameters = nullptr;

        const auto &bufferManager = BufferManager::Get();
        const auto &skyModel = SkyModel::Get();
        m_systemParameter.outputBuffer = bufferManager.GetBuffer2D(IlluminationBuffer);
        m_systemParameter.outNormal = bufferManager.GetBuffer2D(NormalRoughnessBuffer);
        m_systemParameter.outDepth = bufferManager.GetBuffer2D(DepthBuffer);
        m_systemParameter.outAlbedo = bufferManager.GetBuffer2D(AlbedoBuffer);
        m_systemParameter.outMaterial = bufferManager.GetBuffer2D(MaterialBuffer);
        m_systemParameter.outMotionVector = bufferManager.GetBuffer2D(MotionVectorBuffer);
        m_systemParameter.outUiBuffer = bufferManager.GetBuffer2D(UiBuffer);

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

        m_systemParameter.prevDepthBuffer = bufferManager.GetBuffer2D(PrevDepthBuffer);
        m_systemParameter.prevNormalRoughnessBuffer = bufferManager.GetBuffer2D(PrevNormalRoughnessBuffer);
        m_systemParameter.prevGeoNormalThinfilmBuffer = bufferManager.GetBuffer2D(PrevGeoNormalThinfilmBuffer);
        m_systemParameter.prevAlbedoBuffer = bufferManager.GetBuffer2D(PrevAlbedoBuffer);

        m_systemParameter.outGeoNormalThinfilmBuffer = bufferManager.GetBuffer2D(GeoNormalThinfilmBuffer);

        m_systemParameter.neighborOffsetBuffer = bufferManager.neighborOffsetBuffer;

        m_d_ias = 0;

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

    // Create materials
    {
        // Setup GUI material parameters, one for each of the implemented BSDFs.
        MaterialParameter parameters{};

        TextureManager &textureManager = TextureManager::Get();

        // The order in this array matches the instance ID in the root IAS!
        for (const auto &textureFile : GetTextureFiles())
        {
            parameters.indexBSDF = INDEX_BSDF_MICROFACET_REFLECTION;
            parameters.albedo = Float3(1.0f);
            parameters.uvScale = 2.5f;
            parameters.textureAlbedo = textureManager.GetTexture("data/" + textureFile + "_albedo.png");
            parameters.textureNormal = textureManager.GetTexture("data/" + textureFile + "_normal.png");
            parameters.textureRoughness = textureManager.GetTexture("data/" + textureFile + "_rough.png");
            parameters.absorption = Float3(-logf(1.0f), -logf(1.0f), -logf(1.0f)) * 1.0f;
            parameters.ior = 1.4f;
            parameters.flags = 0;
            m_materialParameters.push_back(parameters);
        }

        // Water material
        parameters.indexBSDF = INDEX_BSDF_SPECULAR_REFLECTION_TRANSMISSION;
        parameters.albedo = Float3(1.0f, 1.0f, 1.0f);
        parameters.uvScale = 4.0f;
        parameters.textureAlbedo = 0;
        parameters.textureNormal = textureManager.GetTexture("data/water1.jpg");
        parameters.textureRoughness = 0;
        parameters.absorption = Float3(-logf(0.9f), -logf(0.95f), -logf(1.0f)) * 0.5f;
        parameters.ior = 1.33f;
        parameters.flags = 1;
        m_materialParameters.push_back(parameters);

        // Test material
        parameters.indexBSDF = INDEX_BSDF_DIFFUSE_REFLECTION_TRANSMISSION_THINFILM;
        parameters.albedo = Float3(1.0f);
        parameters.uvScale = 1.0f;
        parameters.textureAlbedo = textureManager.GetTexture("data/GreenLeaf10_4K_back_albedo.png");
        parameters.textureNormal = textureManager.GetTexture("data/GreenLeaf10_4K_back_normal.png");
        parameters.textureRoughness = 0;
        parameters.absorption = Float3(122.0f / 255.0f, 138.0f / 255.0f, 109.0f / 255.0f);
        parameters.ior = 1.33f;
        parameters.flags = 2;
        m_materialParameters.push_back(parameters);

        // Leaf material
        parameters.indexBSDF = INDEX_BSDF_DIFFUSE_REFLECTION_TRANSMISSION_THINFILM;
        parameters.albedo = Float3(1.0f);
        parameters.uvScale = 1.0f;
        parameters.textureAlbedo = textureManager.GetTexture("data/GreenLeaf10_4K_back_albedo.png");
        parameters.textureNormal = textureManager.GetTexture("data/GreenLeaf10_4K_back_normal.png");
        parameters.textureRoughness = 0;
        parameters.absorption = Float3(122.0f / 255.0f, 138.0f / 255.0f, 109.0f / 255.0f);
        parameters.ior = 1.33f;
        parameters.flags = 2;
        m_materialParameters.push_back(parameters);

        // Lantern base
        parameters.indexBSDF = INDEX_BSDF_MICROFACET_REFLECTION_METAL;
        parameters.albedo = Float3(1.0f);
        parameters.uvScale = 1.0f;
        std::string textureFile = "beaten-up-metal1";
        parameters.textureAlbedo = textureManager.GetTexture("data/" + textureFile + "_albedo.png");
        parameters.textureNormal = textureManager.GetTexture("data/" + textureFile + "_normal.png");
        parameters.textureRoughness = textureManager.GetTexture("data/" + textureFile + "_rough.png");
        parameters.textureMetallic = textureManager.GetTexture("data/" + textureFile + "_metal.png");
        parameters.absorption = Float3(-logf(1.0f), -logf(1.0f), -logf(1.0f)) * 1.0f;
        parameters.ior = 1.5f;
        parameters.flags = 0;
        m_materialParameters.push_back(parameters);

        // Lantern light
        parameters.indexBSDF = INDEX_BSDF_EMISSIVE;
        parameters.albedo = GetEmissiveRadiance(BlockTypeTestLight);
        parameters.uvScale = 1.0f;
        parameters.textureAlbedo = 0;
        parameters.textureNormal = 0;
        parameters.textureRoughness = 0;
        parameters.absorption = Float3(-logf(1.0f), -logf(1.0f), -logf(1.0f)) * 1.0f;
        parameters.ior = 1.5f;
        parameters.flags = 0;
        m_materialParameters.push_back(parameters);

        //------------------------ Testing below ------------------------

        // Test thinwall material
        parameters.indexBSDF = INDEX_BSDF_DIFFUSE_REFLECTION_TRANSMISSION_THINFILM;
        parameters.albedo = Float3(1.0f, 0.2f, 0.2f);
        parameters.uvScale = 1.0f;
        parameters.textureAlbedo = 0;
        parameters.textureNormal = 0;
        parameters.textureRoughness = 0;
        parameters.absorption = Float3(1.0f, 0.2f, 0.2f);
        parameters.ior = 1.5f;
        parameters.flags = 0;
        m_materialParameters.push_back(parameters);

        // Emissive material
        parameters.indexBSDF = INDEX_BSDF_EMISSIVE;
        parameters.albedo = Float3(0.2f, 1.0f, 0.2f);
        parameters.uvScale = 1.0f;
        parameters.textureAlbedo = 0;
        parameters.textureNormal = 0;
        parameters.textureRoughness = 0;
        parameters.absorption = Float3(-logf(1.0f), -logf(1.0f), -logf(1.0f)) * 1.0f;
        parameters.ior = 1.5f;
        parameters.flags = 0;
        m_materialParameters.push_back(parameters);

        // Lambert material
        parameters.indexBSDF = INDEX_BSDF_DIFFUSE_REFLECTION;
        parameters.albedo = Float3(1.0f, 0.2f, 0.2f);
        parameters.uvScale = 1.0f;
        parameters.textureAlbedo = 0;
        parameters.textureNormal = 0;
        parameters.textureRoughness = 0;
        parameters.absorption = Float3(-logf(1.0f), -logf(1.0f), -logf(1.0f)) * 1.0f;
        parameters.ior = 1.5f;
        parameters.flags = 0;
        m_materialParameters.push_back(parameters);

        // Mirror material
        parameters.indexBSDF = INDEX_BSDF_SPECULAR_REFLECTION;
        parameters.albedo = Float3(1.0f, 1.0f, 1.0f);
        parameters.uvScale = 1.0f;
        parameters.textureAlbedo = 0;
        parameters.textureNormal = 0;
        parameters.textureRoughness = 0;
        parameters.absorption = Float3(-logf(1.0f), -logf(1.0f), -logf(1.0f)) * 1.0f;
        parameters.ior = 1.33f;
        parameters.flags = 0;
        m_materialParameters.push_back(parameters);

        // Black BSDF for the light. This last material will not be shown inside the GUI!
        parameters.indexBSDF = INDEX_BSDF_SPECULAR_REFLECTION;
        parameters.albedo = Float3(0.0f, 1.0f, 1.0f);
        parameters.uvScale = 1.0f;
        parameters.textureAlbedo = 0;
        parameters.textureNormal = 0;
        parameters.textureRoughness = 0;
        parameters.absorption = Float3(-logf(1.0f), -logf(1.0f), -logf(1.0f)) * 1.0f;
        parameters.ior = 1.0f;
        parameters.flags = 0;
        m_materialParameters.push_back(parameters);
    }

    assert((sizeof(SbtRecordHeader) % OPTIX_SBT_RECORD_ALIGNMENT) == 0);
    assert((sizeof(SbtRecordGeometryInstanceData) % OPTIX_SBT_RECORD_ALIGNMENT) == 0);

    // Create uninstanced geometry BLAS and track instances
    for (int objectId = 0; objectId < scene.uninstancedGeometryCount; ++objectId)
    {
        int blockId = objectId + 1;

        // Create BLAS for the geometry
        GeometryData geometry = {};
        OptixTraversableHandle blasHandle = Scene::CreateGeometry(m_api, m_context, Backend::Get().getCudaStream(), geometry, scene.m_geometryAttibutes[objectId], scene.m_geometryIndices[objectId], scene.m_geometryAttibuteSize[objectId], scene.m_geometryIndicesSize[objectId]);
        m_geometries.push_back(geometry);

        // Create an instance for the geometry
        OptixInstance instance = {};
        const float transformMatrix[12] =
            {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f};
        memcpy(instance.transform, transformMatrix, sizeof(float) * 12);
        instance.instanceId = objectId;
        instance.visibilityMask = (blockId == BlockTypeWater) ? 1 : 255;
        instance.sbtOffset = objectId * numTypesOfRays;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = blasHandle;
        m_instances.push_back(instance);
    }

    // Create instanced geometry and track instances
    for (int objectId = scene.uninstancedGeometryCount; objectId < scene.uninstancedGeometryCount + scene.instancedGeometryCount; ++objectId)
    {
        // Create BLAS for the geometry
        GeometryData geometry = {};
        OptixTraversableHandle blasHandle = Scene::CreateGeometry(m_api, m_context, Backend::Get().getCudaStream(), geometry, scene.m_geometryAttibutes[objectId], scene.m_geometryIndices[objectId], scene.m_geometryAttibuteSize[objectId], scene.m_geometryIndicesSize[objectId]);
        m_geometries.push_back(geometry);
        objectIdxToBlasHandleMap[objectId] = blasHandle;

        for (int instanceId : scene.geometryInstanceIdMap[objectId])
        {
            OptixInstance instance = {};
            memcpy(instance.transform, scene.instanceTransformMatrices[instanceId].data(), sizeof(float) * 12);
            instance.instanceId = instanceId;
            instance.visibilityMask = 255;
            instance.sbtOffset = objectId * numTypesOfRays;
            instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            instance.traversableHandle = blasHandle;
            m_instances.push_back(instance);
            instanceIds.insert(instanceId);
        }
    }

    // Build BVH
    {
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

        CUDA_CHECK(cudaMalloc((void **)&m_d_ias, iasBufferSizes.outputSizeInBytes));

        CUdeviceptr d_tmp;

        CUDA_CHECK(cudaMalloc((void **)&d_tmp, iasBufferSizes.tempSizeInBytes));

        auto &backend = Backend::Get();
        OPTIX_CHECK(m_api.optixAccelBuild(m_context, backend.getCudaStream(),
                                          &accelBuildOptions, &instanceInput, 1,
                                          d_tmp, iasBufferSizes.tempSizeInBytes,
                                          m_d_ias, iasBufferSizes.outputSizeInBytes,
                                          &m_systemParameter.topObject, nullptr, 0));

        CUDA_CHECK(cudaStreamSynchronize(backend.getCudaStream()));

        CUDA_CHECK(cudaFree((void *)d_tmp));
        CUDA_CHECK(cudaFree((void *)d_instances)); // Don't need the instances anymore.
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
        std::string ptxBsdf = ReadPtx("ptx/Bsdf.ptx");
        OptixModule moduleBsdf;
        OPTIX_CHECK(m_api.optixModuleCreate(m_context, &moduleCompileOptions, &pipelineCompileOptions, ptxBsdf.c_str(), ptxBsdf.size(), nullptr, nullptr, &moduleBsdf));

        std::vector<OptixProgramGroupDesc> programGroupDescCallables;

        OptixProgramGroupDesc pgd = {};

        pgd.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        pgd.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

        pgd.callables.moduleDC = moduleBsdf;

        pgd.callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_specular_reflection"; // 0
        programGroupDescCallables.push_back(pgd);
        pgd.callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_specular_reflection_transmission"; // 1
        programGroupDescCallables.push_back(pgd);
        pgd.callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_diffuse_reflection"; // 2
        programGroupDescCallables.push_back(pgd);
        pgd.callables.entryFunctionNameDC = "__direct_callable__eval_bsdf_diffuse_reflection"; // 3
        programGroupDescCallables.push_back(pgd);
        pgd.callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_microfacet_reflection"; // 4
        programGroupDescCallables.push_back(pgd);
        pgd.callables.entryFunctionNameDC = "__direct_callable__eval_bsdf_microfacet_reflection"; // 5
        programGroupDescCallables.push_back(pgd);
        pgd.callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_diffuse_reflection_transmission_thinfilm"; // 6
        programGroupDescCallables.push_back(pgd);
        pgd.callables.entryFunctionNameDC = "__direct_callable__eval_bsdf_diffuse_reflection_transmission_thinfilm"; // 7
        programGroupDescCallables.push_back(pgd);
        pgd.callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_microfacet_reflection_metal"; // 8
        programGroupDescCallables.push_back(pgd);
        pgd.callables.entryFunctionNameDC = "__direct_callable__eval_bsdf_microfacet_reflection_metal"; // 9
        programGroupDescCallables.push_back(pgd);

        programGroupCallables.resize(programGroupDescCallables.size());
        OPTIX_CHECK(m_api.optixProgramGroupCreate(m_context, programGroupDescCallables.data(), programGroupDescCallables.size(), &programGroupOptions, nullptr, nullptr, programGroupCallables.data()));
        programGroups.insert(programGroups.end(), programGroupCallables.begin(), programGroupCallables.end());
        moduleList.push_back(moduleBsdf);
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

        m_sbtRecordGeometryInstanceData.resize(numObjects * numTypesOfRays);

        for (int objectId = 0; objectId < numObjects; ++objectId)
        {
            int idx = objectId * numTypesOfRays;

            memcpy(m_sbtRecordGeometryInstanceData[idx].header, m_sbtRecordHitRadiance.header, OPTIX_SBT_RECORD_HEADER_SIZE);

            m_sbtRecordGeometryInstanceData[idx].data.indices = (Int3 *)m_geometries[objectId].indices;
            m_sbtRecordGeometryInstanceData[idx].data.attributes = (VertexAttributes *)m_geometries[objectId].attributes;
            m_sbtRecordGeometryInstanceData[idx].data.materialIndex = objectId;

            memcpy(m_sbtRecordGeometryInstanceData[idx + 1].header, m_sbtRecordHitShadow.header, OPTIX_SBT_RECORD_HEADER_SIZE);

            m_sbtRecordGeometryInstanceData[idx + 1].data.indices = (Int3 *)m_geometries[objectId].indices;
            m_sbtRecordGeometryInstanceData[idx + 1].data.attributes = (VertexAttributes *)m_geometries[objectId].attributes;
            m_sbtRecordGeometryInstanceData[idx + 1].data.materialIndex = objectId;
        }

        CUDA_CHECK(cudaMalloc((void **)&m_d_sbtRecordGeometryInstanceData, sizeof(SbtRecordGeometryInstanceData) * numTypesOfRays * numObjects));
        CUDA_CHECK(cudaMemcpyAsync((void *)m_d_sbtRecordGeometryInstanceData, m_sbtRecordGeometryInstanceData.data(), sizeof(SbtRecordGeometryInstanceData) * numTypesOfRays * numObjects, cudaMemcpyHostToDevice, Backend::Get().getCudaStream()));
    }

    // Direct callables
    std::vector<SbtRecordHeader> sbtRecordCallables(programGroupCallables.size());
    {
        for (size_t i = 0; i < programGroupCallables.size(); ++i)
        {
            OPTIX_CHECK(m_api.optixSbtRecordPackHeader(programGroupCallables[i], &sbtRecordCallables[i]));
        }

        CUDA_CHECK(cudaMalloc((void **)&m_d_sbtRecordCallables, sizeof(SbtRecordHeader) * sbtRecordCallables.size()));
        CUDA_CHECK(cudaMemcpyAsync((void *)m_d_sbtRecordCallables, sbtRecordCallables.data(), sizeof(SbtRecordHeader) * sbtRecordCallables.size(), cudaMemcpyHostToDevice, Backend::Get().getCudaStream()));
    }

    // Setup the OptixShaderBindingTable
    {
        m_sbt.raygenRecord = m_d_sbtRecordRaygeneration;

        m_sbt.exceptionRecord = 0;

        m_sbt.missRecordBase = m_d_sbtRecordMiss;
        m_sbt.missRecordStrideInBytes = (unsigned int)sizeof(SbtRecordHeader);
        m_sbt.missRecordCount = numTypesOfRays;

        m_sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(m_d_sbtRecordGeometryInstanceData);
        m_sbt.hitgroupRecordStrideInBytes = (unsigned int)sizeof(SbtRecordGeometryInstanceData);
        m_sbt.hitgroupRecordCount = numObjects * numTypesOfRays;

        m_sbt.callablesRecordBase = m_d_sbtRecordCallables;
        m_sbt.callablesRecordStrideInBytes = (unsigned int)sizeof(SbtRecordHeader);
        m_sbt.callablesRecordCount = (unsigned int)sbtRecordCallables.size();
    }

    // Setup "sysParam" data.
    {
        CUDA_CHECK(cudaMalloc((void **)&m_systemParameter.materialParameters, sizeof(MaterialParameter) * m_materialParameters.size()));
        CUDA_CHECK(cudaMemcpyAsync((void *)m_systemParameter.materialParameters, m_materialParameters.data(), sizeof(MaterialParameter) * m_materialParameters.size(), cudaMemcpyHostToDevice, Backend::Get().getCudaStream()));

        CUDA_CHECK(cudaMalloc((void **)&m_d_systemParameter, sizeof(SystemParameter)));
        CUDA_CHECK(cudaMemcpyAsync((void *)m_d_systemParameter, &m_systemParameter, sizeof(SystemParameter), cudaMemcpyHostToDevice, Backend::Get().getCudaStream()));
    }

    // Destroy modules
    for (auto &module : moduleList)
    {
        OPTIX_CHECK(m_api.optixModuleDestroy(module));
    }
}