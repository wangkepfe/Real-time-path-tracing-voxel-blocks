#include "core/Backend.h"
#include "util/DebugUtils.h"
#include "OptixRenderer.h"
#include "core/Scene.h"
#include "core/BufferManager.h"
#include "util/TextureUtils.h"
#include "core/GlobalSettings.h"
#include "sky/Sky.h"
#include "core/RenderCamera.h"

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

namespace jazzfusion
{

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
        auto &backend = jazzfusion::Backend::Get();
        CUDA_CHECK(cudaStreamSynchronize(backend.getCudaStream()));

        CUDA_CHECK(cudaFree((void *)m_systemParameter.lightDefinitions));
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
        auto &backend = jazzfusion::Backend::Get();

        static int iterationIndex = 0;
        m_systemParameter.iterationIndex = iterationIndex++;

        CUDA_CHECK(cudaStreamSynchronize(backend.getCudaStream()));

        auto &camera = RenderCamera::Get().camera;

        camera.update();
        if (backend.getFrameNum() == 0)
        {
            m_systemParameter.historyCamera.Setup(camera);
        }

        constexpr int samplePerIteration = 1;

        m_systemParameter.camera = camera;
        m_systemParameter.noiseBlend = GlobalSettings::GetDenoisingParams().noiseBlend;
        m_systemParameter.accumulationCounter = backend.getAccumulationCounter();
        m_systemParameter.samplePerIteration = samplePerIteration;

        const auto &skyModel = SkyModel::Get();
        m_systemParameter.sunDir = skyModel.getSunDir();

        for (int sampleIndex = 0; sampleIndex < samplePerIteration; ++sampleIndex)
        {
            m_systemParameter.sampleIndex = sampleIndex;
            CUDA_CHECK(cudaMemcpy((void *)m_d_systemParameter, &m_systemParameter, sizeof(SystemParameter), cudaMemcpyHostToDevice));
            OPTIX_CHECK(m_api.optixLaunch(m_pipeline, backend.getCudaStream(), (CUdeviceptr)m_d_systemParameter, sizeof(SystemParameter), &m_sbt, m_width, m_height, 1));
        }

        CUDA_CHECK(cudaStreamSynchronize(backend.getCudaStream()));

        m_systemParameter.historyCamera.Setup(camera);
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

    void OptixRenderer::init()
    {
        {
            m_systemParameter.topObject = 0;
            m_systemParameter.outputBuffer = 0;
            m_systemParameter.lightDefinitions = nullptr;
            m_systemParameter.materialParameters = nullptr;

            const auto &bufferManager = BufferManager::Get();
            const auto &skyModel = SkyModel::Get();
            m_systemParameter.outputBuffer = bufferManager.GetBuffer2D(RenderColorBuffer);
            m_systemParameter.outNormal = bufferManager.GetBuffer2D(NormalBuffer);
            m_systemParameter.outDepth = bufferManager.GetBuffer2D(DepthBuffer);
            m_systemParameter.outAlbedo = bufferManager.GetBuffer2D(AlbedoBuffer);
            m_systemParameter.outMaterial = bufferManager.GetBuffer2D(MaterialBuffer);
            m_systemParameter.outMotionVector = bufferManager.GetBuffer2D(MotionVectorBuffer);

            m_systemParameter.randGen = d_randGen;

            m_systemParameter.skyBuffer = bufferManager.GetBuffer2D(SkyBuffer);
            m_systemParameter.sunBuffer = bufferManager.GetBuffer2D(SunBuffer);
            m_systemParameter.skyCdf = skyModel.getSkyCdf(); // This buffer is allocated in Backend::init()
            m_systemParameter.sunCdf = skyModel.getSunCdf();
            m_systemParameter.skyRes = skyModel.getSkyRes();
            m_systemParameter.sunRes = skyModel.getSunRes();

            m_systemParameter.iterationIndex = 0;
            m_systemParameter.sceneEpsilon = 500.0f * 1.0e-7f;
            m_systemParameter.numLights = 0;

            m_root = 0;
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

            auto &backend = jazzfusion::Backend::Get();
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
            // Lambert material for the floor.
            parameters.indexBSDF = INDEX_BSDF_DIFFUSE_REFLECTION; // Index for the direct callables.
            parameters.albedo = Float3(1.0f);
            parameters.uvScale = 1.0f;
            parameters.textureAlbedo = textureManager.GetTexture("data/TexturesCom_VinylChecker_1K_albedo.png");
            parameters.textureNormal = textureManager.GetTexture("data/TexturesCom_VinylChecker_1K_normal.png");
            parameters.textureRoughness = textureManager.GetTexture("data/TexturesCom_VinylChecker_1K_roughness.png");
            parameters.absorption = Float3(-logf(1.0f), -logf(1.0f), -logf(1.0f)) * 1.0f;
            parameters.ior = 1.5f;
            parameters.flags = 0;                       // FLAG_THINWALLED;
            m_materialParameters.push_back(parameters); // 0

            // Glass material
            parameters.indexBSDF = INDEX_BSDF_SPECULAR_REFLECTION_TRANSMISSION;
            parameters.albedo = Float3(1.0f, 1.0f, 1.0f);
            parameters.textureAlbedo = 0;
            parameters.textureNormal = 0;
            parameters.textureRoughness = 0;
            parameters.flags = 0;
            parameters.absorption = Float3(-logf(0.5f), -logf(0.75f), -logf(0.5f)) * 1.0f; // Green
            parameters.ior = 1.52f;                                                        // Flint glass. Higher IOR than the surrounding box.
            m_materialParameters.push_back(parameters);                                    // 1

            // Lambert material
            parameters.indexBSDF = INDEX_BSDF_DIFFUSE_REFLECTION;
            parameters.albedo = Float3(1.0f, 0.2f, 0.2f);
            parameters.textureAlbedo = 0;
            parameters.textureNormal = 0;
            parameters.textureRoughness = 0;
            parameters.flags = 0;
            parameters.absorption = Float3(-logf(1.0f), -logf(1.0f), -logf(1.0f)) * 1.0f;
            parameters.ior = 1.5f;
            m_materialParameters.push_back(parameters); // 2

            // Tinted mirror material.
            parameters.indexBSDF = INDEX_BSDF_SPECULAR_REFLECTION;
            parameters.albedo = Float3(0.2f, 0.2f, 1.0f); // blue
            parameters.textureAlbedo = 0;
            parameters.textureNormal = 0;
            parameters.textureRoughness = 0;
            parameters.flags = 0;
            parameters.absorption = Float3(-logf(1.0f), -logf(1.0f), -logf(1.0f)) * 1.0f;
            parameters.ior = 1.33f;
            m_materialParameters.push_back(parameters); // 3

            // Black BSDF for the light. This last material will not be shown inside the GUI!
            parameters.indexBSDF = INDEX_BSDF_SPECULAR_REFLECTION;
            parameters.albedo = Float3(0.0f, 1.0f, 1.0f);
            parameters.textureAlbedo = 0;
            parameters.textureNormal = 0;
            parameters.textureRoughness = 0;
            parameters.flags = 0;
            parameters.absorption = Float3(-logf(1.0f), -logf(1.0f), -logf(1.0f)) * 1.0f;
            parameters.ior = 1.0f;
            m_materialParameters.push_back(parameters); // 4
        }

        // Create instances
        assert((sizeof(SbtRecordHeader) % OPTIX_SBT_RECORD_ALIGNMENT) == 0);
        assert((sizeof(SbtRecordGeometryInstanceData) % OPTIX_SBT_RECORD_ALIGNMENT) == 0);

        Scene::Get().createGeometries(m_api, m_context, Backend::Get().getCudaStream(), m_geometries, m_instances);
        Scene::Get().m_updateCallback = [&](int objectId)
        {
            Scene::Get().updateGeometry(m_api, m_context, Backend::Get().getCudaStream(), m_geometries, m_instances, objectId);

            // Shader binding table record hit group geometry
            m_sbtRecordGeometryInstanceData[objectId].data.indices = (Int3 *)m_geometries[objectId].indices;
            m_sbtRecordGeometryInstanceData[objectId].data.attributes = (VertexAttributes *)m_geometries[objectId].attributes;
            CUDA_CHECK(cudaMemcpy((void *)m_d_sbtRecordGeometryInstanceData, m_sbtRecordGeometryInstanceData.data(), sizeof(SbtRecordGeometryInstanceData) * m_instances.size(), cudaMemcpyHostToDevice));

            m_sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(m_d_sbtRecordGeometryInstanceData);

            // Rebuild BVH
            {
                CUDA_CHECK(cudaStreamSynchronize(Backend::Get().getCudaStream()));
                CUDA_CHECK(cudaFree((void *)m_d_ias));

                CUdeviceptr d_instances;

                const size_t instancesSizeInBytes = sizeof(OptixInstance) * m_instances.size();

                CUDA_CHECK(cudaMalloc((void **)&d_instances, instancesSizeInBytes));
                CUDA_CHECK(cudaMemcpy((void *)d_instances, m_instances.data(), instancesSizeInBytes, cudaMemcpyHostToDevice));

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

                auto &backend = jazzfusion::Backend::Get();
                OPTIX_CHECK(m_api.optixAccelBuild(m_context, backend.getCudaStream(),
                                                  &accelBuildOptions, &instanceInput, 1,
                                                  d_tmp, iasBufferSizes.tempSizeInBytes,
                                                  m_d_ias, iasBufferSizes.outputSizeInBytes,
                                                  &m_root, nullptr, 0));

                CUDA_CHECK(cudaStreamSynchronize(backend.getCudaStream()));

                CUDA_CHECK(cudaFree((void *)d_tmp));
                CUDA_CHECK(cudaFree((void *)d_instances)); // Don't need the instances anymore.
            }

            // Update system param
            m_systemParameter.topObject = m_root;
            CUDA_CHECK(cudaMemcpy((void *)m_d_systemParameter, &m_systemParameter, sizeof(SystemParameter), cudaMemcpyHostToDevice));
        };

        // Build BVH
        {
            CUdeviceptr d_instances;

            const size_t instancesSizeInBytes = sizeof(OptixInstance) * m_instances.size();

            CUDA_CHECK(cudaMalloc((void **)&d_instances, instancesSizeInBytes));
            CUDA_CHECK(cudaMemcpy((void *)d_instances, m_instances.data(), instancesSizeInBytes, cudaMemcpyHostToDevice));

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

            auto &backend = jazzfusion::Backend::Get();
            OPTIX_CHECK(m_api.optixAccelBuild(m_context, backend.getCudaStream(),
                                              &accelBuildOptions, &instanceInput, 1,
                                              d_tmp, iasBufferSizes.tempSizeInBytes,
                                              m_d_ias, iasBufferSizes.outputSizeInBytes,
                                              &m_root, nullptr, 0));

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

            // Spherical HDR environment light.
            programGroupDescMissRadiance.miss.entryFunctionName = "__miss__env_sphere";

            OptixProgramGroup programGroupMissRadiance;
            OPTIX_CHECK(m_api.optixProgramGroupCreate(m_context, &programGroupDescMissRadiance, 1, &programGroupOptions, nullptr, nullptr, &programGroupMissRadiance));
            programGroups.push_back(programGroupMissRadiance);
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
            moduleList.push_back(moduleClosesthit);
        }

        // Direct callables
        {
            std::string ptxLightSample = ReadPtx("ptx/LightSample.ptx");
            std::string ptxBsdf = ReadPtx("ptx/Bsdf.ptx");

            OptixModule moduleLightSample;
            OptixModule moduleBsdf;

            OPTIX_CHECK(m_api.optixModuleCreate(m_context, &moduleCompileOptions, &pipelineCompileOptions, ptxLightSample.c_str(), ptxLightSample.size(), nullptr, nullptr, &moduleLightSample));
            OPTIX_CHECK(m_api.optixModuleCreate(m_context, &moduleCompileOptions, &pipelineCompileOptions, ptxBsdf.c_str(), ptxBsdf.size(), nullptr, nullptr, &moduleBsdf));

            std::vector<OptixProgramGroupDesc> programGroupDescCallables;

            OptixProgramGroupDesc pgd = {};

            pgd.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            pgd.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;

            pgd.callables.moduleDC = moduleLightSample;

            pgd.callables.entryFunctionNameDC = "__direct_callable__light_env_sphere";
            programGroupDescCallables.push_back(pgd);
            pgd.callables.entryFunctionNameDC = "__direct_callable__light_parallelogram";
            programGroupDescCallables.push_back(pgd);

            pgd.callables.moduleDC = moduleBsdf;

            pgd.callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_specular_reflection";
            programGroupDescCallables.push_back(pgd);
            pgd.callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_specular_reflection_transmission";
            programGroupDescCallables.push_back(pgd);
            pgd.callables.entryFunctionNameDC = "__direct_callable__sample_bsdf_diffuse_reflection";
            programGroupDescCallables.push_back(pgd);
            pgd.callables.entryFunctionNameDC = "__direct_callable__eval_bsdf_diffuse_reflection";
            programGroupDescCallables.push_back(pgd);

            programGroupCallables.resize(programGroupDescCallables.size());
            OPTIX_CHECK(m_api.optixProgramGroupCreate(m_context, programGroupDescCallables.data(), programGroupDescCallables.size(), &programGroupOptions, nullptr, nullptr, programGroupCallables.data()));
            for (size_t i = 0; i < programGroupDescCallables.size(); ++i)
            {
                programGroups.push_back(programGroupCallables[i]);
            }
            moduleList.push_back(moduleLightSample);
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
            CUDA_CHECK(cudaMemcpy((void *)m_d_sbtRecordRaygeneration, &sbtRecordRaygeneration, sizeof(SbtRecordHeader), cudaMemcpyHostToDevice));
        }

        // Miss group
        {
            SbtRecordHeader sbtRecordMiss;
            OPTIX_CHECK(m_api.optixSbtRecordPackHeader(programGroups[1], &sbtRecordMiss));
            CUDA_CHECK(cudaMalloc((void **)&m_d_sbtRecordMiss, sizeof(SbtRecordHeader)));
            CUDA_CHECK(cudaMemcpy((void *)m_d_sbtRecordMiss, &sbtRecordMiss, sizeof(SbtRecordHeader), cudaMemcpyHostToDevice));
        }

        // Hit group
        {
            OPTIX_CHECK(m_api.optixSbtRecordPackHeader(programGroups[2], &m_sbtRecordHitRadiance));

            const int numInstances = static_cast<int>(m_instances.size());
            m_sbtRecordGeometryInstanceData.resize(numInstances);

            for (int i = 0; i < numInstances; ++i)
            {
                const int idx = i;
                memcpy(m_sbtRecordGeometryInstanceData[idx].header, m_sbtRecordHitRadiance.header, OPTIX_SBT_RECORD_HEADER_SIZE);
                m_sbtRecordGeometryInstanceData[idx].data.indices = (Int3 *)m_geometries[i].indices;
                m_sbtRecordGeometryInstanceData[idx].data.attributes = (VertexAttributes *)m_geometries[i].attributes;
                m_sbtRecordGeometryInstanceData[idx].data.materialIndex = i;
                m_sbtRecordGeometryInstanceData[idx].data.lightIndex = -1;
            }

            CUDA_CHECK(cudaMalloc((void **)&m_d_sbtRecordGeometryInstanceData, sizeof(SbtRecordGeometryInstanceData) * numInstances));
            CUDA_CHECK(cudaMemcpy((void *)m_d_sbtRecordGeometryInstanceData, m_sbtRecordGeometryInstanceData.data(), sizeof(SbtRecordGeometryInstanceData) * numInstances, cudaMemcpyHostToDevice));
        }

        // Direct callables
        std::vector<SbtRecordHeader> sbtRecordCallables(programGroupCallables.size());
        {
            for (size_t i = 0; i < programGroupCallables.size(); ++i)
            {
                OPTIX_CHECK(m_api.optixSbtRecordPackHeader(programGroupCallables[i], &sbtRecordCallables[i]));
            }

            CUDA_CHECK(cudaMalloc((void **)&m_d_sbtRecordCallables, sizeof(SbtRecordHeader) * sbtRecordCallables.size()));
            CUDA_CHECK(cudaMemcpy((void *)m_d_sbtRecordCallables, sbtRecordCallables.data(), sizeof(SbtRecordHeader) * sbtRecordCallables.size(), cudaMemcpyHostToDevice));
        }

        // Setup the OptixShaderBindingTable
        {
            m_sbt.raygenRecord = m_d_sbtRecordRaygeneration;

            m_sbt.exceptionRecord = 0;

            m_sbt.missRecordBase = m_d_sbtRecordMiss;
            m_sbt.missRecordStrideInBytes = (unsigned int)sizeof(SbtRecordHeader);
            m_sbt.missRecordCount = 1;

            m_sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(m_d_sbtRecordGeometryInstanceData);
            m_sbt.hitgroupRecordStrideInBytes = (unsigned int)sizeof(SbtRecordGeometryInstanceData);
            const int numInstances = static_cast<int>(m_instances.size());
            m_sbt.hitgroupRecordCount = numInstances;

            m_sbt.callablesRecordBase = m_d_sbtRecordCallables;
            m_sbt.callablesRecordStrideInBytes = (unsigned int)sizeof(SbtRecordHeader);
            m_sbt.callablesRecordCount = (unsigned int)sbtRecordCallables.size();
        }

        // Setup "sysParam" data.
        {
            m_systemParameter.topObject = m_root;

            assert((sizeof(LightDefinition) & 15) == 0); // Check alignment to float4
            CUDA_CHECK(cudaMalloc((void **)&m_systemParameter.lightDefinitions, sizeof(LightDefinition) * m_lightDefinitions.size()));
            CUDA_CHECK(cudaMemcpy((void *)m_systemParameter.lightDefinitions, m_lightDefinitions.data(), sizeof(LightDefinition) * m_lightDefinitions.size(), cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMalloc((void **)&m_systemParameter.materialParameters, sizeof(MaterialParameter) * m_materialParameters.size()));
            CUDA_CHECK(cudaMemcpy((void *)m_systemParameter.materialParameters, m_materialParameters.data(), sizeof(MaterialParameter) * m_materialParameters.size(), cudaMemcpyHostToDevice));

            m_systemParameter.sceneEpsilon = 500.0f * 1.0e-7f;
            m_systemParameter.numLights = static_cast<unsigned int>(m_lightDefinitions.size());
            m_systemParameter.iterationIndex = 0;

            CUDA_CHECK(cudaMalloc((void **)&m_d_systemParameter, sizeof(SystemParameter)));
            CUDA_CHECK(cudaMemcpy((void *)m_d_systemParameter, &m_systemParameter, sizeof(SystemParameter), cudaMemcpyHostToDevice));
        }

        // Destroy modules
        for (auto &module : moduleList)
        {
            OPTIX_CHECK(m_api.optixModuleDestroy(module));
        }
    }

}