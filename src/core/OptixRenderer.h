#pragma once

// Always include this before any OptiX headers!
#include <cuda_runtime.h>

#include <optix.h>

// OptiX 7 function table structure.
#include <optix_function_table.h>

#include "shaders/SystemParameter.h"

#include "util/Texture.h"

#include "util/DebugUtils.h"
#include "core/Scene.h"

#include "shaders/Camera.h"

namespace jazzfusion
{

struct SbtRecordHeader
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

template <typename T>
struct SbtRecordData
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecordData<GeometryInstanceData> SbtRecordGeometryInstanceData;

class OptixRenderer
{
public:
    static OptixRenderer& Get()
    {
        static OptixRenderer instance;
        return instance;
    }
    OptixRenderer(OptixRenderer const&) = delete;
    void operator=(OptixRenderer const&) = delete;

    void init();
    void clear();
    void render();

    Camera& getCamera() { return m_camera; }
    SystemParameter& getSystemParameter() { return m_systemParameter; }

    void setWidth(int width) { m_width = width; }
    void setHeight(int height) { m_height = height; }

    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }

    SurfObj getOutputBuffer() const { return m_outputBuffer; }
    TexObj getOutputTexture() const { return m_outputTexture; }

private:
    OptixRenderer() {}

    int                                        m_width;
    int                                        m_height;

    Camera                                     m_camera;
    SystemParameter                            m_systemParameter;

    Texture* m_textureEnvironment;
    Texture* m_textureAlbedo;

    OptixFunctionTable                         m_api;
    OptixDeviceContext                         m_context;
    OptixTraversableHandle                     m_root;
    CUdeviceptr                                m_d_ias;
    OptixPipeline                              m_pipeline;

    SystemParameter* m_d_systemParameter;

    std::vector<OptixInstance>                 m_instances;

    std::vector<GeometryData>                  m_geometries;

    OptixShaderBindingTable                    m_sbt;

    std::vector<SbtRecordGeometryInstanceData> m_sbtRecordGeometryInstanceData;

    CUdeviceptr                                m_d_sbtRecordRaygeneration;
    CUdeviceptr                                m_d_sbtRecordMiss;
    CUdeviceptr                                m_d_sbtRecordCallables;

    SbtRecordGeometryInstanceData              m_sbtRecordHitRadiance;
    SbtRecordGeometryInstanceData              m_sbtRecordHitShadow;
    SbtRecordGeometryInstanceData              m_sbtRecordHitRadianceCutout;
    SbtRecordGeometryInstanceData              m_sbtRecordHitShadowCutout;

    SbtRecordGeometryInstanceData* m_d_sbtRecordGeometryInstanceData;

    std::vector<LightDefinition>               m_lightDefinitions;
    std::vector<MaterialParameter>             m_materialParameters;

    TexObj m_outputTexture;
    SurfObj m_outputBuffer;
    cudaTextureDesc m_outputTexDesc;
    cudaArray_t m_outputBufferArray;
};

}