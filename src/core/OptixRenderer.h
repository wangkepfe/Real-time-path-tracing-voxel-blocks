#pragma once

// Always include this before any OptiX headers!
#include <cuda_runtime.h>

#include <optix.h>

// OptiX 7 function table structure.
#include <optix_function_table.h>

#include "shaders/system_parameter.h"
#include "shaders/function_indices.h"
#include "shaders/light_definition.h"
#include "shaders/vertex_attributes.h"

#include "core/Texture.h"

#include "DebugUtils.h"
#include "core/Scene.h"

namespace jazzfusion {

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

    int                        m_width;
    int                        m_height;

    PinholeCamera              m_pinholeCamera;

    SystemParameter            m_systemParameter;

private:
    OptixRenderer() {}

    Texture*                   m_textureEnvironment;
    Texture*                   m_textureAlbedo;

    OptixFunctionTable         m_api;
    OptixDeviceContext         m_context;
    OptixTraversableHandle     m_root;               // Scene root
    CUdeviceptr                m_d_ias;              // Scene root's IAS (instance acceleration structure).
    OptixPipeline              m_pipeline;

    SystemParameter*           m_d_systemParameter;

    std::vector<OptixInstance> m_instances;

    std::vector<GeometryData> m_geometries;

    OptixShaderBindingTable m_sbt;

    std::vector<SbtRecordGeometryInstanceData> m_sbtRecordGeometryInstanceData;

    CUdeviceptr m_d_sbtRecordRaygeneration;
    CUdeviceptr m_d_sbtRecordMiss;
    CUdeviceptr m_d_sbtRecordCallables;

    SbtRecordGeometryInstanceData m_sbtRecordHitRadiance;
    SbtRecordGeometryInstanceData m_sbtRecordHitShadow;
    SbtRecordGeometryInstanceData m_sbtRecordHitRadianceCutout;
    SbtRecordGeometryInstanceData m_sbtRecordHitShadowCutout;

    SbtRecordGeometryInstanceData *m_d_sbtRecordGeometryInstanceData;

    std::vector<LightDefinition> m_lightDefinitions;
    std::vector<MaterialParameter> m_materialParameters;
};

}