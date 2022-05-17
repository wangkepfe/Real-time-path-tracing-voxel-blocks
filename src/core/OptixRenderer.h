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
};

}