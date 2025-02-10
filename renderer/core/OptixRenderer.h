#pragma once

// Always include this before any OptiX headers!
#include <cuda_runtime.h>

#include <optix.h>

// OptiX 7 function table structure.
#include <optix_function_table.h>

#include "shaders/SystemParameter.h"

#include "util/DebugUtils.h"
#include "core/Scene.h"

#include "shaders/Camera.h"

#include "util/RandGenHost.h"
#include "shaders/RandGen.h"

class OptixRenderer
{
public:
    static OptixRenderer &Get()
    {
        static OptixRenderer instance;
        return instance;
    }
    OptixRenderer(OptixRenderer const &) = delete;
    void operator=(OptixRenderer const &) = delete;

    void init();
    void clear();
    void update();
    void render();

    SystemParameter &getSystemParameter() { return m_systemParameter; }

    void setWidth(int width) { m_width = width; }
    void setHeight(int height) { m_height = height; }

    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }

private:
    OptixRenderer() {}

    int m_width;
    int m_height;

    SystemParameter m_systemParameter;

    OptixFunctionTable m_api;
    OptixDeviceContext m_context;
    CUdeviceptr m_d_ias;
    OptixPipeline m_pipeline;

    SystemParameter *m_d_systemParameter;

    std::vector<OptixInstance> m_instances;
    std::unordered_set<unsigned int> instanceIds;
    std::unordered_map<unsigned int, OptixTraversableHandle> objectIdxToBlasHandleMap;

    std::vector<GeometryData> m_geometries;

    OptixShaderBindingTable m_sbt;

    std::vector<SbtRecordGeometryInstanceData> m_sbtRecordGeometryInstanceData;

    CUdeviceptr m_d_sbtRecordRaygeneration;
    CUdeviceptr m_d_sbtRecordMiss;
    CUdeviceptr m_d_sbtRecordCallables;

    SbtRecordGeometryInstanceData m_sbtRecordHitRadiance;
    SbtRecordGeometryInstanceData m_sbtRecordHitShadow;

    SbtRecordGeometryInstanceData *m_d_sbtRecordGeometryInstanceData;

    std::vector<MaterialParameter> m_materialParameters;

    BlueNoiseRandGeneratorHost h_randGen{};
    BlueNoiseRandGenerator d_randGen{static_cast<BlueNoiseRandGenerator>(h_randGen)};
};