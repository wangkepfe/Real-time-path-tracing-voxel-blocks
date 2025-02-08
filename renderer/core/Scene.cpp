#include "core/Scene.h"
#include "shaders/LinearMath.h"

Scene::Scene()
{
    CUDA_CHECK(cudaMallocManaged(&edgeToHighlight, 4 * sizeof(Float3)));
    CUDA_CHECK(cudaMalloc(&d_lightAliasTable, sizeof(AliasTable)));
}

Scene::~Scene()
{
    CUDA_CHECK(cudaFree(edgeToHighlight));
    CUDA_CHECK(cudaFree(d_lightAliasTable));

    for (LightInfo *light : m_lights)
    {
        CUDA_CHECK(cudaFree(light));
    }
    CUDA_CHECK(cudaFree(d_mergedLights));
}

OptixTraversableHandle Scene::CreateGeometry(
    OptixFunctionTable &api,
    OptixDeviceContext &context,
    CUstream cudaStream,
    GeometryData &geometry,
    VertexAttributes *d_attributes,
    unsigned int *d_indices,
    unsigned int attributeSize,
    unsigned int indicesSize)
{
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

    accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
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

    return traversableHandle;
}
