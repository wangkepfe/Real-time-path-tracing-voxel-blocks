#include "postprocessing/PostProcessor.h"
#include "postprocessing/ScalingFilter.h"
#include "postprocessing/BicubicFilter.h"
#include "postprocessing/FilmicToneMapping.h"
#include "postprocessing/SharpeningFilter.h"
#include "postprocessing/PostProcessingPipeline.h"
#include "core/BufferManager.h"
#include "core/GlobalSettings.h"
#include "core/RenderCamera.h"
#include "util/KernelHelper.h"
#include "util/DebugUtils.h"
#include "shaders/Sampler.h"

__global__ void DrawCrosshair(
    SurfObj colorBuffer,
    Int2 size)
{
    Int2 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx.x >= size.x || idx.y >= size.y)
        return;

    Int2 center = Int2(size.x / 2, size.y / 2);
    int crosshairSize = 10;
    int crosshairThickness = 1;

    Float3 color = Load2DFloat4(colorBuffer, idx).xyz;

    // Draw horizontal line
    if (abs(idx.y - center.y) <= crosshairThickness &&
        abs(idx.x - center.x) <= crosshairSize)
    {
        color = Float3(1.0f, 1.0f, 1.0f);
    }

    // Draw vertical line
    if (abs(idx.x - center.x) <= crosshairThickness &&
        abs(idx.y - center.y) <= crosshairSize)
    {
        color = Float3(1.0f, 1.0f, 1.0f);
    }

    Store2DFloat4(Float4(color, 1.0), colorBuffer, idx);
}

__global__ void CopyToInteropBuffer(
    Float4 *out,
    SurfObj outColorBuffer,
    Int2 outSize)
{
    Int2 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx.x >= outSize.x || idx.y >= outSize.y)
        return;
    int linearId = idx.y * outSize.x + idx.x;

    Float3 color = Load2DFloat4(outColorBuffer, idx).xyz;
    out[linearId] = Float4(color, 0);
}

PostProcessor::~PostProcessor()
{
    if (m_pipeline)
    {
        delete m_pipeline;
        m_pipeline = nullptr;
    }
}

void PostProcessor::run(Float4 *interopBuffer, int inputWidthIn, int inputHeightIn, int outputWidthIn, int outputHeightIn, float deltaTime)
{
    inputWidth = inputWidthIn;
    inputHeight = inputHeightIn;
    outputWidth = outputWidthIn;
    outputHeight = outputHeightIn;

    auto &bufferManager = BufferManager::Get();
    const auto &toneMappingParams = GlobalSettings::GetToneMappingParams();
    const auto &pipelineParams = GlobalSettings::GetPostProcessingPipelineParams();
    
    // Initialize pipeline if needed
    if (!m_pipeline)
    {
        m_pipeline = new PostProcessingPipeline();
        m_pipeline->Initialize(inputWidth, inputHeight);
    }
    
    // Delta time is now passed as parameter from Backend's Timer
    
    // Execute the post-processing pipeline and get computed exposure
    float computedExposure = m_pipeline->Execute(
        bufferManager.GetBuffer2D(IlluminationOutputBuffer),
        Int2(inputWidth, inputHeight),
        pipelineParams,
        toneMappingParams,
        deltaTime
    );
    
    // Store the computed exposure for UI display
    m_computedExposure = computedExposure;
    
    // Apply filmic tone mapping with computed exposure from pipeline
    FilmicToneMapping KERNEL_ARGS2(GetGridDim(inputWidth, inputHeight, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
        bufferManager.GetBuffer2D(IlluminationOutputBuffer),
        Int2(inputWidth, inputHeight),
        pipelineParams,     // Pipeline params for bloom and other settings
        toneMappingParams,  // Tone mapping params for curves and color grading
        computedExposure);  // Exposure computed by pipeline's histogram-based auto-exposure

    DrawCrosshair KERNEL_ARGS2(GetGridDim(inputWidth, inputHeight, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
        bufferManager.GetBuffer2D(IlluminationOutputBuffer),
        Int2(inputWidth, inputHeight));

    CopyToInteropBuffer KERNEL_ARGS2(GetGridDim(outputWidth, outputHeight, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(interopBuffer, bufferManager.GetBuffer2D(IlluminationOutputBuffer), Int2(outputWidth, outputHeight));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
}