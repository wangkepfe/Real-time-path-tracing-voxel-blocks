#include "postprocessing/PostProcessor.h"
#include "postprocessing/ScalingFilter.h"
#include "postprocessing/BicubicFilter.h"
#include "postprocessing/Tonemapping.h"
#include "postprocessing/SharpeningFilter.h"
#include "core/BufferManager.h"
#include "core/GlobalSettings.h"
#include "core/RenderCamera.h"
#include "util/KernelHelper.h"
#include "util/DebugUtils.h"

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

void PostProcessor::run(Float4 *interopBuffer, int inputWidthIn, int inputHeightIn, int outputWidthIn, int outputHeightIn)
{
    inputWidth = inputWidthIn;
    inputHeight = inputHeightIn;
    outputWidth = outputWidthIn;
    outputHeight = outputHeightIn;

    auto &bufferManager = BufferManager::Get();
    const auto &postProcessParams = GlobalSettings::GetPostProcessParams();

    ToneMappingReinhardExtended KERNEL_ARGS2(GetGridDim(inputWidth, inputHeight, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
        bufferManager.GetBuffer2D(IlluminationOutputBuffer),
        Int2(inputWidth, inputHeight),
        postProcessParams);

    CopyToInteropBuffer KERNEL_ARGS2(GetGridDim(outputWidth, outputHeight, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(interopBuffer, bufferManager.GetBuffer2D(IlluminationOutputBuffer), Int2(outputWidth, outputHeight));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
}