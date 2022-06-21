#include "postprocessing/PostProcessor.h"
#include "postprocessing/ScalingFilter.h"
#include "postprocessing/BicubicFilter.h"
#include "core/BufferManager.h"
#include "util/KernelHelper.h"
#include "util/DebugUtils.h"

namespace jazzfusion
{

void PostProcessor::run(Float4* interopBuffer, int inputWidthIn, int inputHeightIn, int outputWidthIn, int outputHeightIn)
{
    inputWidth = inputWidthIn;
    inputHeight = inputHeightIn;
    outputWidth = outputWidthIn;
    outputHeight = outputHeightIn;

    auto& bufferManager = BufferManager::Get();

    BicubicFilter KERNEL_ARGS2(GetGridDim(outputWidth, outputHeight, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))
        (interopBuffer,
            bufferManager.GetBuffer2D(RenderColorBuffer),
            Int2(inputWidth, inputHeight),
            Int2(outputWidth, outputHeight));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
}

}