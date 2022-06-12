#include "postprocessing/PostProcessor.h"
#include "postprocessing/ScalingFilter.h"
#include "postprocessing/BicubicFilter.h"
#include "util/KernelHelper.h"
#include "util/DebugUtils.h"

namespace jazzfusion
{

void PostProcessor::init(int inputWidthIn, int inputHeightIn, int outputWidthIn, int outputHeightIn)
{
    inputWidth = inputWidthIn;
    inputHeight = inputHeightIn;
    outputWidth = outputWidthIn;
    outputHeight = outputHeightIn;
}

void PostProcessor::render(Float4* interopBuffer, SurfObj colorBuffer, TexObj colorTex)
{
    BicubicFilter KERNEL_ARGS2(GetGridDim(outputWidth, outputHeight, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))
        (interopBuffer, colorBuffer, Int2(inputWidth, inputHeight), Int2(outputWidth, outputHeight));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());
}

}