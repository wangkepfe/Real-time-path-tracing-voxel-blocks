#include "postprocessing/PostProcessor.h"
#include "postprocessing/ScalingFilter.h"
#include "postprocessing/BicubicFilter.h"
#include "postprocessing/Tonemapping.h"
#include "postprocessing/SharpeningFilter.h"
#include "core/BufferManager.h"
#include "core/GlobalSettings.h"
#include "util/KernelHelper.h"
#include "util/DebugUtils.h"

namespace jazzfusion
{

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

        // float debugVisualization = Load2DFloat4(outColorBuffer, idx).w;

        // if (CUDA_CENTER_PIXEL())
        // {
        //     DEBUG_PRINT(debugVisualization);
        // }
        // float depth = debugVisualization;
        // float depthMin = 0.0f;
        // float depthMax = 10.0f;
        // // Apply a logarithmic curve to enhance visualization
        // float normalizedDepth = (depth > depthMin) ? logf(depth - depthMin + 1.0f) / logf(depthMax - depthMin + 1.0f) : 0.0f;

        // // Clamp to [0.0, 1.0] to avoid artifacts
        // normalizedDepth = fminf(fmaxf(normalizedDepth, 0.0f), 1.0f);

        // out[linearId] = Float4(Float3(normalizedDepth), 0);
    }

    void PostProcessor::run(Float4 *interopBuffer, int inputWidthIn, int inputHeightIn, int outputWidthIn, int outputHeightIn)
    {
        inputWidth = inputWidthIn;
        inputHeight = inputHeightIn;
        outputWidth = outputWidthIn;
        outputHeight = outputHeightIn;

        auto &bufferManager = BufferManager::Get();
        const auto &postProcessParams = GlobalSettings::GetPostProcessParams();
        const auto &renderPassSettings = GlobalSettings::GetRenderPassSettings();

        // ToneMappingReinhardExtended KERNEL_ARGS2(GetGridDim(inputWidth, inputHeight, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
        //     bufferManager.GetBuffer2D(IlluminationPingBuffer),
        //     Int2(inputWidth, inputHeight), postProcessParams);

        // if (renderPassSettings.enableSharpening)
        // {
        //     SharpeningFilter KERNEL_ARGS2(GetGridDim(outputWidth, outputHeight, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(bufferManager.GetBuffer2D(RenderColorBuffer), Int2(inputWidth, inputHeight));
        // }

        // if (renderPassSettings.enableEASU)
        // {
        //     EdgeAdaptiveSpatialUpsampling KERNEL_ARGS2(GetGridDim(outputWidth, outputHeight, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(bufferManager.GetBuffer2D(OutputColorBuffer), bufferManager.GetBuffer2D(RenderColorBuffer),
        //                                                                                                                                      inputWidth, inputHeight, inputWidth, inputHeight, outputWidth, outputHeight);
        // }
        // else
        // {
        // BicubicFilter KERNEL_ARGS2(GetGridDim(outputWidth, outputHeight, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
        //     bufferManager.GetBuffer2D(OutputColorBuffer),
        //     bufferManager.GetBuffer2D(IlluminationPingBuffer),
        //     Int2(inputWidth, inputHeight),
        //     Int2(outputWidth, outputHeight));
        // }

        CopyToInteropBuffer KERNEL_ARGS2(GetGridDim(outputWidth, outputHeight, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(interopBuffer, bufferManager.GetBuffer2D(IlluminationPingBuffer), Int2(outputWidth, outputHeight));

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaPeekAtLastError());
    }

}