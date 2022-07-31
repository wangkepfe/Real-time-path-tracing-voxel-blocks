#include "denoising/Denoiser.h"
#include "denoising/DenoisingUtils.h"
#include "denoising/NoiseVisualization.h"
#include "denoising/SpatialFilter.h"
#include "denoising/SpatialWideFilter.h"
#include "denoising/TemporalFilter.h"
#include "denoising/BilateralFilter.h"
#include "core/GlobalSettings.h"
#include "core/BufferManager.h"
#include "core/Backend.h"
#include "util/KernelHelper.h"
#include <cassert>

namespace jazzfusion
{

// Temporal Filter  <-----------------------------
//         |                                     ^
//         V                                     |
// Spatial Filter 5x5  (Effective range 5x5)     |
//         |                                     |
//         V                                     |
// Copy To History Color Buffer ------------------ AccumulationColorBuffer
//         |
//         V
// Spatial Filter Global 5x5, Stride=2 (Effective range 9x9)
// Spatial Filter Global 5x5, Stride=4 (Effective range 17x17)
// Spatial Filter Global 5x5, Stride=8 (Effective range 33x33)
//         |
//         V
// Temporal Filter 2  <----------------
//         |                          ^
//         V                          |
// Copy To History Color Buffer ------- HistoryColorBuffer
//
// Done!
void Denoiser::run(int width, int height, int historyWidth, int historyHeight)
{
    bufferDim = Int2(width, height);
    historyDim = Int2(historyWidth, historyHeight);

    assert(bufferDim.x != 0 && historyDim.x != 0);

    RenderPassSettings& renderPassSettings = GlobalSettings::GetRenderPassSettings();
    DenoisingParams& denoisingParams = GlobalSettings::GetDenoisingParams();

    auto& bufferManager = BufferManager::Get();
    auto& backend = Backend::Get();

    UInt2 noiseLevel16x16Dim(DivRoundUp(bufferDim.x, 16), DivRoundUp(bufferDim.y, 16));

    int frameNum = backend.getFrameNum();
    int accuCounter = backend.getAccumulationCounter();

    if (frameNum != 0)
    {
        TemporalFilter KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
            frameNum,
            accuCounter,
            bufferManager.GetBuffer2D(RenderColorBuffer),
            bufferManager.GetBuffer2D(AccumulationColorBuffer),
            bufferManager.GetBuffer2D(HistoryColorBuffer),
            bufferManager.GetBuffer2D(NormalBuffer),
            bufferManager.GetBuffer2D(DepthBuffer),
            bufferManager.GetBuffer2D(HistoryDepthBuffer),
            bufferManager.GetBuffer2D(MaterialBuffer),
            bufferManager.GetBuffer2D(MaterialHistoryBuffer),
            bufferManager.GetBuffer2D(AlbedoBuffer),
            bufferManager.GetBuffer2D(HistoryAlbedoBuffer),
            bufferManager.GetBuffer2D(MotionVectorBuffer),
            denoisingParams,
            bufferDim,
            historyDim);
    }

    CopyToAccumulationColorBuffer KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
        bufferManager.GetBuffer2D(RenderColorBuffer),
        bufferManager.GetBuffer2D(AccumulationColorBuffer),
        bufferManager.GetBuffer2D(AlbedoBuffer),
        bufferManager.GetBuffer2D(HistoryAlbedoBuffer),
        bufferDim);

    SpatialFilter KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
        frameNum,
        accuCounter,
        bufferManager.GetBuffer2D(RenderColorBuffer),
        bufferManager.GetBuffer2D(MaterialBuffer),
        bufferManager.GetBuffer2D(NormalBuffer),
        bufferManager.GetBuffer2D(DepthBuffer),
        denoisingParams,
        bufferDim);

    SpatialWideFilter<2> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
        frameNum,
        accuCounter,
        bufferManager.GetBuffer2D(RenderColorBuffer),
        bufferManager.GetBuffer2D(MaterialBuffer),
        bufferManager.GetBuffer2D(NormalBuffer),
        bufferManager.GetBuffer2D(DepthBuffer),
        denoisingParams,
        bufferDim);

    SpatialWideFilter<4> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
        frameNum,
        accuCounter,
        bufferManager.GetBuffer2D(RenderColorBuffer),
        bufferManager.GetBuffer2D(MaterialBuffer),
        bufferManager.GetBuffer2D(NormalBuffer),
        bufferManager.GetBuffer2D(DepthBuffer),
        denoisingParams,
        bufferDim);

    SpatialWideFilter<8> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
        frameNum,
        accuCounter,
        bufferManager.GetBuffer2D(RenderColorBuffer),
        bufferManager.GetBuffer2D(MaterialBuffer),
        bufferManager.GetBuffer2D(NormalBuffer),
        bufferManager.GetBuffer2D(DepthBuffer),
        denoisingParams,
        bufferDim);

    RecoupleAlbedo KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
        bufferManager.GetBuffer2D(RenderColorBuffer),
        bufferManager.GetBuffer2D(AlbedoBuffer),
        bufferManager.GetBuffer2D(MaterialBuffer),
        bufferDim);

    if (frameNum != 0)
    {
        TemporalFilter2 KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
            frameNum,
            accuCounter,
            bufferManager.GetBuffer2D(RenderColorBuffer),
            bufferManager.GetBuffer2D(AccumulationColorBuffer),
            bufferManager.GetBuffer2D(HistoryColorBuffer),
            bufferManager.GetBuffer2D(NormalBuffer),
            bufferManager.GetBuffer2D(DepthBuffer),
            bufferManager.GetBuffer2D(HistoryDepthBuffer),
            bufferManager.GetBuffer2D(MaterialBuffer),
            bufferManager.GetBuffer2D(MaterialHistoryBuffer),
            bufferManager.GetBuffer2D(AlbedoBuffer),
            bufferManager.GetBuffer2D(HistoryAlbedoBuffer),
            bufferManager.GetBuffer2D(MotionVectorBuffer),
            denoisingParams,
            bufferDim,
            historyDim);
    }

    CopyToHistoryColorBuffer KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
        bufferManager.GetBuffer2D(RenderColorBuffer),
        bufferManager.GetBuffer2D(HistoryColorBuffer),
        bufferManager.GetBuffer2D(DepthBuffer),
        bufferManager.GetBuffer2D(HistoryDepthBuffer),
        bufferManager.GetBuffer2D(MaterialBuffer),
        bufferManager.GetBuffer2D(MaterialHistoryBuffer),
        bufferDim);
}

}