#include "denoising/Denoiser.h"
#include "denoising/DenoisingUtils.h"
#include "denoising/NoiseVisualization.h"
#include "denoising/SpatialFilter.h"
#include "denoising/SpatialWideFilter.h"
#include "denoising/TemporalFilter.h"
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
// Spatial Filter 7x7  (Effective range 7x7)     |
//         |                                     |
//         V                                     |
// Copy To History Color Buffer ------------------ AccumulationColorBuffer
//         |
//         V
// Spatial Filter Global 5x5, Stride=3 (Effective range 12x12)
// Spatial Filter Global 5x5, Stride=6 (Effective range 24x24)
// Spatial Filter Global 5x5, Stride=12 (Effective range 48x48)
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

    if (renderPassSettings.enableTemporalDenoising)
    {
        if (frameNum != 0)
        {
            TemporalFilter<true> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
                frameNum,
                bufferManager.GetBuffer2D(RenderColorBuffer),
                bufferManager.GetBuffer2D(AccumulationColorBuffer),
                bufferManager.GetBuffer2D(NormalBuffer),
                bufferManager.GetBuffer2D(DepthBuffer),
                bufferManager.GetBuffer2D(HistoryDepthBuffer),
                bufferManager.GetBuffer2D(MaterialBuffer),
                bufferManager.GetBuffer2D(MaterialHistoryBuffer),
                bufferManager.GetBuffer2D(MotionVectorBuffer),
                denoisingParams,
                bufferDim,
                historyDim);
        }
    }

    if (renderPassSettings.enableLocalSpatialFilter)
    {
        if (0)
        {
            CalculateTileNoiseLevel KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x4x1)) (
                bufferManager.GetBuffer2D(RenderColorBuffer),
                bufferManager.GetBuffer2D(DepthBuffer),
                bufferManager.GetBuffer2D(NoiseLevelBuffer),
                bufferDim);

            TileNoiseLevel8x8to16x16 KERNEL_ARGS2(GetGridDim(noiseLevel16x16Dim.x, noiseLevel16x16Dim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
                bufferManager.GetBuffer2D(NoiseLevelBuffer),
                bufferManager.GetBuffer2D(NoiseLevelBuffer16x16));

            if (renderPassSettings.enableNoiseLevelVisualize)
            {
                TileNoiseLevelVisualize<1> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
                    bufferManager.GetBuffer2D(RenderColorBuffer),
                    bufferManager.GetBuffer2D(NormalBuffer),
                    bufferManager.GetBuffer2D(DepthBuffer),
                    bufferManager.GetBuffer2D(NoiseLevelBuffer16x16),
                    bufferDim,
                    denoisingParams);
            }
        }

        SpatialFilter7x7 KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
            frameNum,
            bufferManager.GetBuffer2D(RenderColorBuffer),
            bufferManager.GetBuffer2D(MaterialBuffer),
            bufferManager.GetBuffer2D(NormalBuffer),
            bufferManager.GetBuffer2D(DepthBuffer),
            bufferManager.GetBuffer2D(NoiseLevelBuffer16x16),
            denoisingParams,
            bufferDim);
    }

    if (renderPassSettings.enableTemporalDenoising)
    {
        CopyToHistoryColorBuffer KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
            bufferManager.GetBuffer2D(RenderColorBuffer),
            bufferManager.GetBuffer2D(AccumulationColorBuffer),
            bufferDim);
    }

    if (renderPassSettings.enableWideSpatialFilter)
    {
        if (0)
        {
            CalculateTileNoiseLevel KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x4x1)) (
                bufferManager.GetBuffer2D(RenderColorBuffer),
                bufferManager.GetBuffer2D(DepthBuffer),
                bufferManager.GetBuffer2D(NoiseLevelBuffer),
                bufferDim);

            TileNoiseLevel8x8to16x16 KERNEL_ARGS2(GetGridDim(noiseLevel16x16Dim.x, noiseLevel16x16Dim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
                bufferManager.GetBuffer2D(NoiseLevelBuffer),
                bufferManager.GetBuffer2D(NoiseLevelBuffer16x16));

            if (renderPassSettings.enableNoiseLevelVisualize)
            {
                TileNoiseLevelVisualize<2> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
                    bufferManager.GetBuffer2D(RenderColorBuffer),
                    bufferManager.GetBuffer2D(NormalBuffer),
                    bufferManager.GetBuffer2D(DepthBuffer),
                    bufferManager.GetBuffer2D(NoiseLevelBuffer16x16),
                    bufferDim,
                    denoisingParams);
            }
        }

        SpatialWideFilter5x5<3> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
            frameNum,
            bufferManager.GetBuffer2D(RenderColorBuffer),
            bufferManager.GetBuffer2D(MaterialBuffer),
            bufferManager.GetBuffer2D(NormalBuffer),
            bufferManager.GetBuffer2D(DepthBuffer),
            bufferManager.GetBuffer2D(NoiseLevelBuffer16x16),
            denoisingParams,
            bufferDim);

        SpatialWideFilter5x5<6> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
            frameNum,
            bufferManager.GetBuffer2D(RenderColorBuffer),
            bufferManager.GetBuffer2D(MaterialBuffer),
            bufferManager.GetBuffer2D(NormalBuffer),
            bufferManager.GetBuffer2D(DepthBuffer),
            bufferManager.GetBuffer2D(NoiseLevelBuffer16x16),
            denoisingParams,
            bufferDim);

        SpatialWideFilter5x5<12> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
            frameNum,
            bufferManager.GetBuffer2D(RenderColorBuffer),
            bufferManager.GetBuffer2D(MaterialBuffer),
            bufferManager.GetBuffer2D(NormalBuffer),
            bufferManager.GetBuffer2D(DepthBuffer),
            bufferManager.GetBuffer2D(NoiseLevelBuffer16x16),
            denoisingParams,
            bufferDim);
    }

    RecoupleAlbedo KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
        bufferManager.GetBuffer2D(RenderColorBuffer),
        bufferManager.GetBuffer2D(AlbedoBuffer),
        bufferDim);

    if (renderPassSettings.enableTemporalDenoising2)
    {
        if (frameNum != 0)
        {
            TemporalFilter<false> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
                frameNum,
                bufferManager.GetBuffer2D(RenderColorBuffer),
                bufferManager.GetBuffer2D(HistoryColorBuffer),
                bufferManager.GetBuffer2D(NormalBuffer),
                bufferManager.GetBuffer2D(DepthBuffer),
                bufferManager.GetBuffer2D(HistoryDepthBuffer),
                bufferManager.GetBuffer2D(MaterialBuffer),
                bufferManager.GetBuffer2D(MaterialHistoryBuffer),
                bufferManager.GetBuffer2D(MotionVectorBuffer),
                denoisingParams,
                bufferDim,
                historyDim);
        }
    }

    CopyToHistoryColorDepthBuffer KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
        bufferManager.GetBuffer2D(RenderColorBuffer),
        bufferManager.GetBuffer2D(DepthBuffer),
        bufferManager.GetBuffer2D(HistoryColorBuffer),
        bufferManager.GetBuffer2D(HistoryDepthBuffer),
        bufferManager.GetBuffer2D(MaterialBuffer),
        bufferManager.GetBuffer2D(MaterialHistoryBuffer),
        bufferDim);
}

}