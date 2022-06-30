#include "denoising/Denoiser.h"
#include "denoising/DenoisingUtils.h"
#include "denoising/NoiseVisualization.h"
#include "denoising/SpatialFilter.h"
#include "denoising/SpatialWideFilter.h"
#include "denoising/TemporalFilter.h"
#include "denoising/BilateralFilter.h"
#include "denoising/SpatialFilterTranslucentMaterial.h"
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
    int accuCounter = backend.getAccumulationCounter();

    // if (frameNum != 0)
    // {
    //     TemporalDenoiseDepthNormalMat KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
    //         accuCounter,
    //         bufferManager.GetBuffer2D(DepthBuffer),
    //         bufferManager.GetBuffer2D(HistoryDepthBuffer),
    //         bufferManager.GetBuffer2D(MaterialBuffer),
    //         bufferManager.GetBuffer2D(MaterialHistoryBuffer),
    //         bufferManager.GetBuffer2D(NormalBuffer),
    //         bufferManager.GetBuffer2D(HistoryNormalBuffer),
    //         bufferDim);
    // }

    if (renderPassSettings.enableTemporalDenoising)
    {
        if (frameNum != 0)
        {
            if (accuCounter == 1)
            {
                TemporalFilter<true, true> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
                    frameNum,
                    accuCounter,
                    bufferManager.GetBuffer2D(RenderColorBuffer),
                    bufferManager.GetBuffer2D(AccumulationColorBuffer),
                    bufferManager.GetBuffer2D(HistoryColorBuffer),
                    bufferManager.GetBuffer2D(NormalBuffer),
                    bufferManager.GetBuffer2D(HistoryNormalBuffer),
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
            else
            {
                TemporalFilter<true, false> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
                    frameNum,
                    accuCounter,
                    bufferManager.GetBuffer2D(RenderColorBuffer),
                    bufferManager.GetBuffer2D(AccumulationColorBuffer),
                    bufferManager.GetBuffer2D(HistoryColorBuffer),
                    bufferManager.GetBuffer2D(NormalBuffer),
                    bufferManager.GetBuffer2D(HistoryNormalBuffer),
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
        }
    }

    if (renderPassSettings.enableLocalSpatialFilter)
    {
        if (ENABLE_DENOISING_NOISE_CALCULATION)
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

        if (accuCounter < 64)
        {
            SpatialFilter KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
                frameNum,
                accuCounter,
                bufferManager.GetBuffer2D(RenderColorBuffer),
                bufferManager.GetBuffer2D(MaterialBuffer),
                bufferManager.GetBuffer2D(NormalBuffer),
                bufferManager.GetBuffer2D(DepthBuffer),
                bufferManager.GetBuffer2D(NoiseLevelBuffer16x16),
                denoisingParams,
                bufferDim);
        }
    }

    // if (renderPassSettings.enableTemporalDenoising)
    // {
    //     CopyToAccumulationColorBuffer KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
    //         bufferManager.GetBuffer2D(RenderColorBuffer),
    //         bufferManager.GetBuffer2D(AccumulationColorBuffer),
    //         bufferDim);
    // }

    CopyToAccumulationColorBuffer KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
        bufferManager.GetBuffer2D(RenderColorBuffer),
        bufferManager.GetBuffer2D(AccumulationColorBuffer),
        bufferManager.GetBuffer2D(AlbedoBuffer),
        bufferManager.GetBuffer2D(HistoryAlbedoBuffer),
        bufferDim);

    if (renderPassSettings.enableWideSpatialFilter)
    {
        if (ENABLE_DENOISING_NOISE_CALCULATION)
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

        if (accuCounter < 64)
        {
            SpatialWideFilter<2> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
                frameNum,
                accuCounter,
                bufferManager.GetBuffer2D(RenderColorBuffer),
                bufferManager.GetBuffer2D(MaterialBuffer),
                bufferManager.GetBuffer2D(NormalBuffer),
                bufferManager.GetBuffer2D(DepthBuffer),
                bufferManager.GetBuffer2D(NoiseLevelBuffer16x16),
                denoisingParams,
                bufferDim);

            if (accuCounter < 32)
            {
                SpatialWideFilter<4> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
                    frameNum,
                    accuCounter,
                    bufferManager.GetBuffer2D(RenderColorBuffer),
                    bufferManager.GetBuffer2D(MaterialBuffer),
                    bufferManager.GetBuffer2D(NormalBuffer),
                    bufferManager.GetBuffer2D(DepthBuffer),
                    bufferManager.GetBuffer2D(NoiseLevelBuffer16x16),
                    denoisingParams,
                    bufferDim);

                if (accuCounter < 16)
                {
                    SpatialWideFilter<8> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
                        frameNum,
                        accuCounter,
                        bufferManager.GetBuffer2D(RenderColorBuffer),
                        bufferManager.GetBuffer2D(MaterialBuffer),
                        bufferManager.GetBuffer2D(NormalBuffer),
                        bufferManager.GetBuffer2D(DepthBuffer),
                        bufferManager.GetBuffer2D(NoiseLevelBuffer16x16),
                        denoisingParams,
                        bufferDim);
                }
            }
        }
    }

    RecoupleAlbedo KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
        bufferManager.GetBuffer2D(RenderColorBuffer),
        bufferManager.GetBuffer2D(AlbedoBuffer),
        bufferManager.GetBuffer2D(MaterialBuffer),
        bufferDim);

    if (1)
    {
        SpatialFilter2 KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
            frameNum,
            accuCounter,
            bufferManager.GetBuffer2D(RenderColorBuffer),
            bufferManager.GetBuffer2D(MaterialBuffer),
            bufferManager.GetBuffer2D(NormalBuffer),
            bufferManager.GetBuffer2D(DepthBuffer),
            bufferManager.GetBuffer2D(NoiseLevelBuffer16x16),
            denoisingParams,
            bufferDim);

        BilateralFilter2 KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
            frameNum,
            accuCounter,
            bufferManager.GetBuffer2D(RenderColorBuffer),
            bufferManager.GetBuffer2D(MaterialBuffer),
            bufferManager.GetBuffer2D(DepthBuffer),
            denoisingParams,
            bufferDim);

        if (accuCounter < 16)
        {
            SpatialWideFilter2<2> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
                frameNum,
                accuCounter,
                bufferManager.GetBuffer2D(RenderColorBuffer),
                bufferManager.GetBuffer2D(MaterialBuffer),
                bufferManager.GetBuffer2D(NormalBuffer),
                bufferManager.GetBuffer2D(DepthBuffer),
                bufferManager.GetBuffer2D(NoiseLevelBuffer16x16),
                denoisingParams,
                bufferDim);

            BilateralFilterWide2<2> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
                frameNum,
                accuCounter,
                bufferManager.GetBuffer2D(RenderColorBuffer),
                bufferManager.GetBuffer2D(MaterialBuffer),
                bufferManager.GetBuffer2D(DepthBuffer),
                denoisingParams,
                bufferDim);

            if (accuCounter < 4)
            {
                SpatialWideFilter2<4> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
                    frameNum,
                    accuCounter,
                    bufferManager.GetBuffer2D(RenderColorBuffer),
                    bufferManager.GetBuffer2D(MaterialBuffer),
                    bufferManager.GetBuffer2D(NormalBuffer),
                    bufferManager.GetBuffer2D(DepthBuffer),
                    bufferManager.GetBuffer2D(NoiseLevelBuffer16x16),
                    denoisingParams,
                    bufferDim);

                BilateralFilterWide2<4> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
                    frameNum,
                    accuCounter,
                    bufferManager.GetBuffer2D(RenderColorBuffer),
                    bufferManager.GetBuffer2D(MaterialBuffer),
                    bufferManager.GetBuffer2D(DepthBuffer),
                    denoisingParams,
                    bufferDim);

                if (accuCounter < 2)
                {
                    SpatialWideFilter2<8> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
                        frameNum,
                        accuCounter,
                        bufferManager.GetBuffer2D(RenderColorBuffer),
                        bufferManager.GetBuffer2D(MaterialBuffer),
                        bufferManager.GetBuffer2D(NormalBuffer),
                        bufferManager.GetBuffer2D(DepthBuffer),
                        bufferManager.GetBuffer2D(NoiseLevelBuffer16x16),
                        denoisingParams,
                        bufferDim);

                    BilateralFilterWide2<8> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
                        frameNum,
                        accuCounter,
                        bufferManager.GetBuffer2D(RenderColorBuffer),
                        bufferManager.GetBuffer2D(MaterialBuffer),
                        bufferManager.GetBuffer2D(DepthBuffer),
                        denoisingParams,
                        bufferDim);
                }
            }
        }


        if (renderPassSettings.enableTemporalDenoising2)
        {
            if (frameNum != 0)
            {
                if (accuCounter == 1)
                {
                    TemporalFilter<false, true> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
                        frameNum,
                        accuCounter,
                        bufferManager.GetBuffer2D(RenderColorBuffer),
                        bufferManager.GetBuffer2D(AccumulationColorBuffer),
                        bufferManager.GetBuffer2D(HistoryColorBuffer),
                        bufferManager.GetBuffer2D(NormalBuffer),
                        bufferManager.GetBuffer2D(HistoryNormalBuffer),
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
                else
                {
                    TemporalFilter<false, false> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
                        frameNum,
                        accuCounter,
                        bufferManager.GetBuffer2D(RenderColorBuffer),
                        bufferManager.GetBuffer2D(AccumulationColorBuffer),
                        bufferManager.GetBuffer2D(HistoryColorBuffer),
                        bufferManager.GetBuffer2D(NormalBuffer),
                        bufferManager.GetBuffer2D(HistoryNormalBuffer),
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
            }
            else
            {
                CopyToHistoryColorBuffer KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
                    bufferManager.GetBuffer2D(RenderColorBuffer),
                    bufferManager.GetBuffer2D(HistoryColorBuffer),
                    bufferManager.GetBuffer2D(DepthBuffer),
                    bufferManager.GetBuffer2D(HistoryDepthBuffer),
                    bufferManager.GetBuffer2D(MaterialBuffer),
                    bufferManager.GetBuffer2D(MaterialHistoryBuffer),
                    bufferManager.GetBuffer2D(NormalBuffer),
                    bufferManager.GetBuffer2D(HistoryNormalBuffer),
                    bufferDim);
            }
        }

        // TemporalSpatialFilterTranslucentMaterial7x7 KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
        //     frameNum,
        //     accuCounter,
        //     bufferManager.GetBuffer2D(RenderColorBuffer),
        //     bufferManager.GetBuffer2D(MaterialBuffer),
        //     bufferManager.GetBuffer2D(NormalFrontBuffer),
        //     bufferManager.GetBuffer2D(DepthFrontBuffer),
        //     bufferManager.GetBuffer2D(NoiseLevelBuffer16x16),
        //     denoisingParams,
        //     bufferDim);

        // if (renderPassSettings.enableBilateralFilter)
        // {
        //     BilateralFilter KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
        //         frameNum,
        //         accuCounter,
        //         bufferManager.GetBuffer2D(RenderColorBuffer),
        //         bufferManager.GetBuffer2D(MaterialBuffer),
        //         denoisingParams,
        //         bufferDim);

        //     BilateralFilterWide<2> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
        //         frameNum,
        //         accuCounter,
        //         bufferManager.GetBuffer2D(RenderColorBuffer),
        //         bufferManager.GetBuffer2D(MaterialBuffer),
        //         denoisingParams,
        //         bufferDim);
        // }
    }

}
}