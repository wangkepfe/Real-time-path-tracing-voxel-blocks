#include "Denoiser.h"
#include "core/Settings.h"

namespace jazzfusion
{

void Denoiser::init(int width, int height, int historyWidth, int historyHeight)
{
    bufferDim = Int2(width, height);
    historyDim = Int2(historyWidth, historyHeight);
}

// Calculate Noise Level
//
//         |
//         V
//
// Temporal Filter  <-----------------------------
//
//         |                                     ^
//         V                                     |
//
// Spatial Filter 7x7  (Effective range 7x7)
//
//         |                                     ^
//         V                                     |
//
// Copy To History Color Buffer ------------------
//
//         |
//         V
//
// Calculate Noise Level
//
//         |
//         V
//
// Spatial Filter Global 5x5, Stride=3 (Effective range 15x15)
// Spatial Filter Global 5x5, Stride=6 (Effective range 30x30)
// Spatial Filter Global 5x5, Stride=12 (Effective range 60x60)
//
//         |
//         V
//
// Temporal Filter 2  <----------------
//
//         |                          ^
//         V                          |
//
// Copy To History Color Buffer -------
//
// Done!
void Denoiser::run(SurfObj inColorBuffer, TexObj outColorBuffer)
{
    RenderPassSettings& renderPassSettings = GlobalSettings::GetRenderPassSettings();
    DenoisingParams& denoisingParams = GlobalSettings::GetDenoisingParams();

    UInt2 noiseLevel16x16Dim(divRoundUp(bufferDim.x, 16), divRoundUp(bufferDim.y, 16));
    if (renderPassSettings.enableTemporalDenoising)
    {
        if (cbo.frameNum != 1)
        {
            TemporalFilter KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
                cbo,
                GetBuffer2D(RenderColorBuffer),
                GetBuffer2D(AccumulationColorBuffer),
                GetBuffer2D(NormalBuffer),
                GetBuffer2D(DepthBuffer),
                GetBuffer2D(HistoryDepthBuffer),
                GetBuffer2D(MotionVectorBuffer),
                denoisingParams,
                bufferDim, historyDim);
        }
    }

    if (renderPassSettings.enableLocalSpatialFilter)
    {

        CalculateTileNoiseLevel KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x4x1)) (
            GetBuffer2D(RenderColorBuffer),
            GetBuffer2D(DepthBuffer),
            GetBuffer2D(NoiseLevelBuffer),
            bufferDim);

        TileNoiseLevel8x8to16x16 KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
            GetBuffer2D(NoiseLevelBuffer),
            GetBuffer2D(NoiseLevelBuffer16x16));

        if (renderPassSettings.enableNoiseLevelVisualize)
        {
            TileNoiseLevelVisualize KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
                GetBuffer2D(RenderColorBuffer),
                GetBuffer2D(NormalBuffer),
                GetBuffer2D(DepthBuffer),
                GetBuffer2D(NoiseLevelBuffer16x16),
                bufferDim,
                1,
                denoisingParams);
        }

        SpatialFilter7x7 KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
            cbo,
            GetBuffer2D(RenderColorBuffer),
            GetBuffer2D(NormalBuffer),
            GetBuffer2D(DepthBuffer),
            GetBuffer2D(NoiseLevelBuffer16x16),
            denoisingParams,
            bufferDim);
    }

    if (renderPassSettings.enableTemporalDenoising)
    {
        CopyToHistoryColorBuffer KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
            GetBuffer2D(RenderColorBuffer),
            GetBuffer2D(AccumulationColorBuffer),
            bufferDim);
    }

    if (renderPassSettings.enableWideSpatialFilter)
    {
        CalculateTileNoiseLevel KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x4x1)) (
            GetBuffer2D(RenderColorBuffer),
            GetBuffer2D(DepthBuffer),
            GetBuffer2D(NoiseLevelBuffer),
            bufferDim);

        TileNoiseLevel8x8to16x16 KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
            GetBuffer2D(NoiseLevelBuffer),
            GetBuffer2D(NoiseLevelBuffer16x16));

        if (renderPassSettings.enableNoiseLevelVisualize)
        {
            TileNoiseLevelVisualize KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
                GetBuffer2D(RenderColorBuffer),
                GetBuffer2D(NormalBuffer),
                GetBuffer2D(DepthBuffer),
                GetBuffer2D(NoiseLevelBuffer16x16),
                bufferDim,
                2,
                denoisingParams);
        }

        SpatialFilterGlobal5x5<3> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
            cbo,
            GetBuffer2D(RenderColorBuffer),
            GetBuffer2D(NormalBuffer),
            GetBuffer2D(DepthBuffer),
            GetBuffer2D(NoiseLevelBuffer16x16),
            denoisingParams,
            bufferDim);

        SpatialFilterGlobal5x5<6> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
            cbo,
            GetBuffer2D(RenderColorBuffer),
            GetBuffer2D(NormalBuffer),
            GetBuffer2D(DepthBuffer),
            GetBuffer2D(NoiseLevelBuffer16x16),
            denoisingParams,
            bufferDim);

        SpatialFilterGlobal5x5<12> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_16x16x1), GetBlockDim(BLOCK_DIM_16x16x1)) (
            cbo,
            GetBuffer2D(RenderColorBuffer),
            GetBuffer2D(NormalBuffer),
            GetBuffer2D(DepthBuffer),
            GetBuffer2D(NoiseLevelBuffer16x16),
            denoisingParams,
            bufferDim);
    }

    ApplyAlbedo KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
        GetBuffer2D(RenderColorBuffer),
        GetBuffer2D(AlbedoBuffer),
        bufferDim);

    if (renderPassSettings.enableTemporalDenoising2)
    {
        if (cbo.frameNum != 1)
        {
            TemporalFilter2 KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
                cbo,
                GetBuffer2D(RenderColorBuffer),
                GetBuffer2D(HistoryColorBuffer),
                GetBuffer2D(NormalBuffer),
                GetBuffer2D(DepthBuffer),
                GetBuffer2D(HistoryDepthBuffer),
                GetBuffer2D(MotionVectorBuffer),
                GetBuffer2D(NoiseLevelBuffer),
                bufferDim, historyDim);
        }

        CopyToHistoryColorDepthBuffer KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)) (
            GetBuffer2D(RenderColorBuffer),
            GetBuffer2D(DepthBuffer),
            GetBuffer2D(HistoryColorBuffer),
            GetBuffer2D(HistoryDepthBuffer),
            bufferDim);
    }
}

}