#include "denoising/Denoiser.h"

#include "denoising/FireflyFilter.h"
#include "denoising/BufferCopy.h"

#include "denoising/HitDistReconstruction.h"
#include "denoising/PrePass.h"
#include "denoising/TemporalAccumulation.h"
#include "denoising/HistoryFix.h"
#include "denoising/HistoryClamping.h"
#include "denoising/AtrousSmem.h"
#include "denoising/Atrous.h"

#include "core/GlobalSettings.h"
#include "core/BufferManager.h"
#include "core/Backend.h"
#include "core/OfflineBackend.h"
#include "core/RenderCamera.h"

#include "util/KernelHelper.h"
#include "util/BufferUtils.h"
#include <cassert>

void Denoiser::run(int width, int height, int historyWidth, int historyHeight)
{
    bufferDim = Int2(width, height);
    historyDim = Int2(historyWidth, historyHeight);
    Float2 invBufferDim = Float2(1.0f / (float)width, 1.0f / (float)height);

    auto &camera = RenderCamera::Get().camera;
    auto &historyCamera = RenderCamera::Get().historyCamera;

    assert(bufferDim.x != 0 && historyDim.x != 0);

    DenoisingParams &denoisingParams = GlobalSettings::GetDenoisingParams();

    int &iterationIndex = GlobalSettings::Get().iterationIndex;

    auto &bufferManager = BufferManager::Get();

    const int usedIterationIndex = iterationIndex > 0 ? iterationIndex - 1 : 0;
    const int reservoirParity = usedIterationIndex & 1;
    const int reservoirStride = bufferDim.x * bufferDim.y;

    if (denoisingParams.enableFireflyFilter && bufferManager.reservoirBuffer != nullptr)
    {
        const dim3 gridDim = GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x4x1);
        const dim3 blockDim = GetBlockDim(BLOCK_DIM_8x4x1);
        FireflyBoilingFilter KERNEL_ARGS2(gridDim, blockDim)(
            bufferDim,
            bufferManager.GetBuffer2D(IlluminationBuffer),
            bufferManager.GetBuffer2D(NormalRoughnessBuffer),
            bufferManager.GetBuffer2D(DepthBuffer),
            bufferManager.GetBuffer2D(MaterialBuffer),
            bufferManager.reservoirBuffer,
            reservoirStride,
            reservoirParity,
            80.0f,
            5.0f,
            0.8f,
            0.02f,
            denoisingParams.phiLuminance,
            camera);
    }

    int frameNum;
    if (GlobalSettings::IsOfflineMode())
    {
        auto &offlineBackend = OfflineBackend::Get();
        frameNum = offlineBackend.getFrameNum();
    }
    else
    {
        auto &backend = Backend::Get();
        frameNum = backend.getFrameNum();
    }

    // Track which buffer contains the final result
    int finalResultBuffer = 0; // 0 = IlluminationBuffer, 1 = IlluminationPingBuffer, 2 = IlluminationPongBuffer

    // Hit distance reconstruction pass (controlled by parameter)
    if (denoisingParams.enableHitDistanceReconstruction)
    {
        HitDistReconstruction<8, 2> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
            bufferDim,
            invBufferDim,
            bufferManager.GetBuffer2D(NormalRoughnessBuffer),
            bufferManager.GetBuffer2D(DepthBuffer),
            bufferManager.GetBuffer2D(IlluminationBuffer),
            bufferManager.GetBuffer2D(IlluminationPingBuffer));
        finalResultBuffer = 1; // Result in IlluminationPingBuffer
    }

    BufferCopySky KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
        bufferDim,
        bufferManager.GetBuffer2D(IlluminationBuffer),
        bufferManager.GetBuffer2D(DepthBuffer),
        bufferManager.GetBuffer2D(IlluminationOutputBuffer));

    // Pre-pass (controlled by parameter)
    if (denoisingParams.enablePrePass)
    {
        PrePass KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
            bufferDim,
            invBufferDim,

            bufferManager.GetTexture2D(IlluminationPingBuffer),

            bufferManager.GetBuffer2D(MaterialBuffer),
            bufferManager.GetBuffer2D(NormalRoughnessBuffer),
            bufferManager.GetBuffer2D(DepthBuffer),
            bufferManager.GetBuffer2D(IlluminationPingBuffer),

            bufferManager.GetBuffer2D(IlluminationBuffer),

            camera,
            iterationIndex);
        // Pre-pass typically doesn't change the final result buffer
    }

    if (frameNum == 0)
    {
        BufferCopyFloat4(
            bufferDim,
            bufferManager.GetBuffer2D(IlluminationBuffer),
            bufferManager.GetBuffer2D(PrevIlluminationBuffer));

        BufferCopyFloat4(
            bufferDim,
            bufferManager.GetBuffer2D(IlluminationBuffer),
            bufferManager.GetBuffer2D(PrevFastIlluminationBuffer));

        BufferSetFloat1(
            bufferDim,
            bufferManager.GetBuffer2D(HistoryLengthBuffer),
            0.0f);

        BufferSetFloat1(
            bufferDim,
            bufferManager.GetBuffer2D(PrevHistoryLengthBuffer),
            0.0f);
    }

    // Temporal accumulation pass (controlled by parameter)
    if (denoisingParams.enableTemporalAccumulation)
    {
        if (frameNum > 0)
        {
            TemporalAccumulation<8, 1> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                bufferDim,
                invBufferDim,

                bufferManager.GetBuffer2D(PrevDepthBuffer),
                bufferManager.GetBuffer2D(PrevMaterialBuffer),
                bufferManager.GetBuffer2D(PrevIlluminationBuffer),
                bufferManager.GetBuffer2D(PrevFastIlluminationBuffer),
                bufferManager.GetBuffer2D(PrevHistoryLengthBuffer),
                bufferManager.GetBuffer2D(PrevNormalRoughnessBuffer),

                bufferManager.GetBuffer2D(IlluminationBuffer),
                bufferManager.GetBuffer2D(DepthBuffer),
                bufferManager.GetBuffer2D(NormalRoughnessBuffer),
                bufferManager.GetBuffer2D(MaterialBuffer),

                bufferManager.GetBuffer2D(IlluminationPingBuffer),
                bufferManager.GetBuffer2D(IlluminationPongBuffer),
                bufferManager.GetBuffer2D(HistoryLengthBuffer),

                bufferManager.GetBuffer2D(DebugBuffer),

                camera,
                historyCamera,

                // Pass denoising parameters to kernel
                denoisingParams.denoisingRange,
                denoisingParams.disocclusionThreshold,
                denoisingParams.disocclusionThresholdAlternate,
                denoisingParams.maxAccumulatedFrameNum,
                denoisingParams.maxFastAccumulatedFrameNum);

            finalResultBuffer = 1; // Temporal accumulation outputs to IlluminationPingBuffer

            // Optional buffer copy (controlled by parameter - currently disabled)
            if (false) // This was originally if (0), keeping disabled for now
            {
                BufferCopyFloat4(
                    bufferDim,
                    bufferManager.GetBuffer2D(IlluminationPingBuffer),
                    bufferManager.GetBuffer2D(PrevIlluminationBuffer));

                BufferCopyFloat4(
                    bufferDim,
                    bufferManager.GetBuffer2D(IlluminationPongBuffer),
                    bufferManager.GetBuffer2D(PrevFastIlluminationBuffer));

                BufferCopyFloat1(
                    bufferDim,
                    bufferManager.GetBuffer2D(HistoryLengthBuffer),
                    bufferManager.GetBuffer2D(PrevHistoryLengthBuffer));
            }

            // History fix pass (controlled by parameter)
            if (denoisingParams.enableHistoryFix)
            {
                HistoryFix KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                    bufferDim,
                    invBufferDim,

                    bufferManager.GetBuffer2D(DepthBuffer),
                    bufferManager.GetBuffer2D(MaterialBuffer),
                    bufferManager.GetBuffer2D(NormalRoughnessBuffer),
                    bufferManager.GetBuffer2D(HistoryLengthBuffer),
                    bufferManager.GetBuffer2D(IlluminationPingBuffer),

                    bufferManager.GetBuffer2D(IlluminationPongBuffer),

                    camera);
                finalResultBuffer = 2; // History fix outputs to IlluminationPongBuffer
            }

            // History clamping pass (controlled by parameter)
            if (denoisingParams.enableHistoryClamping)
            {
                HistoryClamping<8, 2> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                    bufferDim,
                    invBufferDim,

                    bufferManager.GetBuffer2D(DepthBuffer),

                    bufferManager.GetBuffer2D(IlluminationBuffer),
                    bufferManager.GetBuffer2D(IlluminationPingBuffer),
                    bufferManager.GetBuffer2D(IlluminationPongBuffer),
                    bufferManager.GetBuffer2D(HistoryLengthBuffer),

                    bufferManager.GetBuffer2D(PrevIlluminationBuffer),
                    bufferManager.GetBuffer2D(PrevFastIlluminationBuffer),
                    bufferManager.GetBuffer2D(PrevHistoryLengthBuffer));
                // History clamping outputs the current frame data to PrevIlluminationBuffer
                finalResultBuffer = 3; // 3 = PrevIlluminationBuffer (after history clamping)
            }
        }
    }

    // Spatial filtering (A-trous) pass (controlled by parameter)
    if (denoisingParams.enableSpatialFiltering)
    {
        // According to NRD ReLaX implementation, A-trous always takes PrevIlluminationBuffer as input
        // This is because History Clamping writes the temporally denoised result to PrevIlluminationBuffer
        AtrousSmem<8, 2> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
            bufferDim,
            invBufferDim,

            bufferManager.GetBuffer2D(PrevIlluminationBuffer),

            bufferManager.GetBuffer2D(NormalRoughnessBuffer),
            bufferManager.GetBuffer2D(MaterialBuffer),
            bufferManager.GetBuffer2D(DepthBuffer),
            bufferManager.GetBuffer2D(HistoryLengthBuffer),

            bufferManager.GetBuffer2D(IlluminationPingBuffer),

            camera,

            // Pass denoising parameters to kernel
            denoisingParams.phiLuminance,
            denoisingParams.depthThreshold,
            denoisingParams.roughnessFraction,
            denoisingParams.lobeAngleFraction);

        finalResultBuffer = 1; // AtrousSmem outputs to IlluminationPingBuffer

        // A-trous iterations (controlled by parameter)
        if (denoisingParams.atrousIterationNum > 0)
        {
            int atrousIndex = 1;
            int atrousStep = 1 << atrousIndex;
            int maxIterations = denoisingParams.atrousIterationNum * 2; // Each iteration does two passes

            while (atrousIndex < maxIterations)
            {
                Atrous KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                    bufferDim,
                    invBufferDim,

                    bufferManager.GetBuffer2D(IlluminationPingBuffer),

                    bufferManager.GetBuffer2D(NormalRoughnessBuffer),
                    bufferManager.GetBuffer2D(MaterialBuffer),
                    bufferManager.GetBuffer2D(DepthBuffer),
                    bufferManager.GetBuffer2D(HistoryLengthBuffer),

                    bufferManager.GetBuffer2D(IlluminationPongBuffer),

                    camera,
                    iterationIndex,
                    atrousStep,

                    // Pass denoising parameters to kernel
                    denoisingParams.phiLuminance,
                    denoisingParams.depthThreshold,
                    denoisingParams.roughnessFraction,
                    denoisingParams.lobeAngleFraction);

                finalResultBuffer = 2; // First A-trous outputs to IlluminationPongBuffer
                ++atrousIndex;
                atrousStep = 1 << atrousIndex;

                Atrous KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                    bufferDim,
                    invBufferDim,

                    bufferManager.GetBuffer2D(IlluminationPongBuffer),

                    bufferManager.GetBuffer2D(NormalRoughnessBuffer),
                    bufferManager.GetBuffer2D(MaterialBuffer),
                    bufferManager.GetBuffer2D(DepthBuffer),
                    bufferManager.GetBuffer2D(HistoryLengthBuffer),

                    bufferManager.GetBuffer2D(IlluminationPingBuffer),

                    camera,
                    iterationIndex,
                    atrousStep,

                    // Pass denoising parameters to kernel
                    denoisingParams.phiLuminance,
                    denoisingParams.depthThreshold,
                    denoisingParams.roughnessFraction,
                    denoisingParams.lobeAngleFraction);

                finalResultBuffer = 1; // Second A-trous outputs back to IlluminationPingBuffer
                ++atrousIndex;
                atrousStep = 1 << atrousIndex;
            }

            Atrous KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                bufferDim,
                invBufferDim,

                bufferManager.GetBuffer2D(IlluminationPingBuffer),

                bufferManager.GetBuffer2D(NormalRoughnessBuffer),
                bufferManager.GetBuffer2D(MaterialBuffer),
                bufferManager.GetBuffer2D(DepthBuffer),
                bufferManager.GetBuffer2D(HistoryLengthBuffer),

                bufferManager.GetBuffer2D(IlluminationPongBuffer),

                camera,
                iterationIndex,
                atrousStep,

                // Pass denoising parameters to kernel
                denoisingParams.phiLuminance,
                denoisingParams.depthThreshold,
                denoisingParams.roughnessFraction,
                denoisingParams.lobeAngleFraction);

            finalResultBuffer = 2; // Final A-trous outputs to IlluminationPongBuffer
        }
    }

    // Use the correct final result buffer based on which passes were enabled
    SurfObj finalOutputBuffer;
    if (finalResultBuffer == 1)
        finalOutputBuffer = bufferManager.GetBuffer2D(IlluminationPingBuffer);
    else if (finalResultBuffer == 2)
        finalOutputBuffer = bufferManager.GetBuffer2D(IlluminationPongBuffer);
    else if (finalResultBuffer == 3)
        finalOutputBuffer = bufferManager.GetBuffer2D(PrevIlluminationBuffer);
    else
        finalOutputBuffer = bufferManager.GetBuffer2D(IlluminationBuffer);

    BufferCopyNonSky KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
        bufferDim,

        finalOutputBuffer,

        // bufferManager.GetBuffer2D(PrevIlluminationBuffer),
        // bufferManager.GetBuffer2D(IlluminationBuffer),
        // bufferManager.GetBuffer2D(IlluminationPingBuffer),

        // bufferManager.GetBuffer2D(AlbedoBuffer),
        // bufferManager.GetBuffer2D(DepthBuffer),
        // bufferManager.GetBuffer2D(PrevNormalRoughnessBuffer),
        // bufferManager.GetBuffer2D(PrevGeoNormalThinfilmBuffer),

        bufferManager.GetBuffer2D(DepthBuffer),
        bufferManager.GetBuffer2D(AlbedoBuffer),
        bufferManager.GetBuffer2D(UIBuffer),
        bufferManager.GetBuffer2D(IlluminationOutputBuffer));

    BufferCopyFloat4(
        bufferDim,
        bufferManager.GetBuffer2D(NormalRoughnessBuffer),
        bufferManager.GetBuffer2D(PrevNormalRoughnessBuffer));

    BufferCopyFloat1(
        bufferDim,
        bufferManager.GetBuffer2D(DepthBuffer),
        bufferManager.GetBuffer2D(PrevDepthBuffer));

    BufferCopyFloat1(
        bufferDim,
        bufferManager.GetBuffer2D(MaterialBuffer),
        bufferManager.GetBuffer2D(PrevMaterialBuffer));
}
