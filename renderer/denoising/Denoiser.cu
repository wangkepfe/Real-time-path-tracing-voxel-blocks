#include "denoising/Denoiser.h"

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
#include "core/RenderCamera.h"

#include "util/KernelHelper.h"
#include "util/BufferUtils.h"
#include <cassert>

namespace jazzfusion
{
    void Denoiser::run(int width, int height, int historyWidth, int historyHeight)
    {
        bufferDim = Int2(width, height);
        historyDim = Int2(historyWidth, historyHeight);
        Float2 invBufferDim = Float2(1.0f / (float)width, 1.0f / (float)height);

        auto &camera = RenderCamera::Get().camera;
        auto &historyCamera = RenderCamera::Get().historyCamera;

        assert(bufferDim.x != 0 && historyDim.x != 0);

        RenderPassSettings &renderPassSettings = GlobalSettings::GetRenderPassSettings();
        DenoisingParams &denoisingParams = GlobalSettings::GetDenoisingParams();

        int &iterationIndex = GlobalSettings::Get().iterationIndex;

        auto &bufferManager = BufferManager::Get();
        auto &backend = Backend::Get();

        int frameNum = backend.getFrameNum();
        int accuCounter = backend.getAccumulationCounter();

        if (1)
        {
            HitDistReconstruction<8, 2> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                bufferDim,
                invBufferDim,
                bufferManager.GetBuffer2D(NormalRoughnessBuffer),
                bufferManager.GetBuffer2D(DepthBuffer),
                bufferManager.GetBuffer2D(IlluminationBuffer),
                bufferManager.GetBuffer2D(IlluminationPingBuffer));
        }

        BufferCopySky KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
            bufferDim,
            bufferManager.GetBuffer2D(IlluminationBuffer),
            bufferManager.GetBuffer2D(DepthBuffer),
            bufferManager.GetBuffer2D(IlluminationOutputBuffer));

        if (1)
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
        }

        if (1)
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
                    historyCamera);

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
            }
            else
            {
                BufferCopyFloat4 KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                    bufferDim,
                    bufferManager.GetBuffer2D(IlluminationBuffer),
                    bufferManager.GetBuffer2D(PrevIlluminationBuffer));

                BufferCopyFloat4 KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
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

            BufferCopyFloat4 KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                bufferDim,
                bufferManager.GetBuffer2D(NormalRoughnessBuffer),
                bufferManager.GetBuffer2D(PrevNormalRoughnessBuffer));

            BufferCopyFloat1 KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                bufferDim,
                bufferManager.GetBuffer2D(DepthBuffer),
                bufferManager.GetBuffer2D(PrevDepthBuffer));

            BufferCopyFloat1 KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                bufferDim,
                bufferManager.GetBuffer2D(MaterialBuffer),
                bufferManager.GetBuffer2D(PrevMaterialBuffer));
        }

        if (1)
        {
            AtrousSmem<8, 2> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                bufferDim,
                invBufferDim,

                bufferManager.GetBuffer2D(PrevIlluminationBuffer),

                bufferManager.GetBuffer2D(NormalRoughnessBuffer),
                bufferManager.GetBuffer2D(MaterialBuffer),
                bufferManager.GetBuffer2D(DepthBuffer),
                bufferManager.GetBuffer2D(HistoryLengthBuffer),

                bufferManager.GetBuffer2D(IlluminationPingBuffer),

                camera);
        }

        if (1)
        {
            int atrousIndex = 1;
            int atrousStep = 1 << atrousIndex;
            while (atrousIndex < 7)
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
                    atrousStep);

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
                    atrousStep);

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
                atrousStep);
        }

        BufferCopyNonSky KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
            bufferDim,
            bufferManager.GetBuffer2D(IlluminationPongBuffer),
            bufferManager.GetBuffer2D(DepthBuffer),
            bufferManager.GetBuffer2D(AlbedoBuffer),
            bufferManager.GetBuffer2D(UiBuffer),
            bufferManager.GetBuffer2D(IlluminationOutputBuffer));
    }
}