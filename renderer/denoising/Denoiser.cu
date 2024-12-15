#include "denoising/Denoiser.h"

#include "denoising/HitDistReconstruction.h"

#include "core/GlobalSettings.h"
#include "core/BufferManager.h"
#include "core/Backend.h"
#include "util/KernelHelper.h"
#include <cassert>

namespace jazzfusion
{
    void Denoiser::run(int width, int height, int historyWidth, int historyHeight)
    {
        bufferDim = Int2(width, height);
        historyDim = Int2(historyWidth, historyHeight);
        Float2 invBufferDim = Float2(1.0f / (float)width, 1.0f / (float)height);

        assert(bufferDim.x != 0 && historyDim.x != 0);

        RenderPassSettings &renderPassSettings = GlobalSettings::GetRenderPassSettings();
        DenoisingParams &denoisingParams = GlobalSettings::GetDenoisingParams();

        auto &bufferManager = BufferManager::Get();
        auto &backend = Backend::Get();

        int frameNum = backend.getFrameNum();
        int accuCounter = backend.getAccumulationCounter();

        HitDistReconstruction<8, 1> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
            bufferDim,
            invBufferDim,
            bufferManager.GetBuffer2D(NormalRoughnessBuffer),
            bufferManager.GetBuffer2D(DepthBuffer),
            bufferManager.GetBuffer2D(IlluminationBuffer),
            bufferManager.GetBuffer2D(IlluminationPingBuffer));
    }

}