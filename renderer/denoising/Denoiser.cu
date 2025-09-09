#include "denoising/Denoiser.h"

#include "denoising/BufferCopy.h"

#include "denoising/HitDistReconstruction.h"
#include "denoising/PrePass.h"
#include "denoising/TemporalAccumulation.h"
#include "denoising/HistoryFix.h"
#include "denoising/HistoryClamping.h"
#include "denoising/AtrousSmem.h"
#include "denoising/Atrous.h"
#include "denoising/ComputeGradients.h"
#include "denoising/FilterGradients.h"
#include "denoising/GenerateConfidence.h"

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
    if (denoisingParams.enableDenoiser && denoisingParams.enableHitDistanceReconstruction)
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
    if (denoisingParams.enableDenoiser && denoisingParams.enablePrePass)
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
    if (denoisingParams.enableDenoiser && denoisingParams.enableTemporalAccumulation)
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


            // Confidence computation pipeline (RTXDI-based)
            if (denoisingParams.enableConfidenceComputation)
            {
                // 1. Compute gradients between current and previous frame ReSTIR luminance
                ComputeGradients KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                    bufferDim,
                    invBufferDim,

                    // Current frame ReSTIR luminance data
                    bufferManager.GetBuffer2D(RestirLuminanceBuffer),
                    bufferManager.GetBuffer2D(DepthBuffer),
                    bufferManager.GetBuffer2D(MaterialBuffer),
                    bufferManager.GetBuffer2D(MotionVectorBuffer),

                    // Previous frame ReSTIR luminance data
                    bufferManager.GetBuffer2D(PrevRestirLuminanceBuffer),

                    // Output gradient buffers
                    bufferManager.GetBuffer2D(DiffuseGradientBuffer),
                    bufferManager.GetBuffer2D(SpecularGradientBuffer),

                    // Camera data
                    camera,
                    historyCamera,
                    iterationIndex);

                // 2. Spatial gradient filtering (A-trous) for diffuse
                FilterGradients KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                    bufferDim,
                    invBufferDim,

                    bufferManager.GetBuffer2D(DiffuseGradientBuffer),
                    bufferManager.GetBuffer2D(FilteredDiffuseGradientBuffer),

                    denoisingParams.gradientFilterRadius,
                    denoisingParams.gradientFilterStepSize);

                // 3. Spatial gradient filtering (A-trous) for specular
                FilterGradients KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                    bufferDim,
                    invBufferDim,

                    bufferManager.GetBuffer2D(SpecularGradientBuffer),
                    bufferManager.GetBuffer2D(FilteredSpecularGradientBuffer),

                    denoisingParams.gradientFilterRadius,
                    denoisingParams.gradientFilterStepSize);

                // 4. Generate confidence from filtered gradients
                GenerateConfidence KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                    bufferDim,
                    invBufferDim,

                    bufferManager.GetBuffer2D(FilteredDiffuseGradientBuffer),
                    bufferManager.GetBuffer2D(FilteredSpecularGradientBuffer),
                    bufferManager.GetBuffer2D(PrevDiffuseConfidenceBuffer),
                    bufferManager.GetBuffer2D(PrevSpecularConfidenceBuffer),
                    bufferManager.GetBuffer2D(DepthBuffer),
                    bufferManager.GetBuffer2D(MaterialBuffer),

                    bufferManager.GetBuffer2D(DiffuseConfidenceBuffer),
                    bufferManager.GetBuffer2D(SpecularConfidenceBuffer),

                    camera,
                    historyCamera,

                    denoisingParams.gradientScale,
                    denoisingParams.enableTemporalFiltering,
                    denoisingParams.temporalWeight,
                    iterationIndex);
            }

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
    if (denoisingParams.enableDenoiser && denoisingParams.enableSpatialFiltering)
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
                    
                    // Confidence buffers
                    bufferManager.GetBuffer2D(DiffuseConfidenceBuffer),
                    bufferManager.GetBuffer2D(SpecularConfidenceBuffer),

                    camera,
                    iterationIndex,
                    atrousStep,

                    // Pass denoising parameters to kernel
                    denoisingParams.phiLuminance,
                    denoisingParams.depthThreshold,
                    denoisingParams.roughnessFraction,
                    denoisingParams.lobeAngleFraction,
                    
                    // Confidence parameters
                    denoisingParams.enableConfidenceComputation,
                    denoisingParams.confidenceDrivenRelaxationMultiplier,
                    denoisingParams.confidenceDrivenLuminanceEdgeStoppingRelaxation);

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
                    
                    // Confidence buffers
                    bufferManager.GetBuffer2D(DiffuseConfidenceBuffer),
                    bufferManager.GetBuffer2D(SpecularConfidenceBuffer),

                    camera,
                    iterationIndex,
                    atrousStep,

                    // Pass denoising parameters to kernel
                    denoisingParams.phiLuminance,
                    denoisingParams.depthThreshold,
                    denoisingParams.roughnessFraction,
                    denoisingParams.lobeAngleFraction,
                    
                    // Confidence parameters
                    denoisingParams.enableConfidenceComputation,
                    denoisingParams.confidenceDrivenRelaxationMultiplier,
                    denoisingParams.confidenceDrivenLuminanceEdgeStoppingRelaxation);

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
                
                // Confidence buffers
                bufferManager.GetBuffer2D(DiffuseConfidenceBuffer),
                bufferManager.GetBuffer2D(SpecularConfidenceBuffer),

                camera,
                iterationIndex,
                atrousStep,

                // Pass denoising parameters to kernel
                denoisingParams.phiLuminance,
                denoisingParams.depthThreshold,
                denoisingParams.roughnessFraction,
                denoisingParams.lobeAngleFraction,
                
                // Confidence parameters
                denoisingParams.enableConfidenceComputation,
                denoisingParams.confidenceDrivenRelaxationMultiplier,
                denoisingParams.confidenceDrivenLuminanceEdgeStoppingRelaxation);

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

    // Copy confidence and ReSTIR luminance for next frame (if confidence computation is enabled)
    if (denoisingParams.enableConfidenceComputation)
    {
        BufferCopyFloat1(
            bufferDim,
            bufferManager.GetBuffer2D(DiffuseConfidenceBuffer),
            bufferManager.GetBuffer2D(PrevDiffuseConfidenceBuffer));
            
        BufferCopyFloat1(
            bufferDim,
            bufferManager.GetBuffer2D(SpecularConfidenceBuffer),
            bufferManager.GetBuffer2D(PrevSpecularConfidenceBuffer));
            
        BufferCopyFloat2(
            bufferDim,
            bufferManager.GetBuffer2D(RestirLuminanceBuffer),
            bufferManager.GetBuffer2D(PrevRestirLuminanceBuffer));
    }
}