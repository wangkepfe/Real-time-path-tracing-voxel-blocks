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
#include "denoising/AntiFirefly.h"

#include "core/GlobalSettings.h"
#include "core/BufferManager.h"
#include "core/Backend.h"
#include "core/OfflineBackend.h"
#include "core/RenderCamera.h"

#include "util/KernelHelper.h"
#include "util/BufferUtils.h"
#include <cassert>

// Simple kernel to combine separate diffuse and specular results
__global__ void CombineDiffuseSpecular(
    Int2 screenResolution,
    SurfObj diffuseBuffer,
    SurfObj specularBuffer,
    SurfObj combinedOutputBuffer)
{
    Int2 pixelPos;
    pixelPos.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelPos.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelPos.x >= screenResolution.x || pixelPos.y >= screenResolution.y)
        return;

    // Load diffuse and specular contributions
    Float4 diffuse = Load2DFloat4(diffuseBuffer, pixelPos);
    Float4 specular = Load2DFloat4(specularBuffer, pixelPos);

    // Combine by adding diffuse and specular (standard lighting model)
    Float4 combined = Float4(diffuse.xyz + specular.xyz, diffuse.w + specular.w);

    // DO NOT CHANGE THIS: I'm testing the specular contribution now!!
    Store2DFloat4(specular, combinedOutputBuffer, pixelPos);
}

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

        BufferCopyFloat4(
            bufferDim,
            bufferManager.GetBuffer2D(DiffuseIlluminationBuffer),
            bufferManager.GetBuffer2D(PrevDiffuseIlluminationBuffer));

        BufferCopyFloat4(
            bufferDim,
            bufferManager.GetBuffer2D(DiffuseIlluminationBuffer),
            bufferManager.GetBuffer2D(PrevDiffuseFastIlluminationBuffer));

        BufferCopyFloat4(
            bufferDim,
            bufferManager.GetBuffer2D(SpecularIlluminationBuffer),
            bufferManager.GetBuffer2D(PrevSpecularIlluminationBuffer));

        BufferCopyFloat4(
            bufferDim,
            bufferManager.GetBuffer2D(SpecularIlluminationBuffer),
            bufferManager.GetBuffer2D(PrevSpecularFastIlluminationBuffer));

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
            // Create RELAX denoising parameters structure
            RELAXDenoisingParams relaxParams = {};
            relaxParams.resetHistory = (frameNum == 0);
            relaxParams.frameIndex = frameNum;
            relaxParams.framerateScale = 1.0f; // Default framerate scale
            relaxParams.roughnessFraction = denoisingParams.roughnessFraction;
            relaxParams.strandMaterialID = 255; // No strand material by default
            relaxParams.strandThickness = 1.0f;
            relaxParams.cameraAttachedReflectionMaterialID = 254; // No camera attached reflections by default
            relaxParams.specVarianceBoost = 1.0f;                 // Default variance boost
            relaxParams.hasHistoryConfidence = false;             // Disable for now until confidence buffers are properly integrated
            relaxParams.hasDisocclusionThresholdMix = false;      // Disable for now

            TemporalAccumulation<8, 1> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                bufferDim,
                invBufferDim,

                bufferManager.GetBuffer2D(PrevDepthBuffer),
                bufferManager.GetBuffer2D(PrevMaterialBuffer),
                bufferManager.GetBuffer2D(PrevDiffuseIlluminationBuffer),
                bufferManager.GetBuffer2D(PrevDiffuseFastIlluminationBuffer),
                bufferManager.GetBuffer2D(PrevSpecularIlluminationBuffer),
                bufferManager.GetBuffer2D(PrevSpecularFastIlluminationBuffer),
                bufferManager.GetBuffer2D(PrevSpecularHitDistBuffer),
                bufferManager.GetBuffer2D(PrevHistoryLengthBuffer),
                bufferManager.GetBuffer2D(PrevNormalRoughnessBuffer),

                bufferManager.GetBuffer2D(DiffuseIlluminationBuffer),
                bufferManager.GetBuffer2D(SpecularIlluminationBuffer),
                bufferManager.GetBuffer2D(DepthBuffer),
                bufferManager.GetBuffer2D(NormalRoughnessBuffer),
                bufferManager.GetBuffer2D(MaterialBuffer),

                bufferManager.GetBuffer2D(DiffuseIlluminationPingBuffer),
                bufferManager.GetBuffer2D(DiffuseIlluminationPongBuffer),
                bufferManager.GetBuffer2D(SpecularIlluminationPingBuffer),
                bufferManager.GetBuffer2D(SpecularIlluminationPongBuffer),
                bufferManager.GetBuffer2D(SpecularHitDistBuffer),
                bufferManager.GetBuffer2D(SpecularReprojectionConfidenceBuffer),
                bufferManager.GetBuffer2D(HistoryLengthBuffer),

                // History confidence buffers
                bufferManager.GetBuffer2D(DiffuseHistoryConfidenceBuffer),
                bufferManager.GetBuffer2D(PrevDiffuseHistoryConfidenceBuffer),
                bufferManager.GetBuffer2D(SpecularHistoryConfidenceBuffer),
                bufferManager.GetBuffer2D(PrevSpecularHistoryConfidenceBuffer),
                bufferManager.GetBuffer2D(DepthBuffer), // disocclusionThresholdMixBuffer (placeholder)

                bufferManager.GetBuffer2D(DebugBuffer),

                camera,
                historyCamera,

                // Pass denoising parameters to kernel
                denoisingParams.denoisingRange,
                denoisingParams.disocclusionThreshold,
                denoisingParams.disocclusionThresholdAlternate,
                denoisingParams.maxAccumulatedFrameNum,     // diffuse max
                denoisingParams.maxFastAccumulatedFrameNum, // diffuse fast max
                denoisingParams.maxAccumulatedFrameNum,     // specular max (same for now)
                denoisingParams.maxFastAccumulatedFrameNum, // specular fast max (same for now)
                1.0f,                                       // specular variance boost
                denoisingParams.roughnessFraction,
                relaxParams);

            finalResultBuffer = 1; // Temporal accumulation outputs to IlluminationPingBuffer

            // Anti-firefly pass (controlled by parameter)
            if (denoisingParams.enableAntiFirefly)
            {
                AntiFirefly KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                    bufferDim,
                    invBufferDim,

                    bufferManager.GetBuffer2D(DiffuseIlluminationPingBuffer),  // diffuse input
                    bufferManager.GetBuffer2D(SpecularIlluminationPingBuffer), // specular input (same for now)

                    bufferManager.GetBuffer2D(NormalRoughnessBuffer),
                    bufferManager.GetBuffer2D(MaterialBuffer),
                    bufferManager.GetBuffer2D(DepthBuffer),

                    bufferManager.GetBuffer2D(DiffuseIlluminationPongBuffer),  // diffuse output
                    bufferManager.GetBuffer2D(SpecularIlluminationPongBuffer), // specular output (same for now)

                    camera,

                    denoisingParams.denoisingRange,
                    15);               // maxSamples
                finalResultBuffer = 2; // Anti-firefly outputs to IlluminationPongBuffer
            }

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
                    denoisingParams.enableTemporalConfidenceFiltering,
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

            // History fix pass
            if (denoisingParams.enableHistoryFix)
            {
                HistoryFix KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                    bufferDim,
                    invBufferDim,

                    bufferManager.GetBuffer2D(DepthBuffer),
                    bufferManager.GetBuffer2D(MaterialBuffer),
                    bufferManager.GetBuffer2D(NormalRoughnessBuffer),
                    bufferManager.GetBuffer2D(HistoryLengthBuffer),

                    // Separate diffuse buffers
                    bufferManager.GetBuffer2D(DiffuseIlluminationPingBuffer),
                    bufferManager.GetBuffer2D(DiffuseIlluminationPongBuffer),

                    // Separate specular buffers
                    bufferManager.GetBuffer2D(SpecularIlluminationPingBuffer),
                    bufferManager.GetBuffer2D(SpecularIlluminationPongBuffer),
                    bufferManager.GetBuffer2D(SpecularHitDistBuffer),

                    camera,

                    // RELAX parameters
                    8.0f,                                              // historyFixEdgeStoppingNormalPower (reasonable default)
                    (float)denoisingParams.historyFixBasePixelStride); // convert stride parameter
                finalResultBuffer = 2;
            }

            // If history clamping is disabled, we still need to copy the buffers to Prev for next frame
            if (!denoisingParams.enableHistoryClamping)
            {
                // Copy the denoised specular buffers to prev buffers for next frame's temporal accumulation
                // The source depends on which pass was last (stored in finalResultBuffer)
                if (finalResultBuffer == 1)
                {
                    // From Ping buffers
                    BufferCopyFloat4(
                        bufferDim,
                        bufferManager.GetBuffer2D(DiffuseIlluminationPingBuffer),
                        bufferManager.GetBuffer2D(PrevDiffuseIlluminationBuffer));
                    BufferCopyFloat4(
                        bufferDim,
                        bufferManager.GetBuffer2D(DiffuseIlluminationPingBuffer),
                        bufferManager.GetBuffer2D(PrevDiffuseFastIlluminationBuffer));
                    BufferCopyFloat4(
                        bufferDim,
                        bufferManager.GetBuffer2D(SpecularIlluminationPingBuffer),
                        bufferManager.GetBuffer2D(PrevSpecularIlluminationBuffer));
                    BufferCopyFloat4(
                        bufferDim,
                        bufferManager.GetBuffer2D(SpecularIlluminationPingBuffer),
                        bufferManager.GetBuffer2D(PrevSpecularFastIlluminationBuffer));
                }
                else if (finalResultBuffer == 2)
                {
                    // From Pong buffers
                    BufferCopyFloat4(
                        bufferDim,
                        bufferManager.GetBuffer2D(DiffuseIlluminationPongBuffer),
                        bufferManager.GetBuffer2D(PrevDiffuseIlluminationBuffer));
                    BufferCopyFloat4(
                        bufferDim,
                        bufferManager.GetBuffer2D(DiffuseIlluminationPongBuffer),
                        bufferManager.GetBuffer2D(PrevDiffuseFastIlluminationBuffer));
                    BufferCopyFloat4(
                        bufferDim,
                        bufferManager.GetBuffer2D(SpecularIlluminationPongBuffer),
                        bufferManager.GetBuffer2D(PrevSpecularIlluminationBuffer));
                    BufferCopyFloat4(
                        bufferDim,
                        bufferManager.GetBuffer2D(SpecularIlluminationPongBuffer),
                        bufferManager.GetBuffer2D(PrevSpecularFastIlluminationBuffer));
                }

                BufferCopyFloat1(
                    bufferDim,
                    bufferManager.GetBuffer2D(HistoryLengthBuffer),
                    bufferManager.GetBuffer2D(PrevHistoryLengthBuffer));
            }

            // History clamping pass
            if (denoisingParams.enableHistoryClamping)
            {
                HistoryClamping<8, 2> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                    bufferDim,
                    invBufferDim,

                    bufferManager.GetBuffer2D(DepthBuffer),
                    bufferManager.GetBuffer2D(HistoryLengthBuffer),

                    // Separate diffuse buffers
                    bufferManager.GetBuffer2D(DiffuseIlluminationBuffer),
                    bufferManager.GetBuffer2D(DiffuseIlluminationPingBuffer),
                    bufferManager.GetBuffer2D(DiffuseIlluminationPongBuffer),
                    bufferManager.GetBuffer2D(PrevDiffuseIlluminationBuffer),
                    bufferManager.GetBuffer2D(PrevDiffuseFastIlluminationBuffer),

                    // Separate specular buffers
                    bufferManager.GetBuffer2D(SpecularIlluminationBuffer),
                    bufferManager.GetBuffer2D(SpecularIlluminationPingBuffer),
                    bufferManager.GetBuffer2D(SpecularIlluminationPongBuffer),
                    bufferManager.GetBuffer2D(PrevSpecularIlluminationBuffer),
                    bufferManager.GetBuffer2D(PrevSpecularFastIlluminationBuffer),
                    bufferManager.GetBuffer2D(SpecularHitDistBuffer),
                    bufferManager.GetBuffer2D(PrevSpecularHitDistBuffer),

                    bufferManager.GetBuffer2D(PrevHistoryLengthBuffer),

                    // RELAX clamping parameters
                    denoisingParams.historyClampingColorBoxSigmaScale,
                    denoisingParams.antilagAccelerationAmount,
                    denoisingParams.antilagResetAmount);
                finalResultBuffer = 3;
            }
        }
    }
    else if (denoisingParams.enableDenoiser)
    {
        // Temporal accumulation is disabled - copy original buffers to Ping buffers for downstream passes
        BufferCopyFloat4(
            bufferDim,
            bufferManager.GetBuffer2D(DiffuseIlluminationBuffer),
            bufferManager.GetBuffer2D(DiffuseIlluminationPingBuffer));

        BufferCopyFloat4(
            bufferDim,
            bufferManager.GetBuffer2D(SpecularIlluminationBuffer),
            bufferManager.GetBuffer2D(SpecularIlluminationPingBuffer));

        finalResultBuffer = 1; // Data is now in Ping buffers
    }

    // Spatial filtering (A-trous) pass using shared memory optimization
    if (denoisingParams.enableDenoiser && denoisingParams.enableSpatialFiltering)
    {
        AtrousSmem<8, 2> KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
            bufferDim,
            invBufferDim,

            bufferManager.GetBuffer2D(DiffuseIlluminationPingBuffer),
            bufferManager.GetBuffer2D(SpecularIlluminationPingBuffer),

            bufferManager.GetBuffer2D(NormalRoughnessBuffer),
            bufferManager.GetBuffer2D(MaterialBuffer),
            bufferManager.GetBuffer2D(DepthBuffer),
            bufferManager.GetBuffer2D(HistoryLengthBuffer),

            bufferManager.GetBuffer2D(DiffuseIlluminationPongBuffer),
            bufferManager.GetBuffer2D(SpecularIlluminationPongBuffer),

            camera,

            denoisingParams.phiLuminance,
            denoisingParams.phiLuminance, // specular phi luminance
            denoisingParams.depthThreshold,
            denoisingParams.roughnessFraction,
            denoisingParams.lobeAngleFraction,
            denoisingParams.lobeAngleFraction); // specular lobe angle fraction

        finalResultBuffer = 1;

        if (denoisingParams.atrousIterationNum > 0)
        {
            int atrousIndex = 1;
            int atrousStep = 1 << atrousIndex;
            int maxIterations = denoisingParams.atrousIterationNum * 2;

            while (atrousIndex < maxIterations)
            {
                Atrous KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                    bufferDim,
                    invBufferDim,

                    bufferManager.GetBuffer2D(DiffuseIlluminationPingBuffer),  // diffuse input
                    bufferManager.GetBuffer2D(SpecularIlluminationPingBuffer), // specular input

                    bufferManager.GetBuffer2D(NormalRoughnessBuffer),
                    bufferManager.GetBuffer2D(MaterialBuffer),
                    bufferManager.GetBuffer2D(DepthBuffer),
                    bufferManager.GetBuffer2D(HistoryLengthBuffer),

                    bufferManager.GetBuffer2D(DiffuseIlluminationPongBuffer),  // diffuse output
                    bufferManager.GetBuffer2D(SpecularIlluminationPongBuffer), // specular output

                    // Confidence buffers
                    bufferManager.GetBuffer2D(DiffuseConfidenceBuffer),
                    bufferManager.GetBuffer2D(SpecularConfidenceBuffer),
                    bufferManager.GetBuffer2D(SpecularReprojectionConfidenceBuffer), // specular reprojection confidence

                    camera,
                    iterationIndex,
                    atrousStep,

                    // Pass denoising parameters to kernel
                    denoisingParams.phiLuminance,
                    denoisingParams.phiLuminance, // specular phi luminance
                    denoisingParams.depthThreshold,
                    denoisingParams.roughnessFraction,
                    denoisingParams.lobeAngleFraction, // diffuse lobe angle
                    denoisingParams.lobeAngleFraction, // specular lobe angle
                    0.1f,                              // specular lobe angle slack

                    // Confidence parameters
                    denoisingParams.enableConfidenceComputation,
                    denoisingParams.confidenceDrivenRelaxationMultiplier,
                    denoisingParams.confidenceDrivenLuminanceEdgeStoppingRelaxation,
                    0.9f); // normal edge stopping relaxation

                finalResultBuffer = 2; // First A-trous outputs to IlluminationPongBuffer
                ++atrousIndex;
                atrousStep = 1 << atrousIndex;

                Atrous KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                    bufferDim,
                    invBufferDim,

                    bufferManager.GetBuffer2D(DiffuseIlluminationPongBuffer),  // diffuse input
                    bufferManager.GetBuffer2D(SpecularIlluminationPongBuffer), // specular input

                    bufferManager.GetBuffer2D(NormalRoughnessBuffer),
                    bufferManager.GetBuffer2D(MaterialBuffer),
                    bufferManager.GetBuffer2D(DepthBuffer),
                    bufferManager.GetBuffer2D(HistoryLengthBuffer),

                    bufferManager.GetBuffer2D(DiffuseIlluminationPingBuffer),  // diffuse output
                    bufferManager.GetBuffer2D(SpecularIlluminationPingBuffer), // specular output

                    // Confidence buffers
                    bufferManager.GetBuffer2D(DiffuseConfidenceBuffer),
                    bufferManager.GetBuffer2D(SpecularConfidenceBuffer),
                    bufferManager.GetBuffer2D(SpecularReprojectionConfidenceBuffer), // specular reprojection confidence

                    camera,
                    iterationIndex,
                    atrousStep,

                    // Pass denoising parameters to kernel
                    denoisingParams.phiLuminance,
                    denoisingParams.phiLuminance, // specular phi luminance
                    denoisingParams.depthThreshold,
                    denoisingParams.roughnessFraction,
                    denoisingParams.lobeAngleFraction, // diffuse lobe angle
                    denoisingParams.lobeAngleFraction, // specular lobe angle
                    0.1f,                              // specular lobe angle slack

                    // Confidence parameters
                    denoisingParams.enableConfidenceComputation,
                    denoisingParams.confidenceDrivenRelaxationMultiplier,
                    denoisingParams.confidenceDrivenLuminanceEdgeStoppingRelaxation,
                    0.9f); // normal edge stopping relaxation

                finalResultBuffer = 1; // Second A-trous outputs back to IlluminationPingBuffer
                ++atrousIndex;
                atrousStep = 1 << atrousIndex;
            }

            Atrous KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
                bufferDim,
                invBufferDim,

                bufferManager.GetBuffer2D(IlluminationPingBuffer), // diffuse input
                bufferManager.GetBuffer2D(IlluminationPingBuffer), // specular input

                bufferManager.GetBuffer2D(NormalRoughnessBuffer),
                bufferManager.GetBuffer2D(MaterialBuffer),
                bufferManager.GetBuffer2D(DepthBuffer),
                bufferManager.GetBuffer2D(HistoryLengthBuffer),

                bufferManager.GetBuffer2D(IlluminationPongBuffer), // diffuse output
                bufferManager.GetBuffer2D(IlluminationPongBuffer), // specular output

                // Confidence buffers
                bufferManager.GetBuffer2D(DiffuseConfidenceBuffer),
                bufferManager.GetBuffer2D(SpecularConfidenceBuffer),
                bufferManager.GetBuffer2D(SpecularConfidenceBuffer), // specular reprojection confidence

                camera,
                iterationIndex,
                atrousStep,

                // Pass denoising parameters to kernel
                denoisingParams.phiLuminance,
                denoisingParams.phiLuminance, // specular phi luminance
                denoisingParams.depthThreshold,
                denoisingParams.roughnessFraction,
                denoisingParams.lobeAngleFraction, // diffuse lobe angle
                denoisingParams.lobeAngleFraction, // specular lobe angle
                0.1f,                              // specular lobe angle slack

                // Confidence parameters
                denoisingParams.enableConfidenceComputation,
                denoisingParams.confidenceDrivenRelaxationMultiplier,
                denoisingParams.confidenceDrivenLuminanceEdgeStoppingRelaxation,
                0.9f); // normal edge stopping relaxation

            finalResultBuffer = 2;
        }
    }
    // If spatial filtering is disabled but denoiser is enabled, data remains in Ping buffers
    else if (denoisingParams.enableDenoiser && finalResultBuffer == 0)
    {
        // This shouldn't happen with our temporal accumulation fix, but safety check
        finalResultBuffer = 1;
    }

    // Combine separate diffuse and specular results back into IlluminationPingBuffer
    SurfObj diffuseSource, specularSource;
    if (finalResultBuffer == 0)
    {
        diffuseSource = bufferManager.GetBuffer2D(DiffuseIlluminationBuffer);
        specularSource = bufferManager.GetBuffer2D(SpecularIlluminationBuffer);
    }
    else if (finalResultBuffer == 1)
    {
        // Results are in Ping buffers (from temporal accumulation or default)
        diffuseSource = bufferManager.GetBuffer2D(DiffuseIlluminationPingBuffer);
        specularSource = bufferManager.GetBuffer2D(SpecularIlluminationPingBuffer);
    }
    else if (finalResultBuffer == 2)
    {
        // Results are in Pong buffers (from anti-firefly or A-trous passes)
        diffuseSource = bufferManager.GetBuffer2D(DiffuseIlluminationPongBuffer);
        specularSource = bufferManager.GetBuffer2D(SpecularIlluminationPongBuffer);
    }
    else if (finalResultBuffer == 3)
    {
        // Results are in Prev buffers (from history clamping)
        diffuseSource = bufferManager.GetBuffer2D(PrevDiffuseIlluminationBuffer);
        specularSource = bufferManager.GetBuffer2D(PrevSpecularIlluminationBuffer);
    }

    CombineDiffuseSpecular KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
        bufferDim,
        diffuseSource,
        specularSource,
        bufferManager.GetBuffer2D(IlluminationPingBuffer)); // Combined output

    // Final output is always IlluminationPingBuffer since we just combined results there
    SurfObj finalOutputBuffer = bufferManager.GetBuffer2D(IlluminationPingBuffer);

    BufferCopyNonSky KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
        bufferDim,
        finalOutputBuffer,
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

    // Copy specular hit distance buffer for next frame - CRITICAL for temporal accumulation!
    BufferCopyFloat1(
        bufferDim,
        bufferManager.GetBuffer2D(SpecularHitDistBuffer),
        bufferManager.GetBuffer2D(PrevSpecularHitDistBuffer));

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