#include "postprocessing/PostProcessingPipeline.h"
#include "util/KernelHelper.h"
#include "util/DebugUtils.h"
#include "core/BufferManager.h"
#include "core/RenderCamera.h"
#include "sky/Sky.h"
#include "shaders/Common.h"
#include "shaders/Sampler.h"
#include <cuda_runtime.h>

// Bloom bright pass extraction kernel with neighbor filtering
__global__ void BloomExtractBrightPixelsKernel(
    SurfObj inputBuffer,
    SurfObj outputBuffer,
    Int2 size,
    float threshold,
    bool useNeighborFilter)
{
    Int2 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx.x >= size.x || idx.y >= size.y)
        return;

    Float3 color = Load2DFloat4(inputBuffer, idx).xyz;

    // Compute luminance
    float luminance = dot(color, Float3(0.2126f, 0.7152f, 0.0722f));

    // Extract bright pixels above threshold
    if (luminance > threshold)
    {
        bool isValidBloomPixel = true;

        // Neighbor filtering to reduce fireflies in bloom
        if (useNeighborFilter)
        {
            float maxNeighborLum = 0.0f;

            // Check immediate neighbors (+ pattern)
            Int2 neighbors[4] = {
                Int2(idx.x - 1, idx.y), Int2(idx.x + 1, idx.y),
                Int2(idx.x, idx.y - 1), Int2(idx.x, idx.y + 1)};

            for (int i = 0; i < 4; i++)
            {
                if (neighbors[i].x >= 0 && neighbors[i].x < size.x &&
                    neighbors[i].y >= 0 && neighbors[i].y < size.y)
                {
                    Float3 neighborColor = Load2DFloat4(inputBuffer, neighbors[i]).xyz;
                    float neighborLum = dot(neighborColor, Float3(0.2126f, 0.7152f, 0.0722f));
                    maxNeighborLum = fmaxf(maxNeighborLum, neighborLum);
                }
            }

            // Require at least one neighbor to be somewhat bright
            if (maxNeighborLum < threshold * 0.4f)
            {
                isValidBloomPixel = false;
            }
        }

        if (isValidBloomPixel)
        {
            // Preserve color while scaling down by threshold (reduced intensity)
            color = clamp3f((color - Float3(threshold)) * 0.7f, Float3(0.0f), Float3(100.0f));
        }
        else
        {
            color = Float3(0.0f);
        }
    }
    else
    {
        color = Float3(0.0f);
    }

    Store2DFloat4(Float4(color, 1.0f), outputBuffer, idx);
}

// Gaussian blur kernel for bloom
__global__ void BloomBlurKernel(
    SurfObj inputBuffer,
    SurfObj outputBuffer,
    Int2 size,
    Int2 direction, // (1,0) for horizontal, (0,1) for vertical
    float radius)
{
    Int2 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx.x >= size.x || idx.y >= size.y)
        return;

    Float3 result = Float3(0.0f);
    float totalWeight = 0.0f;

    // Simple box blur with radius
    int kernelSize = (int)(radius * 2.0f) + 1;
    int halfKernel = kernelSize / 2;

    for (int i = -halfKernel; i <= halfKernel; i++)
    {
        Int2 samplePos = idx + direction * i;

        // Clamp to buffer bounds
        samplePos.x = clampf(samplePos.x, 0, size.x - 1);
        samplePos.y = clampf(samplePos.y, 0, size.y - 1);

        Float3 sample = Load2DFloat4(inputBuffer, samplePos).xyz;
        float weight = 1.0f; // Simple box filter

        result += sample * weight;
        totalWeight += weight;
    }

    if (totalWeight > 0.0f)
    {
        result /= totalWeight;
    }

    Store2DFloat4(Float4(result, 1.0f), outputBuffer, idx);
}

// Bloom composite kernel
__global__ void BloomCompositeKernel(
    SurfObj colorBuffer, // Original scene color (modified in-place)
    SurfObj bloomBuffer, // Blurred bloom texture
    Int2 size,
    float intensity)
{
    Int2 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx.x >= size.x || idx.y >= size.y)
        return;

    Float3 originalColor = Load2DFloat4(colorBuffer, idx).xyz;
    Float3 bloomColor = Load2DFloat4(bloomBuffer, idx).xyz;

    // Additive blending with intensity control
    Float3 finalColor = originalColor + bloomColor * intensity;

    Store2DFloat4(Float4(finalColor, 1.0f), colorBuffer, idx);
}

// Vignette effect kernel
__global__ void VignetteKernel(
    SurfObj colorBuffer,
    Int2 size,
    float strength,
    float radius,
    float smoothness)
{
    Int2 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx.x >= size.x || idx.y >= size.y)
        return;

    Float3 color = Load2DFloat4(colorBuffer, idx).xyz;

    // Calculate normalized coordinates from center (-1 to 1)
    float x = (2.0f * idx.x - size.x) / (float)size.x;
    float y = (2.0f * idx.y - size.y) / (float)size.y;

    // Calculate distance from center
    float distance = sqrtf(x * x + y * y);

    // Apply vignette falloff using manual smoothstep
    float t = clampf((distance - radius) / smoothness, 0.0f, 1.0f);
    float smoothed = t * t * (3.0f - 2.0f * t); // smoothstep interpolation
    float vignette = 1.0f - smoothed;
    vignette = 1.0f - strength * (1.0f - vignette);
    vignette = clampf(vignette, 0.0f, 1.0f);

    // Apply vignette to color
    color *= vignette;

    Store2DFloat4(Float4(color, 1.0f), colorBuffer, idx);
}

static bool ProjectSunToScreen(const Camera &camera, const Float3 &sunDir, Int2 size, Float2 &outUV, Int2 &outPixel)
{
    Float3 uvw = camera.worldToUv * sunDir;
    if (uvw.z <= 0.0f)
        return false;

    Float2 uv = Float2(uvw.x, uvw.y) / uvw.z;
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f)
        return false;

    auto clampTo = [](int value, int minValue, int maxValue) {
        return value < minValue ? minValue : (value > maxValue ? maxValue : value);
    };

    outUV = uv;
    outPixel = Int2(
        clampTo(static_cast<int>(uv.x * size.x), 0, size.x - 1),
        clampTo(static_cast<int>(uv.y * size.y), 0, size.y - 1));
    return true;
}

static bool IsSunVisible(const Int2 &sunPixel, cudaArray_t depthArray)
{
    float depthValue = RayMax;
    CUDA_CHECK(cudaMemcpy2DFromArray(
        &depthValue,
        sizeof(float),
        depthArray,
        static_cast<size_t>(sunPixel.x) * sizeof(float),
        static_cast<size_t>(sunPixel.y),
        sizeof(float),
        1,
        cudaMemcpyDeviceToHost));
    return depthValue >= RayMaxLowerBound;
}

__global__ void LensFlareKernel(
    SurfObj colorBuffer,
    Int2 size,
    Float2 sunUV,
    float sunLuminance,
    float intensity,
    float ghostSpacing,
    int ghostCount,
    float haloRadius,
    float sunSize,
    float distortion)
{
    Int2 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx.x >= size.x || idx.y >= size.y)
        return;

    Float3 originalColor = Load2DFloat4(colorBuffer, idx).xyz;
    Float3 flareColor = Float3(0.0f);

    Float2 uv = Float2((float)idx.x / size.x, (float)idx.y / size.y);
    Float2 center = Float2(0.5f, 0.5f);

    float aspect = (float)size.x / (float)size.y;
    Float2 uvAspect = Float2(uv.x * aspect, uv.y);
    Float2 sunUVAspect = Float2(sunUV.x * aspect, sunUV.y);
    Float2 centerAspect = Float2(center.x * aspect, center.y);

    Float2 toSunAspect = uvAspect - sunUVAspect;
    float dist = sqrtf(toSunAspect.x * toSunAspect.x + toSunAspect.y * toSunAspect.y);

    Float2 sunToCenterAspect = centerAspect - sunUVAspect;
    float axisDistance = sqrtf(sunToCenterAspect.x * sunToCenterAspect.x + sunToCenterAspect.y * sunToCenterAspect.y);

    float effectiveSunSize = fmaxf(sunSize, 0.0005f);
    float lightIntensity = fmaxf(sunLuminance, 1.0f);

    if (axisDistance > 0.0001f)
    {
        Float2 axisDir = sunToCenterAspect / axisDistance;

        if (dist < effectiveSunSize)
        {
            float sunFalloff = 1.0f - (dist / effectiveSunSize);
            sunFalloff = sunFalloff * sunFalloff;
            flareColor += Float3(1.0f, 0.9f, 0.7f) * sunFalloff * intensity * lightIntensity * 0.1f;
        }

        if (haloRadius > 0.0001f)
        {
            float haloFalloff = expf(-dist * dist / (haloRadius * haloRadius));
            flareColor += Float3(1.0f, 0.8f, 0.6f) * haloFalloff * intensity * lightIntensity * 0.08f;
        }

        for (int g = 1; g <= ghostCount; ++g)
        {
            float ghostFactor = ghostSpacing * static_cast<float>(g);
            float ghostDistance = fminf(ghostFactor, 1.0f) * axisDistance;
            Float2 ghostCenterAspect = sunUVAspect + axisDir * ghostDistance;

            Float2 toGhostAspect = uvAspect - ghostCenterAspect;
            float ghostDist = sqrtf(toGhostAspect.x * toGhostAspect.x + toGhostAspect.y * toGhostAspect.y);

            float ghostSize = 0.02f + (g % 3) * 0.01f;
            float ghostFalloff = expf(-ghostDist * ghostDist / (ghostSize * ghostSize));

            Float3 ghostTint;
            switch (g % 4)
            {
            case 0: ghostTint = Float3(1.0f, 0.7f, 0.3f); break;
            case 1: ghostTint = Float3(0.8f, 1.0f, 0.5f); break;
            case 2: ghostTint = Float3(0.6f, 0.8f, 1.0f); break;
            default: ghostTint = Float3(1.0f, 0.6f, 0.8f); break;
            }

            float ghostIntensity = intensity * lightIntensity * 0.04f * (1.0f - (float)g / fmaxf((float)ghostCount, 1.0f));
            flareColor += ghostTint * ghostFalloff * ghostIntensity;
        }

        if (distortion > 0.0f)
        {
            float aberrationStart = fmaxf(effectiveSunSize * 1.5f, 0.02f);
            float aberrationFade = clampf((dist - aberrationStart) / 0.5f, 0.0f, 1.0f);
            float aberrationStrength = distortion * intensity * lightIntensity * 0.02f;
            float falloff = (1.0f / (1.0f + dist * 6.0f)) * aberrationFade * aberrationFade;
            flareColor += Float3(aberrationStrength, 0.0f, -aberrationStrength) * falloff;
        }
    }

    Float3 finalColor = originalColor + flareColor;
    Store2DFloat4(Float4(finalColor, 1.0f), colorBuffer, idx);
}

// Simple luminance histogram computation kernel
__global__ void ComputeLuminanceHistogramKernel(
    SurfObj colorBuffer,
    Int2 size,
    float *histogram,
    int numBins,
    float minLogLum,
    float maxLogLum)
{
    Int2 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx.x >= size.x || idx.y >= size.y)
        return;

    Float3 color = Load2DFloat4(colorBuffer, idx).xyz;

    // Compute luminance
    float luminance = dot(color, Float3(0.2126f, 0.7152f, 0.0722f));

    // Skip dark pixels
    if (luminance < 0.001f)
        return;

    // Map to histogram bin
    float logLum = log10f(luminance);
    float t = clampf((logLum - minLogLum) / (maxLogLum - minLogLum), 0.0f, 1.0f);
    int bin = min((int)(t * numBins), numBins - 1);

    // Atomic add to histogram
    atomicAdd(&histogram[bin], 1.0f);
}

// Compute average luminance from histogram
__global__ void ComputeAverageLuminanceKernel(
    float *histogram,
    int numBins,
    float minLogLum,
    float maxLogLum,
    float histogramMinPercent,
    float histogramMaxPercent,
    float *avgLuminance)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    // Sum total pixels
    float totalPixels = 0.0f;
    for (int i = 0; i < numBins; i++)
    {
        totalPixels += histogram[i];
    }

    if (totalPixels == 0.0f)
    {
        *avgLuminance = 0.18f; // Default middle gray
        return;
    }

    // Find range based on percentiles
    float minCount = totalPixels * histogramMinPercent / 100.0f;
    float maxCount = totalPixels * histogramMaxPercent / 100.0f;

    float accumCount = 0.0f;
    int minBin = 0, maxBin = numBins - 1;

    // Find min bin
    for (int i = 0; i < numBins; i++)
    {
        accumCount += histogram[i];
        if (accumCount >= minCount)
        {
            minBin = i;
            break;
        }
    }

    // Find max bin
    accumCount = 0.0f;
    for (int i = 0; i < numBins; i++)
    {
        accumCount += histogram[i];
        if (accumCount >= maxCount)
        {
            maxBin = i;
            break;
        }
    }

    // Compute average in range
    float weightedSum = 0.0f;
    float weightTotal = 0.0f;

    for (int i = minBin; i <= maxBin; i++)
    {
        float binCenter = minLogLum + (i + 0.5f) * (maxLogLum - minLogLum) / numBins;
        weightedSum += histogram[i] * binCenter;
        weightTotal += histogram[i];
    }

    if (weightTotal > 0.0f)
    {
        float avgLogLum = weightedSum / weightTotal;
        *avgLuminance = powf(10.0f, avgLogLum);
    }
    else
    {
        *avgLuminance = 0.18f;
    }
}

PostProcessingPipeline::PostProcessingPipeline()
    : d_luminanceHistogram(nullptr)
    , histogramBins(256)
    , m_currentAvgLuminance(0.18f)
    , m_targetAvgLuminance(0.18f)
    , m_lastFrameTime(0.0f)
{
}

PostProcessingPipeline::~PostProcessingPipeline()
{
    if (d_luminanceHistogram)
    {
        cudaFree(d_luminanceHistogram);
        d_luminanceHistogram = nullptr;
    }
}

void PostProcessingPipeline::Initialize(int width, int height)
{
    // Allocate histogram buffer
    if (!d_luminanceHistogram)
    {
        cudaMalloc(&d_luminanceHistogram, histogramBins * sizeof(float));
    }
}

float PostProcessingPipeline::Execute(
    SurfObj colorBuffer,
    Int2 size,
    const PostProcessingPipelineParams &pipelineParams,
    const ToneMappingParams &toneMappingParams,
    float deltaTime)
{
    float computedExposure = toneMappingParams.manualExposure;

    // Auto-exposure calculation
    if (pipelineParams.enableAutoExposure)
    {
        // Clear histogram
        cudaMemset(d_luminanceHistogram, 0, histogramBins * sizeof(float));

        // Compute luminance histogram
        ComputeLuminanceHistogramKernel<<<GetGridDim(size.x, size.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)>>>(
            colorBuffer,
            size,
            d_luminanceHistogram,
            histogramBins,
            -8.0f, // minLogLum
            4.0f   // maxLogLum
        );

        // Compute average luminance
        float *d_avgLum;
        cudaMalloc(&d_avgLum, sizeof(float));

        ComputeAverageLuminanceKernel<<<1, 1>>>(
            d_luminanceHistogram,
            histogramBins,
            -8.0f, // minLogLum
            4.0f,  // maxLogLum
            pipelineParams.histogramMinPercent,
            pipelineParams.histogramMaxPercent,
            d_avgLum);

        float avgLum;
        cudaMemcpy(&avgLum, d_avgLum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_avgLum);

        // Smooth adaptation
        m_targetAvgLuminance = avgLum;
        float adaptSpeed = pipelineParams.exposureSpeed * deltaTime;
        m_currentAvgLuminance = lerp(m_currentAvgLuminance, m_targetAvgLuminance, clampf(adaptSpeed, 0.0f, 1.0f));

        // Compute exposure from average luminance
        float keyValue = pipelineParams.targetLuminance;
        computedExposure = keyValue / max(m_currentAvgLuminance, 0.001f);

        // Apply exposure compensation
        computedExposure *= powf(2.0f, pipelineParams.exposureCompensation);

        // Clamp to min/max exposure
        float minExposure = powf(2.0f, pipelineParams.exposureMin);
        float maxExposure = powf(2.0f, pipelineParams.exposureMax);
        computedExposure = clampf(computedExposure, minExposure, maxExposure);
    }

    // Bloom effect
    if (pipelineParams.enableBloom)
    {
        auto &bufferManager = BufferManager::Get();

        // Extract bright pixels
        BloomExtractBrightPixelsKernel<<<GetGridDim(size.x, size.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)>>>(
            colorBuffer,
            bufferManager.GetBuffer2D(BloomExtractBuffer),
            size,
            pipelineParams.bloomThreshold,
            true);

        // Horizontal blur pass
        BloomBlurKernel<<<GetGridDim(size.x, size.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)>>>(
            bufferManager.GetBuffer2D(BloomExtractBuffer),
            bufferManager.GetBuffer2D(BloomTempBuffer),
            size,
            Int2(1, 0), // Horizontal direction
            pipelineParams.bloomRadius);

        // Vertical blur pass
        BloomBlurKernel<<<GetGridDim(size.x, size.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)>>>(
            bufferManager.GetBuffer2D(BloomTempBuffer),
            bufferManager.GetBuffer2D(BloomExtractBuffer),
            size,
            Int2(0, 1), // Vertical direction
            pipelineParams.bloomRadius);

        // Composite bloom back onto original image
        BloomCompositeKernel<<<GetGridDim(size.x, size.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)>>>(
            colorBuffer,
            bufferManager.GetBuffer2D(BloomExtractBuffer),
            size,
            pipelineParams.bloomIntensity);
    }

    // Lens flare effect (before vignette so it can be darkened by vignette)
    if (pipelineParams.enableLensFlare)
    {
        auto &bufferManager = BufferManager::Get();
        const Camera &camera = RenderCamera::Get().camera;
        const auto &skyModel = SkyModel::Get();
        Float3 sunDir = normalize(skyModel.getSunDir());

        Float2 sunUV;
        Int2 sunPixel;
        if (ProjectSunToScreen(camera, sunDir, size, sunUV, sunPixel))
        {
            cudaArray_t depthArray = bufferManager.GetBufferArray(DepthBuffer);
            if (IsSunVisible(sunPixel, depthArray))
            {
                float sunLuminance = skyModel.getAccumulatedSunLuminance();
                if (!(sunLuminance > 0.0f))
                    sunLuminance = 1.0f;

                LensFlareKernel<<<GetGridDim(size.x, size.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)>>>(
                    colorBuffer,
                    size,
                    sunUV,
                    sunLuminance,
                    pipelineParams.lensFlareIntensity,
                    pipelineParams.lensFlareGhostSpacing,
                    pipelineParams.lensFlareGhostCount,
                    pipelineParams.lensFlareHaloRadius,
                    pipelineParams.lensFlareSunSize,
                    pipelineParams.lensFlareDistortion);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }

    // Vignette effect
    if (pipelineParams.enableVignette)
    {
        VignetteKernel<<<GetGridDim(size.x, size.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1)>>>(
            colorBuffer,
            size,
            pipelineParams.vignetteStrength,
            pipelineParams.vignetteRadius,
            pipelineParams.vignetteSmoothness);
    }

    return computedExposure;
}
