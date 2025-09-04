#include "core/OfflineBackend.h"
#include "util/DebugUtils.h"
#include "core/OptixRenderer.h"
#include "postprocessing/PostProcessor.h"
#include "denoising/Denoiser.h"
#include "sky/Sky.h"
#include "core/RenderCamera.h"
#include "voxelengine/VoxelEngine.h"
#include "core/GlobalSettings.h"
#include "util/PerformanceTracker.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <execution>
#include <thread>
#include <future>
#include <atomic>

#ifdef OFFLINE_MODE
#include "stb/stb_image_write.h"
#endif

void OfflineBackend::init(int width, int height)
{
    m_width = width;
    m_height = height;

    std::cout << "Initializing offline backend with resolution: " << m_width << "x" << m_height << std::endl;

    dumpSystemInformation();
    initCuda();
    initFrameBuffer();

    SkyModel::Get().init();

    auto &renderer = OptixRenderer::Get();
    renderer.setWidth(m_width);
    renderer.setHeight(m_height);

    m_timer.init();
}

void OfflineBackend::renderFrame(const std::string &outputPath)
{
    auto &renderer = OptixRenderer::Get();
    auto &postProcessor = PostProcessor::Get();
    auto &denoiser = Denoiser::Get();
    auto &voxelengine = VoxelEngine::Get();
    auto &perfTracker = PerformanceTracker::Get();

    m_timer.update();

    // Scene preparation timing
    perfTracker.startTiming("scenePrep");
    SkyModel::Get().update();
    perfTracker.setScenePreparationTime(perfTracker.endTiming("scenePrep"));

    // Renderer update timing
    perfTracker.startTiming("rendererUpdate");
    renderer.update();
    voxelengine.update(m_timer.getDeltaTime());
    perfTracker.setRendererUpdateTime(perfTracker.endTiming("rendererUpdate"));

    // Path tracing timing
    perfTracker.startTiming("pathTracing");
    renderer.render();
    perfTracker.setPathTracingTime(perfTracker.endTiming("pathTracing"));

    // Denoiser timing
    perfTracker.startTiming("denoiser");
    denoiser.run(renderer.getWidth(), renderer.getHeight(), renderer.getWidth(), renderer.getHeight());
    perfTracker.setDenoiserTime(perfTracker.endTiming("denoiser"));

    // Post processing timing
    perfTracker.startTiming("postProcessing");
    postProcessor.run(m_frameBuffer, renderer.getWidth(), renderer.getHeight(), m_width, m_height, m_timer.getDeltaTime());
    perfTracker.setPostProcessingTime(perfTracker.endTiming("postProcessing"));

    // Only store frame in batch if outputPath is provided (not empty)
    if (!outputPath.empty())
    {
        storeFrameInBatch(outputPath);
    }

    m_frameNum++;
}

void OfflineBackend::clear()
{
    CUDA_CHECK(cudaStreamSynchronize(m_cudaStream));
    CUDA_CHECK(cudaStreamDestroy(m_cudaStream));

    if (m_frameBuffer)
    {
        CUDA_CHECK(cudaFree(m_frameBuffer));
        m_frameBuffer = nullptr;
    }
}

void OfflineBackend::initCuda()
{
    CUDA_CHECK(cudaFree(0));
    cuCtxGetCurrent(&m_cudaContext);
    CUDA_CHECK(cudaStreamCreate(&m_cudaStream));
}

void OfflineBackend::initFrameBuffer()
{
    const size_t bufferSize = m_width * m_height * sizeof(Float4);
    CUDA_CHECK(cudaMalloc((void**)&m_frameBuffer, bufferSize));
    CUDA_CHECK(cudaMemset(m_frameBuffer, 0, bufferSize));
}

void OfflineBackend::storeFrameInBatch(const std::string &outputPath)
{
    // Copy frame buffer from GPU to CPU and store in batch
    const size_t bufferSize = m_width * m_height * sizeof(Float4);
    std::vector<Float4> hostBuffer(m_width * m_height);

    CUDA_CHECK(cudaMemcpy(hostBuffer.data(), m_frameBuffer, bufferSize, cudaMemcpyDeviceToHost));

    // Store the frame data for later batch writing
    BatchedFrame frame;
    frame.hostBuffer = std::move(hostBuffer);
    frame.outputPath = outputPath;

    m_batchedFrames.push_back(std::move(frame));
}

void OfflineBackend::writeAllBatchedFrames()
{
    std::cout << "Writing " << m_batchedFrames.size() << " batched frames in parallel..." << std::endl;

    // Get the number of CPU threads to use
    const unsigned int numThreads = std::thread::hardware_concurrency();
    std::cout << "Using " << numThreads << " CPU threads for parallel PNG writing" << std::endl;

    // Use parallel execution to write PNG files
    std::atomic<size_t> completedFrames{0};

    try {
        // Try to use C++17 parallel algorithms first
        std::for_each(std::execution::par, m_batchedFrames.begin(), m_batchedFrames.end(),
            [this, &completedFrames](const BatchedFrame &frame) {
                writeFrameBufferToPNG(frame.hostBuffer, frame.outputPath);
                completedFrames.fetch_add(1);
            });
    }
    catch (const std::exception&) {
        // Fallback to manual threading if parallel algorithms not available
        std::cout << "Parallel algorithms not available, using manual threading..." << std::endl;

        const size_t numFrames = m_batchedFrames.size();
        const size_t framesPerThread = std::max(size_t(1), numFrames / numThreads);

        std::vector<std::future<void>> futures;
        futures.reserve(numThreads);

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            size_t startIdx = threadId * framesPerThread;
            size_t endIdx = (threadId == numThreads - 1) ? numFrames : std::min(startIdx + framesPerThread, numFrames);

            if (startIdx >= numFrames) break;

            futures.emplace_back(std::async(std::launch::async, [this, startIdx, endIdx, &completedFrames]() {
                for (size_t i = startIdx; i < endIdx; ++i) {
                    const auto &frame = m_batchedFrames[i];
                    writeFrameBufferToPNG(frame.hostBuffer, frame.outputPath);
                    completedFrames.fetch_add(1);
                }
            }));
        }

        // Wait for all threads to complete
        for (auto &future : futures) {
            future.wait();
        }
    }

    std::cout << "Successfully written all " << m_batchedFrames.size() << " frames in parallel" << std::endl;
}

void OfflineBackend::clearBatchedFrames()
{
    m_batchedFrames.clear();
}

void OfflineBackend::writeFrameBufferToPNG(const std::vector<Float4> &hostBuffer, const std::string &outputPath)
{
    // Convert Float4 to RGB bytes
    std::vector<unsigned char> imageData(m_width * m_height * 3);

    for (int y = 0; y < m_height; y++)
    {
        for (int x = 0; x < m_width; x++)
        {
            int srcIdx = y * m_width + x;
            int dstIdx = ((m_height - 1 - y) * m_width + x) * 3;  // Flip Y for proper image orientation

            Float4 pixel = hostBuffer[srcIdx];

            // Tone map and gamma correct
            float r = std::min(1.0f, std::max(0.0f, pixel.x));
            float g = std::min(1.0f, std::max(0.0f, pixel.y));
            float b = std::min(1.0f, std::max(0.0f, pixel.z));

            imageData[dstIdx + 0] = (unsigned char)(r * 255.0f);
            imageData[dstIdx + 1] = (unsigned char)(g * 255.0f);
            imageData[dstIdx + 2] = (unsigned char)(b * 255.0f);
        }
    }

    // Use stb_image_write to save PNG (no per-frame logging)
    if (!stbi_write_png(outputPath.c_str(), m_width, m_height, 3, imageData.data(), m_width * 3))
    {
        std::cerr << "Failed to save image to: " << outputPath << std::endl;
    }
}

void OfflineBackend::dumpSystemInformation()
{
    int versionDriver = 0;
    CUDA_CHECK(cudaDriverGetVersion(&versionDriver));

    // The version is returned as (1000 * major + 10 * minor).
    int major = versionDriver / 1000;
    int minor = (versionDriver - major * 1000) / 10;
    std::cout << "Driver Version  = " << major << "." << minor << '\n';

    int versionRuntime = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&versionRuntime));

    // The version is returned as (1000 * major + 10 * minor).
    major = versionRuntime / 1000;
    minor = (versionRuntime - major * 1000) / 10;
    std::cout << "Runtime Version = " << major << "." << minor << '\n';

    int countDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&countDevices));
    std::cout << "Device Count    = " << countDevices << '\n';

    for (int i = 0; i < countDevices; ++i)
    {
        cudaDeviceProp properties;

        CUDA_CHECK(cudaGetDeviceProperties(&properties, i));

        m_deviceProperties.push_back(properties);

        std::cout << "Device " << i << ": " << properties.name << '\n';
        std::cout << "  SM " << properties.major << "." << properties.minor << '\n';
        std::cout << "  Total Mem = " << properties.totalGlobalMem << '\n';
        std::cout << "  ClockRate [kHz] = " << properties.clockRate << '\n';
        std::cout << "  MaxThreadsPerBlock = " << properties.maxThreadsPerBlock << '\n';
        std::cout << "  SM Count = " << properties.multiProcessorCount << '\n';
        std::cout << "  Timeout Enabled = " << properties.kernelExecTimeoutEnabled << '\n';
        std::cout << "  TCC Driver = " << properties.tccDriver << '\n';
    }
}