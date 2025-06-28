#include "core/OfflineBackend.h"
#include "util/DebugUtils.h"
#include "core/OptixRenderer.h"
#include "postprocessing/PostProcessor.h"
#include "denoising/Denoiser.h"
#include "sky/Sky.h"
#include "core/RenderCamera.h"
#include "voxelengine/VoxelEngine.h"
#include "core/GlobalSettings.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

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

    m_timer.update();

    SkyModel::Get().update();

    renderer.update();
    voxelengine.update();

    renderer.render();

    denoiser.run(renderer.getWidth(), renderer.getHeight(), renderer.getWidth(), renderer.getHeight());

    postProcessor.run(m_frameBuffer, renderer.getWidth(), renderer.getHeight(), m_width, m_height);

    // Write frame buffer to PNG using stb_image_write
    writeFrameBufferToPNG(outputPath);

    m_frameNum++;
    m_accumulationCounter++;

    float deltaTime = m_timer.getDeltaTime();
    m_currentFPS = 1000.0f / deltaTime;

    std::cout << "Frame " << m_frameNum << " rendered. Time: " << deltaTime << "ms, FPS: " << m_currentFPS << std::endl;
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

void OfflineBackend::writeFrameBufferToPNG(const std::string &outputPath)
{
    // Copy frame buffer from GPU to CPU
    const size_t bufferSize = m_width * m_height * sizeof(Float4);
    std::vector<Float4> hostBuffer(m_width * m_height);

    CUDA_CHECK(cudaMemcpy(hostBuffer.data(), m_frameBuffer, bufferSize, cudaMemcpyDeviceToHost));

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

            // Simple gamma correction
            r = std::pow(r, 1.0f / 2.2f);
            g = std::pow(g, 1.0f / 2.2f);
            b = std::pow(b, 1.0f / 2.2f);

            imageData[dstIdx + 0] = (unsigned char)(r * 255.0f);
            imageData[dstIdx + 1] = (unsigned char)(g * 255.0f);
            imageData[dstIdx + 2] = (unsigned char)(b * 255.0f);
        }
    }

    // Use stb_image_write to save PNG
    if (stbi_write_png(outputPath.c_str(), m_width, m_height, 3, imageData.data(), m_width * 3))
    {
        std::cout << "Successfully saved image to: " << outputPath << std::endl;
    }
    else
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