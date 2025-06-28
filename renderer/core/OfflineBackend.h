#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

#include "shaders/LinearMath.h"
#include "util/Timer.h"

#include <vector>
#include <string>

class OfflineBackend
{
public:
    static OfflineBackend &Get()
    {
        static OfflineBackend instance;
        return instance;
    }
    OfflineBackend(OfflineBackend const &) = delete;
    void operator=(OfflineBackend const &) = delete;

    void init(int width = 1920, int height = 1080);
    void clear();
    void renderFrame(const std::string &outputPath = "output.png");

    CUstream getCudaStream() const { return m_cudaStream; }
    CUcontext getCudaContext() const { return m_cudaContext; }
    const Timer &getTimer() const { return m_timer; }
    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }
    int getMaxRenderWidth() const { return m_width; }
    int getMaxRenderHeight() const { return m_height; }
    float getCurrentFPS() const { return m_currentFPS; }
    int getCurrentRenderWidth() const { return m_width; }
    int getFrameNum() const { return m_frameNum; }
    void resetAccumulationCounter() { m_accumulationCounter = 1; }
    int getAccumulationCounter() const { return m_accumulationCounter; }

    // Frame buffer for offline rendering
    Float4 *getFrameBuffer() const { return m_frameBuffer; }

private:
    OfflineBackend() {}

    void dumpSystemInformation();
    void initCuda();
    void initFrameBuffer();
    void writeFrameBufferToPNG(const std::string &outputPath);

    int m_width;
    int m_height;

    // CUDA stuffs
    CUcontext m_cudaContext;
    CUstream m_cudaStream;

    // Frame buffer for offline rendering
    Float4 *m_frameBuffer;

    std::vector<cudaDeviceProp> m_deviceProperties;

    Timer m_timer;

    int m_frameNum = 0;
    int m_accumulationCounter = 1;
    float m_currentFPS = 0.0f;
};