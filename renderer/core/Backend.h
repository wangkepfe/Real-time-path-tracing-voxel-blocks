#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

#include "shaders/LinearMath.h"

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif

#include <windows.h>
#endif

#ifndef OFFLINE_MODE
#ifndef __APPLE__
#include <GL/glew.h>
#if defined(_WIN32)
#include <GL/wglew.h>
#endif
#endif

// Needs to be included after OpenGL headers! CUDA Runtime API version.
#include <cuda_gl_interop.h>

#include <GLFW/glfw3.h>
#endif // OFFLINE_MODE
#include <vector>
#include <functional>

#include "util/Timer.h"

class Backend
{
public:
    static Backend &Get()
    {
        static Backend instance;
        return instance;
    }
    Backend(Backend const &) = delete;
    void operator=(Backend const &) = delete;

    void init();
    void mainloop();
    void clear();
    CUstream getCudaStream() const { return m_cudaStream; }
#ifndef OFFLINE_MODE
    GLFWwindow *getWindow() { return m_window; }
#endif
    CUcontext getCudaContext() const { return m_cudaContext; }
    float *getToneMapGain() { return &m_toneMapGain; }
    float *getToneMapMaxWhite() { return &m_toneMapMaxWhite; }
    const Timer &getTimer() const { return m_timer; }
    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }
    int getMaxRenderWidth() const { return m_maxRenderWidth; }
    int getMaxRenderHeight() const { return m_maxRenderHeight; }
    float getCurrentFPS() const { return m_currentFPS; }
    int getCurrentRenderWidth() const { return m_currentRenderWidth; }
    int getFrameNum() const { return m_frameNum; }

    const std::string GlslVersion{"#version 330"};

private:
    Backend() {}

    void initOpenGL();
    void initInterop();
    void mapInteropBuffer();
    void unmapInteropBuffer();
    void display();
    void dumpSystemInformation();
    void dynamicResolution();

    // FPS limiter
    float m_maxFpsAllowed = 144.0f;

    // Dynamic resolution
    bool m_dynamicResolution = false;

    float m_targetFPS = 60.0f;

    int m_minRenderWidth;
    int m_minRenderHeight;

    int m_maxRenderWidth;
    int m_maxRenderHeight;

    int m_historyRenderWidth;
    int m_historyRenderHeight;

    // For UI display
    float m_currentFPS;
    float m_currentRenderWidth;

#ifndef OFFLINE_MODE
    // Window
    GLFWwindow *m_window;
#endif
    int m_width;
    int m_height;

#ifndef OFFLINE_MODE
    // OpenGL variables
    GLuint m_pbo;
    GLuint m_hdrTexture;
    GLuint m_vboAttributes;
    GLuint m_vboIndices;
    GLint m_positionLocation;
    GLint m_texCoordLocation;

    // GLSL shaders objects and program.
    GLuint m_glslVS;
    GLuint m_glslFS;
    GLuint m_glslProgram;
#endif

    // CUDA stuffs
    CUcontext m_cudaContext;
    CUstream m_cudaStream;
#ifndef OFFLINE_MODE
    cudaGraphicsResource *m_cudaGraphicsResource;

    // buffer
    Float4 *m_interopBuffer;
#endif

    // tone mapping
    float m_toneMapGain = 1.0f;
    float m_toneMapMaxWhite = 100.0f;

    std::vector<cudaDeviceProp> m_deviceProperties;

    Timer m_timer;

    int m_frameNum;
};