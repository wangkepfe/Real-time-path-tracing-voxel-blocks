#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#include "shaders/LinearMath.h"

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif

#include <windows.h>
#endif

#ifndef __APPLE__
#include <GL/glew.h>
#if defined(_WIN32)
#include <GL/wglew.h>
#endif
#endif

// Needs to be included after OpenGL headers! CUDA Runtime API version.
#include <cuda_gl_interop.h>

#include <GLFW/glfw3.h>
#include <vector>

#include "util/Timer.h"

namespace jazzfusion
{

class Backend
{
public:
    static Backend& Get()
    {
        static Backend instance;
        return instance;
    }
    Backend(Backend const&) = delete;
    void operator=(Backend const&) = delete;

    void init();
    void mainloop();
    void clear();
    CUstream getCudaStream() const { return m_cudaStream; }
    GLFWwindow* getWindow() { return m_window; }
    CUcontext getCudaContext() const { return m_cudaContext; }
    float* getToneMapGain() { return &m_toneMapGain; }
    float* getToneMapMaxWhite() { return &m_toneMapMaxWhite; }
    const Timer& getTimer() const { return m_timer; }

private:
    Backend() {}

    void initOpenGL();
    void initInterop();
    void mapInteropBuffer();
    void unmapInteropBuffer();
    void display();
    void dumpSystemInformation();

    // Window
    GLFWwindow* m_window;
    int m_width;
    int m_height;

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

    // CUDA stuffs
    CUcontext             m_cudaContext;
    CUstream              m_cudaStream;
    cudaGraphicsResource* m_cudaGraphicsResource;

    // buffer
    Float4* m_interopBuffer;

    // tone mapping
    float m_toneMapGain = 1.0f;
    float m_toneMapMaxWhite = 100.0f;

    std::vector<cudaDeviceProp> m_deviceProperties;

    Timer m_timer;
};

}