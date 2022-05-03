#pragma once

// Always include this before any OptiX headers!
#include <cuda_runtime.h>

#include <optix.h>


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

namespace jazzfusion {

class Backend
{
public:
    static Backend& Get()
    {
        static Backend instance;
        return instance;
    }
    Backend(Backend const&) = delete;
    void operator=(Backend const&)  = delete;

    void run();

    GLFWwindow* getWindow() { return m_window; }

private:
    Backend() {}

    // Window
    GLFWwindow* m_window;
    // int m_width;
    // int m_height;
    // int m_renderWidth;
    // int m_renderHeight;

    // // OpenGL variables
    // GLuint m_pbo;
    // GLuint m_hdrTexture;
    // GLuint m_vboAttributes;
    // GLuint m_vboIndices;
    // GLint m_positionLocation;
    // GLint m_texCoordLocation;

    // // CUDA stuffs
    // CUcontext m_cudaContext;
    // CUstream m_cudaStream;
    // cudaGraphicsResource *m_cudaGraphicsResource;

    // // Output buffer
    // float4 *m_outputBuffer;

    // // Frame number
    // int m_frames;

    // // Tonemapper group
    // float m_gamma;
    // float3 m_colorBalance;
    // float m_whitePoint;
    // float m_burnHighlights;
    // float m_crushBlacks;
    // float m_saturation;
    // float m_brightness;

    // // Timer
    // Timer m_timer;
};

}