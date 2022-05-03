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
    ~Backend() { clear(); }
    Backend(Backend const&) = delete;
    void operator=(Backend const&)  = delete;

    void init();
    void mainloop();
    void clear();

    GLFWwindow* getWindow() { return m_window; }

private:
    Backend() {}

    void initOpenGL();
    void reshape();
    void initInterop();
    void mapInteropBuffer();
    void unmapInteropBuffer();
    void display();

    // Window
    GLFWwindow* m_window;
    int m_width;
    int m_height;
    // int m_renderWidth;
    // int m_renderHeight;

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
    CUcontext m_cudaContext;
    CUstream m_cudaStream;
    cudaGraphicsResource *m_cudaGraphicsResource;

    // buffer
    float4* m_interopBuffer;

    // // Frame number
    // int m_frames;

    // // Timer
    // Timer m_timer;


    float m_gamma;
    float3 m_colorBalance;
    float m_whitePoint;
    float m_burnHighlights;
    float m_crushBlacks;
    float m_saturation;
    float m_brightness;
};

}