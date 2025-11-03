#pragma once

#include <cuda_runtime.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#ifndef __APPLE__
#include <GL/glew.h>
#if defined(_WIN32)
#include <GL/wglew.h>
#endif
#endif

// Needs to be included after OpenGL headers!
// CUDA Runtime API version.
#include <cuda_gl_interop.h>

#include <GLFW/glfw3.h>

class DeveloperGUIOverlay
{
public:
    static DeveloperGUIOverlay &Get()
    {
        static DeveloperGUIOverlay instance;
        return instance;
    }
    DeveloperGUIOverlay(DeveloperGUIOverlay const &) = delete;
    void operator=(DeveloperGUIOverlay const &) = delete;
    ~DeveloperGUIOverlay();

    void init();
    void clear();
    void update();
    void render();
    void eventHandler();

private:
    DeveloperGUIOverlay() {}
};
