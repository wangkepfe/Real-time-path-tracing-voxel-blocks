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

namespace jazzfusion
{

class UI
{
public:
    static UI& Get()
    {
        static UI instance;
        return instance;
    }
    UI(UI const&) = delete;
    void operator=(UI const&) = delete;

    void init();
    void clear();
    void update();
    void render();
    void eventHandler();

private:
    UI() {}
};

}