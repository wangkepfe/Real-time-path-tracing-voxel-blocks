#pragma once

#include <cuda_runtime.h>

#include "imgui.h"

#define IMGUI_DEFINE_MATH_OPERATORS 1
#include "imgui_internal.h"

#include "imgui_impl_glfw_gl3.h"

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

enum GuiState
{
    GUI_STATE_NONE,
    GUI_STATE_ORBIT,
    GUI_STATE_PAN,
    GUI_STATE_DOLLY,
    GUI_STATE_FOCUS
};

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

    GuiState m_guiState;
    bool m_isVisibleGUI;
    float m_mouseSpeedRatio;
};

}