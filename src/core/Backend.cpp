#include "core/Backend.h"
#include "core/Options.h"
#include "core/Application.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace jazzfusion {

void Backend::run()
{
    Options options = {};

    int widthClient = 1920;
    int heightClient = 1080;

    m_window = glfwCreateWindow(widthClient, heightClient, "intro_runtime - Copyright (c) 2020 NVIDIA Corporation", NULL, NULL);
    if (!m_window)
    {
        std::cerr << "Error: glfwCreateWindow() failed.\n";
        return;
    }

    glfwMakeContextCurrent(m_window);

    if (glewInit() != GL_NO_ERROR)
    {
        std::cerr << "Error: GLEW failed to initialize.\n";
        return;
    }

    ilInit(); // Initialize DevIL once.

    auto& app = Application::Get();
    app.init(m_window);

    if (!app.isValid())
    {
        std::cerr << "ERROR: Application failed to initialize successfully.\n";
        ilShutDown();
        return;
    }

    // Main loop
    while (!glfwWindowShouldClose(m_window))
    {
        glfwPollEvents(); // Render continuously.

        glfwGetFramebufferSize(m_window, &widthClient, &heightClient);
        app.reshape(widthClient, heightClient);
        app.render(); // OptiX rendering.
        app.guiNewFrame();
        app.guiWindow();
        app.guiEventHandler(); // SPACE to toggle the GUI windows and all mouse tracking via GuiState.
        app.display(); // OpenGL display always required to lay the background for the GUI.
        app.guiRender(); // Render all ImGUI elements at last.

        glfwSwapBuffers(m_window);
    }

    ilShutDown();
    return; // Success.
}

}