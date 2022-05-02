

#include "shaders/app_config.h"

#include "core/Options.h"
#include "core/Application.h"

#include <IL/il.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

static Application *g_app = nullptr;

static bool displayGUI = true;

static void error_callback(int error, const char *description)
{
    std::cerr << "Error: " << error << ": " << description << '\n';
}

int runApp(Options const &options)
{
    int widthClient = std::max(1, options.getClientWidth());
    int heightClient = std::max(1, options.getClientHeight());

    // glfwWindowHint(GLFW_DECORATED, windowBorder);

    GLFWwindow *window = glfwCreateWindow(widthClient, heightClient, "intro_runtime - Copyright (c) 2020 NVIDIA Corporation", NULL, NULL);
    if (!window)
    {
        error_callback(APP_ERROR_CREATE_WINDOW, "glfwCreateWindow() failed.");
        return APP_ERROR_CREATE_WINDOW;
    }

    glfwMakeContextCurrent(window);

    if (glewInit() != GL_NO_ERROR)
    {
        error_callback(APP_ERROR_GLEW_INIT, "GLEW failed to initialize.");
        return APP_ERROR_GLEW_INIT;
    }

    ilInit(); // Initialize DevIL once.

    g_app = new Application(window, options);

    if (!g_app->isValid())
    {
        std::cerr << "ERROR: Application failed to initialize successfully.\n";
        ilShutDown();
        return APP_ERROR_APP_INIT;
    }

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents(); // Render continuously.

        glfwGetFramebufferSize(window, &widthClient, &heightClient);
        g_app->reshape(widthClient, heightClient);
        g_app->render(); // OptiX rendering.
        g_app->guiNewFrame();
        g_app->guiWindow();
        g_app->guiEventHandler(); // SPACE to toggle the GUI windows and all mouse tracking via GuiState.
        g_app->display(); // OpenGL display always required to lay the background for the GUI.
        g_app->guiRender(); // Render all ImGUI elements at last.

        glfwSwapBuffers(window);
    }

    delete g_app;

    ilShutDown();

    return APP_EXIT_SUCCESS; // Success.
}

int main(int argc, char *argv[])
{
    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
    {
        error_callback(APP_ERROR_GLFW_INIT, "GLFW failed to initialize.");
        return APP_ERROR_GLFW_INIT;
    }

    int result = APP_ERROR_UNKNOWN;

    Options options;

    if (options.parseCommandLine(argc, argv))
    {
        result = runApp(options);
    }

    glfwTerminate();

    return result;
}
