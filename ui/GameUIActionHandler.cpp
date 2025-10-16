#include "ui/GameUIActionHandler.h"

#include <GLFW/glfw3.h>

DefaultGameUIActionHandler::DefaultGameUIActionHandler(GLFWwindow* window)
    : m_window(window)
{
}

void DefaultGameUIActionHandler::OnQuitRequested()
{
    if (m_window)
    {
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    }
}
