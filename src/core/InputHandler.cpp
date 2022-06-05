#include "core/InputHandler.h"
#include "core/UI.h"
#include "core/OptixRenderer.h"
#include "core/Backend.h"
#include "shaders/LinearMath.h"
#include "core/GlobalSettings.h"

#include <iostream>
#include <fstream>

namespace jazzfusion
{

void InputHandler::SetKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto& inputHandler = InputHandler::Get();

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        // exit
    }

    if (mods == GLFW_MOD_CONTROL)
    {
        // camera save & load
        if (key == GLFW_KEY_C && action == GLFW_PRESS) { InputHandler::SaveCameraToFile(GlobalSettings::GetCameraSaveFileName()); }
        if (key == GLFW_KEY_V && action == GLFW_PRESS) { InputHandler::LoadCameraFromFile(GlobalSettings::GetCameraSaveFileName()); }
    }
    else
    {
        // movement
        if (key == GLFW_KEY_W) { if (action == GLFW_PRESS) inputHandler.moveW = 1; else if (action == GLFW_RELEASE) inputHandler.moveW = 0; }
        if (key == GLFW_KEY_S) { if (action == GLFW_PRESS) inputHandler.moveS = 1; else if (action == GLFW_RELEASE) inputHandler.moveS = 0; }
        if (key == GLFW_KEY_A) { if (action == GLFW_PRESS) inputHandler.moveA = 1; else if (action == GLFW_RELEASE) inputHandler.moveA = 0; }
        if (key == GLFW_KEY_D) { if (action == GLFW_PRESS) inputHandler.moveD = 1; else if (action == GLFW_RELEASE) inputHandler.moveD = 0; }
        if (key == GLFW_KEY_C) { if (action == GLFW_PRESS) inputHandler.moveC = 1; else if (action == GLFW_RELEASE) inputHandler.moveC = 0; }
        if (key == GLFW_KEY_X) { if (action == GLFW_PRESS) inputHandler.moveX = 1; else if (action == GLFW_RELEASE) inputHandler.moveX = 0; }

        if (key == GLFW_KEY_LEFT_SHIFT) { if (action == GLFW_PRESS) inputHandler.moveSpeed = 0.001f; else if (action == GLFW_RELEASE) inputHandler.moveSpeed = 0.01f; }
    }
}

void InputHandler::SetCursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    auto& inputHandler = InputHandler::Get();

    if (inputHandler.cursorReset)
    {
        inputHandler.cursorReset = false;
        inputHandler.xpos = xpos;
        inputHandler.ypos = ypos;
        return;
    }

    inputHandler.deltax = (float)(xpos - inputHandler.xpos);
    inputHandler.deltay = (float)(ypos - inputHandler.ypos);

    inputHandler.xpos = xpos;
    inputHandler.ypos = ypos;

    Camera& camera = OptixRenderer::Get().getCamera();

    camera.yaw -= inputHandler.deltax * inputHandler.cursorMoveSpeed;
    camera.pitch -= inputHandler.deltay * inputHandler.cursorMoveSpeed;
    camera.pitch = clampf(camera.pitch, -PI_OVER_2 + 0.1f, PI_OVER_2 - 0.1f);
}

void InputHandler::update()
{
    Camera& camera = OptixRenderer::Get().getCamera();
    Backend& backend = Backend::Get();

    if (moveW || moveS || moveA || moveD || moveC || moveX)
    {
        Float3 movingDir = 0;
        Float3 strafeDir = cross(camera.dir, Float3(0, 1, 0)).normalize();

        if (moveW) movingDir += camera.dir;
        if (moveS) movingDir -= camera.dir;
        if (moveA) movingDir -= strafeDir;
        if (moveD) movingDir += strafeDir;
        if (moveC) movingDir += Float3(0, 1, 0);
        if (moveX) movingDir -= Float3(0, 1, 0);

        camera.pos += movingDir * backend.getTimer().getDeltaTime() * moveSpeed;
    }
}

void InputHandler::SaveCameraToFile(const std::string& camFileName)
{
    Camera& camera = OptixRenderer::Get().getCamera();
    using namespace std;
    ofstream myfile(camFileName, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
    if (myfile.is_open())
    {
        myfile.write(reinterpret_cast<char*>(&camera), sizeof(Camera));
        myfile.close();
        cout << "Successfully saved camera to file \"" << camFileName.c_str() << "\".\n";
    }
    else
    {
        cout << "Error: Failed to save camera to file \"" << camFileName.c_str() << "\".\n";
    }

}

void InputHandler::LoadCameraFromFile(const std::string& camFileName)
{
    Camera& camera = OptixRenderer::Get().getCamera();
    using namespace std;
    if (camFileName.empty())
    {
        cout << "Error: Camera file name is not valid.\n";
        return;
    }
    ifstream infile(camFileName, std::ifstream::in | std::ifstream::binary);
    if (infile.good())
    {
        char* buffer = new char[sizeof(Camera)];
        infile.read(buffer, sizeof(Camera));
        camera = *reinterpret_cast<Camera*>(buffer);
        delete[] buffer;
        infile.close();
    }
    else
    {
        cout << "Error: Failed to read camera file \"" << camFileName.c_str() << "\".\n";
    }
}

}