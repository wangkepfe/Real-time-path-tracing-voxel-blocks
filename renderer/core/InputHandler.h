#pragma once

#include <iostream>
#include <functional>

struct GLFWwindow;

namespace jazzfusion
{

enum class AppMode
{
    Gameplay,
    Menu,
};

class InputHandler
{
public:
    static InputHandler& Get()
    {
        static InputHandler instance;
        return instance;
    }
    InputHandler(InputHandler const&) = delete;
    void operator=(InputHandler const&) = delete;

    static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void CursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

    static void SaveCameraToFile(const std::string& camFileName);
    static void LoadCameraFromFile(const std::string& camFileName);

    void update();
    void setMouseButtonCallbackFunc(std::function<void(int, int, int)> mouseButtonCallbackFuncIn)
    {
        mouseButtonCallbackFunc = mouseButtonCallbackFuncIn;
    }

private:
    InputHandler() {}

    float moveSpeed = 0.01f;
    float cursorMoveSpeed = 0.001f;

    double xpos = 0;
    double ypos = 0;

    bool moveW = 0;
    bool moveS = 0;
    bool moveA = 0;
    bool moveD = 0;
    bool moveC = 0;
    bool moveX = 0;

    float deltax = 0;
    float deltay = 0;

    int cursorReset = 1;

    std::function<void(int, int, int)> mouseButtonCallbackFunc;

    AppMode appmode = AppMode::Gameplay;
};

}