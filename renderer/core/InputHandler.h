#pragma once

#include <iostream>
#include <functional>
#include <memory>
#include "../shaders/LinearMath.h"

struct GLFWwindow;
class Character;
class CameraController;

enum class AppMode
{
    GUI,
    FreeMove,
    CharacterFollow,
};

class InputHandler
{
public:
    static InputHandler &Get()
    {
        static InputHandler instance;
        return instance;
    }
    InputHandler(InputHandler const &) = delete;
    void operator=(InputHandler const &) = delete;

    static void KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
    static void CursorPosCallback(GLFWwindow *window, double xpos, double ypos);
    static void MouseButtonCallback(GLFWwindow *window, int button, int action, int mods);

    static void SaveSceneToFile();
    static void LoadSceneFromFile();

    void update();
    void setMouseButtonCallbackFunc(std::function<void(int, int, int)> mouseButtonCallbackFuncIn)
    {
        mouseButtonCallbackFunc = mouseButtonCallbackFuncIn;
    }

    int currentSelectedBlockId = 1;
    
    // Character control
    void setCharacter(Character* character);
    Character* getCharacter() const { return m_character; }
    
    // Camera controller management
    CameraController* getCurrentCameraController() const { return m_currentCameraController; }
    AppMode getCurrentMode() const { return appmode; }

private:
    InputHandler();
    ~InputHandler(); // Need custom destructor for unique_ptr with incomplete type
    void initializeCameraControllers();
    void switchCameraController(AppMode newMode);

    double xpos = 0;
    double ypos = 0;

    bool moveW = 0;
    bool moveS = 0;
    bool moveA = 0;
    bool moveD = 0;
    bool moveC = 0;
    bool moveX = 0;
    
    // Speed modifier keys
    bool slowMode = 0;   // Ctrl modifier
    bool fastMode = 0;   // Shift modifier

    float deltax = 0;
    float deltay = 0;

    int cursorReset = 1;

    std::function<void(int, int, int)> mouseButtonCallbackFunc;

    AppMode appmode = AppMode::GUI;
    
    // Legacy members still used in some places
    float moveSpeed = 0.003f;
    float cursorMoveSpeed = 0.001f;
    
    // Character control
    Character* m_character = nullptr;
    
    // Unified camera management
    std::unique_ptr<CameraController> m_freeCameraController;
    std::unique_ptr<CameraController> m_characterFollowCameraController;  
    CameraController* m_currentCameraController = nullptr;
};