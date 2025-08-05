#include "core/InputHandler.h"
#include "core/UI.h"
#include "core/OptixRenderer.h"
#include "core/Backend.h"
#include "shaders/LinearMath.h"
#include "core/GlobalSettings.h"
#include "core/RenderCamera.h"
#include "core/Character.h"
#include "core/CameraController.h"
#include "core/FreeCameraController.h"
#include "core/CharacterFollowCameraController.h"
#include "voxelengine/VoxelEngine.h"
#include "voxelengine/Block.h"

#include <fstream>

InputHandler::InputHandler()
{
    initializeCameraControllers();
}

InputHandler::~InputHandler()
{
    // Destructor implementation needed for unique_ptr with incomplete type in header
    // The camera controllers will be automatically destroyed
}

void InputHandler::initializeCameraControllers()
{
    // Create camera controllers
    m_freeCameraController = std::make_unique<FreeCameraController>();
    m_characterFollowCameraController = std::make_unique<CharacterFollowCameraController>();
    
    // Start with free camera by default
    m_currentCameraController = m_freeCameraController.get();
}

void InputHandler::switchCameraController(AppMode newMode)
{
    if (appmode == newMode)
    {
        return; // Already in this mode
    }
        
    auto &camera = RenderCamera::Get().camera;
    
    // Deactivate current controller
    if (m_currentCameraController)
    {
        m_currentCameraController->onDeactivate(camera);
    }
    
    // Switch to new controller
    appmode = newMode;
    CameraController* newController = nullptr;
    
    switch (newMode)
    {
        case AppMode::FreeMove:
            newController = m_freeCameraController.get();
            break;
        case AppMode::CharacterFollow:
            newController = m_characterFollowCameraController.get();
            break;
        case AppMode::GUI:
            // Keep current controller but don't process input
            newController = m_currentCameraController;
            break;
    }
    
    if (newController && newController != m_currentCameraController)
    {
        m_currentCameraController = newController;
        m_currentCameraController->onActivate(camera);
    }
}

void InputHandler::setCharacter(Character* character)
{
    m_character = character;
    
    // Set character in the character follow controller
    if (m_characterFollowCameraController)
    {
        auto* charController = static_cast<CharacterFollowCameraController*>(m_characterFollowCameraController.get());
        charController->setCharacter(character);
    }
    
    // Switch to CharacterFollow mode if we have a character
    if (m_character)
    {
        switchCameraController(AppMode::CharacterFollow);
    }
    else
    {
    }
}

void InputHandler::KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    auto &inputHandler = InputHandler::Get();

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, 1);
    }

    if (key == GLFW_KEY_M && action == GLFW_PRESS)
    {
        if (inputHandler.appmode == AppMode::GUI)
        {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            inputHandler.switchCameraController(AppMode::CharacterFollow);
            inputHandler.cursorReset = 1;
        }
        else if (inputHandler.appmode == AppMode::CharacterFollow)
        {
            inputHandler.switchCameraController(AppMode::FreeMove);
            inputHandler.cursorReset = 1;
        }
        else if (inputHandler.appmode == AppMode::FreeMove)
        {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            inputHandler.switchCameraController(AppMode::GUI);
        }
    }

    if (mods == GLFW_MOD_CONTROL)
    {
        // camera save & load
        if (key == GLFW_KEY_C && action == GLFW_PRESS)
        {
            InputHandler::SaveSceneToFile();
        }
        if (key == GLFW_KEY_V && action == GLFW_PRESS)
        {
            InputHandler::LoadSceneFromFile();
        }
    }

    // Universal movement input handling (all camera modes use same keys)
    if (inputHandler.appmode != AppMode::GUI)
    {
        // WASD movement
        if (key == GLFW_KEY_W)
        {
            if (action == GLFW_PRESS)
                inputHandler.moveW = 1;
            else if (action == GLFW_RELEASE)
                inputHandler.moveW = 0;
        }
        if (key == GLFW_KEY_S)
        {
            if (action == GLFW_PRESS)
                inputHandler.moveS = 1;
            else if (action == GLFW_RELEASE)
                inputHandler.moveS = 0;
        }
        if (key == GLFW_KEY_A)
        {
            if (action == GLFW_PRESS)
                inputHandler.moveA = 1;
            else if (action == GLFW_RELEASE)
                inputHandler.moveA = 0;
        }
        if (key == GLFW_KEY_D)
        {
            if (action == GLFW_PRESS)
                inputHandler.moveD = 1;
            else if (action == GLFW_RELEASE)
                inputHandler.moveD = 0;
        }
        
        // Vertical movement (different meaning for different controllers)
        if (key == GLFW_KEY_C || key == GLFW_KEY_SPACE)
        {
            if (action == GLFW_PRESS)
                inputHandler.moveC = 1;
            else if (action == GLFW_RELEASE)
                inputHandler.moveC = 0;
        }
        if (key == GLFW_KEY_X)
        {
            if (action == GLFW_PRESS)
                inputHandler.moveX = 1;
            else if (action == GLFW_RELEASE)
                inputHandler.moveX = 0;
        }
    }

    // Speed modifier keys for free camera
    if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL)
    {
        if (action == GLFW_PRESS)
            inputHandler.slowMode = 1;
        else if (action == GLFW_RELEASE)
            inputHandler.slowMode = 0;
    }
    
    if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT)
    {
        if (action == GLFW_PRESS)
            inputHandler.fastMode = 1;
        else if (action == GLFW_RELEASE)
            inputHandler.fastMode = 0;
    }

    for (int i = 0; i < 10; ++i)
    {
        if (key == GLFW_KEY_0 + i)
        {
            if (action == GLFW_PRESS)
            {
                inputHandler.currentSelectedBlockId = i;

                if (mods == GLFW_MOD_SHIFT)
                {
                    inputHandler.currentSelectedBlockId += 10;
                }
                else if (mods == GLFW_MOD_CONTROL)
                {
                    inputHandler.currentSelectedBlockId += 20;
                }

                inputHandler.currentSelectedBlockId = min(inputHandler.currentSelectedBlockId, BlockTypeNum - 1);
            }
        }
    }
}

void InputHandler::CursorPosCallback(GLFWwindow *window, double xpos, double ypos)
{
    auto &inputHandler = InputHandler::Get();
    Backend &backend = Backend::Get();
    if (inputHandler.appmode != AppMode::GUI)
    {
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

        // Delegate mouse movement to current camera controller
        if (inputHandler.m_currentCameraController)
        {
            auto &camera = RenderCamera::Get().camera;
            inputHandler.m_currentCameraController->handleMouseMovement(camera, inputHandler.deltax, inputHandler.deltay);
        }
    }
}

void InputHandler::MouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    auto &inputHandler = InputHandler::Get();
    Backend &backend = Backend::Get();
    if (inputHandler.appmode == AppMode::FreeMove || inputHandler.appmode == AppMode::CharacterFollow)
    {
        inputHandler.mouseButtonCallbackFunc(button, action, mods);

        if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
        {
        }
        else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
        {
            // Trigger place animation in character following mode
            if (inputHandler.appmode == AppMode::CharacterFollow && inputHandler.m_character)
            {
                inputHandler.m_character->triggerPlaceAnimation();
            }
        }
    }
}

void InputHandler::update()
{
    auto &camera = RenderCamera::Get().camera;
    Backend &backend = Backend::Get();

    float deltaTimeMs = backend.getTimer().getDeltaTime();
    
    // Debug output (reduced frequency)
    static int frameCount = 0;
    frameCount++;

    // Skip camera updates in GUI mode
    if (appmode == AppMode::GUI)
    {
        return;
    }

    // UNIFIED CAMERA UPDATE: Only one place where camera gets updated!
    if (m_currentCameraController)
    {
        // Set movement input in the current controller
        m_currentCameraController->setMovementInput(moveW, moveS, moveA, moveD, moveC, moveX);
        
        // Set speed modifiers for all camera controllers
        m_currentCameraController->setSpeedModifiers(slowMode, fastMode);
        
        // Let the controller update the camera
        m_currentCameraController->updateCamera(camera, deltaTimeMs);
    }
}

// NOTE: The old updateCameraFollowing() and initializeCameraForCharacter() methods
// have been moved to CharacterFollowCameraController and are no longer needed here.

void InputHandler::SaveSceneToFile()
{
    using namespace std;
    std::string saveId = std::to_string(Get().currentSelectedBlockId);
    std::string fileName = "save" + saveId;
    {
        std::string camFileName = fileName + "cam.bin";
        auto &camera = RenderCamera::Get().camera;
        ofstream myfile(camFileName, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
        if (myfile.is_open())
        {
            myfile.write(reinterpret_cast<char *>(&camera), sizeof(Camera));
            myfile.close();
            cout << "Successfully saved camera to file \"" << camFileName << "\".\n";
        }
        else
        {
            cout << "Error: Failed to save camera to file \"" << camFileName << "\".\n";
        }
    }

    // Save multi-chunk voxel data
    {
        auto &voxelEngine = VoxelEngine::Get();
        auto &voxelChunks = voxelEngine.voxelChunks;
        auto &chunkConfig = voxelEngine.chunkConfig;

        // Save chunk configuration first
        std::string configFileName = fileName + "config.bin";
        ofstream configFile(configFileName, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
        if (configFile.is_open())
        {
            configFile.write(reinterpret_cast<const char *>(&chunkConfig), sizeof(ChunkConfiguration));
            configFile.close();
            cout << "Successfully saved chunk configuration to file \"" << configFileName << "\".\n";
        }
        else
        {
            cout << "Error: Failed to save chunk configuration to file \"" << configFileName << "\".\n";
        }

        // Save each chunk individually
        for (unsigned int chunkIndex = 0; chunkIndex < voxelChunks.size(); ++chunkIndex)
        {
            std::string chunkFileName = fileName + "chunk" + std::to_string(chunkIndex) + ".bin";
            ofstream chunkFile(chunkFileName, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
            if (chunkFile.is_open())
            {
                chunkFile.write(reinterpret_cast<char *>(voxelChunks[chunkIndex].data), voxelChunks[chunkIndex].size());
                chunkFile.close();
                cout << "Successfully saved chunk " << chunkIndex << " to file \"" << chunkFileName << "\".\n";
            }
            else
            {
                cout << "Error: Failed to save chunk " << chunkIndex << " to file \"" << chunkFileName << "\".\n";
            }
        }
    }
}

void InputHandler::LoadSceneFromFile()
{
    using namespace std;
    std::string saveId = std::to_string(Get().currentSelectedBlockId);
    std::string fileName = "save" + saveId;
    {
        std::string camFileName = fileName + "cam.bin";
        auto &camera = RenderCamera::Get().camera;
        if (camFileName.empty())
        {
            cout << "Error: Camera file name " << camFileName << " is not valid.\n";
            return;
        }
        ifstream infile(camFileName, std::ifstream::in | std::ifstream::binary);
        if (infile.good())
        {
            char *buffer = new char[sizeof(Camera)];
            infile.read(buffer, sizeof(Camera));
            camera = *reinterpret_cast<Camera *>(buffer);
            delete[] buffer;
            infile.close();
        }
        else
        {
            cout << "Error: Failed to read camera file \"" << camFileName.c_str() << "\".\n";
        }
    }

    // Load multi-chunk voxel data
    {
        auto &voxelEngine = VoxelEngine::Get();

        // Load chunk configuration first
        std::string configFileName = fileName + "config.bin";
        ifstream configFile(configFileName, std::ifstream::in | std::ifstream::binary);
        if (configFile.good())
        {
            ChunkConfiguration loadedConfig;
            configFile.read(reinterpret_cast<char *>(&loadedConfig), sizeof(ChunkConfiguration));
            configFile.close();

            // Update the chunk configuration and resize chunks if needed
            voxelEngine.chunkConfig = loadedConfig;
            voxelEngine.voxelChunks.resize(loadedConfig.getTotalChunks());

            cout << "Successfully loaded chunk configuration from file \"" << configFileName << "\".\n";
        }
        else
        {
            cout << "Error: Failed to read chunk configuration file \"" << configFileName.c_str() << "\".\n";
            return;
        }

        // Load each chunk individually
        auto &voxelChunks = voxelEngine.voxelChunks;
        bool allChunksLoaded = true;

        for (unsigned int chunkIndex = 0; chunkIndex < voxelChunks.size(); ++chunkIndex)
        {
            std::string chunkFileName = fileName + "chunk" + std::to_string(chunkIndex) + ".bin";
            ifstream chunkFile(chunkFileName, std::ifstream::in | std::ifstream::binary);
            if (chunkFile.good())
            {
                chunkFile.read(reinterpret_cast<char *>(voxelChunks[chunkIndex].data), voxelChunks[chunkIndex].size());
                chunkFile.close();
                cout << "Successfully loaded chunk " << chunkIndex << " from file \"" << chunkFileName << "\".\n";
            }
            else
            {
                cout << "Error: Failed to read chunk file \"" << chunkFileName.c_str() << "\".\n";
                allChunksLoaded = false;
            }
        }

        if (allChunksLoaded)
        {
            VoxelEngine::Get().reload();
        }
        else
        {
            cout << "Error: Not all chunks could be loaded. Scene reload aborted.\n";
        }
    }
}
