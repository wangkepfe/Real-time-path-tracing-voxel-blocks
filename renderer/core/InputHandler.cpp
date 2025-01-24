#include "core/InputHandler.h"
#include "core/UI.h"
#include "core/OptixRenderer.h"
#include "core/Backend.h"
#include "shaders/LinearMath.h"
#include "core/GlobalSettings.h"
#include "core/RenderCamera.h"
#include "voxelengine/VoxelEngine.h"
#include "voxelengine/Block.h"

#include <iostream>
#include <fstream>

namespace jazzfusion
{

    void InputHandler::KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
    {
        auto &inputHandler = InputHandler::Get();

        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, 1);
        }

        if (key == GLFW_KEY_M && action == GLFW_PRESS)
        {
            if (inputHandler.appmode == AppMode::Menu)
            {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                inputHandler.appmode = AppMode::FreeMove;
                inputHandler.cursorReset = 1;
            }
            else if (inputHandler.appmode == AppMode::FreeMove)
            {
                inputHandler.appmode = AppMode::Gameplay;
            }
            else if (inputHandler.appmode == AppMode::Gameplay)
            {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                inputHandler.appmode = AppMode::Menu;
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

        if (inputHandler.appmode == AppMode::FreeMove)
        {
            // movement
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
            if (key == GLFW_KEY_C)
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

        if (inputHandler.appmode == AppMode::Gameplay)
        {
            // movement
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

            if (key == GLFW_KEY_SPACE)
            {
                if (action == GLFW_PRESS)
                {
                    inputHandler.fallSpeed = -0.008f;
                }
            }

            if (key == GLFW_KEY_LEFT_CONTROL)
            {
                if (action == GLFW_PRESS)
                    inputHandler.height = 1.0f;
                else if (action == GLFW_RELEASE)
                    inputHandler.height = 1.5f;
            }
        }

        if (key == GLFW_KEY_LEFT_SHIFT)
        {
            if (action == GLFW_PRESS)
                inputHandler.moveSpeed = 0.01f;
            else if (action == GLFW_RELEASE)
                inputHandler.moveSpeed = 0.003f;
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

                    if (inputHandler.currentSelectedBlockId == BlockTypeWater)
                    {
                        inputHandler.currentSelectedBlockId = 0;
                    }
                }
            }
        }
    }

    void InputHandler::CursorPosCallback(GLFWwindow *window, double xpos, double ypos)
    {
        auto &inputHandler = InputHandler::Get();
        Backend &backend = Backend::Get();
        if (inputHandler.appmode == AppMode::FreeMove || inputHandler.appmode == AppMode::Gameplay)
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

            auto &camera = RenderCamera::Get().camera;

            camera.yaw -= inputHandler.deltax * inputHandler.cursorMoveSpeed;
            camera.pitch -= inputHandler.deltay * inputHandler.cursorMoveSpeed;
            camera.pitch = clampf(camera.pitch, -PI_OVER_2 + 0.1f, PI_OVER_2 - 0.1f);

            backend.resetAccumulationCounter();
        }
    }

    void InputHandler::MouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
    {
        auto &inputHandler = InputHandler::Get();
        Backend &backend = Backend::Get();
        if (inputHandler.appmode == AppMode::FreeMove || inputHandler.appmode == AppMode::Gameplay)
        {
            inputHandler.mouseButtonCallbackFunc(button, action, mods);

            if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
            {
            }
            else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
            {
            }
            backend.resetAccumulationCounter();
        }
    }

    void InputHandler::update()
    {
        auto &camera = RenderCamera::Get().camera;
        Backend &backend = Backend::Get();

        float deltaTimeMs = backend.getTimer().getDeltaTime();

        if (appmode == AppMode::FreeMove)
        {
            if (moveW || moveS || moveA || moveD || moveC || moveX)
            {
                Float3 movingDir{0};
                Float3 strafeDir = cross(camera.dir, Float3(0, 1, 0)).normalize();

                if (moveW)
                    movingDir += camera.dir;
                if (moveS)
                    movingDir -= camera.dir;
                if (moveA)
                    movingDir -= strafeDir;
                if (moveD)
                    movingDir += strafeDir;
                if (moveC)
                    movingDir += Float3(0, 1, 0);
                if (moveX)
                    movingDir -= Float3(0, 1, 0);

                camera.pos += movingDir * deltaTimeMs * moveSpeed;

                backend.resetAccumulationCounter();
            }
        }

        if (appmode == AppMode::Gameplay)
        {
            auto &voxelEngine = vox::VoxelEngine::Get();
            auto &voxelChunk = voxelEngine.voxelChunk;

            // Horizontal
            if (moveW || moveS || moveA || moveD)
            {
                Float3 movingDir{0};

                Float3 upDir = Float3(0, 1, 0);
                Float3 strafeRightDir = cross(camera.dir, upDir).normalize();
                Float3 frontDir = cross(upDir, strafeRightDir).normalize();

                if (moveW)
                    movingDir += frontDir;
                if (moveS)
                    movingDir -= frontDir;
                if (moveA)
                    movingDir -= strafeRightDir;
                if (moveD)
                    movingDir += strafeRightDir;

                Float3 horizontalMove = movingDir * deltaTimeMs * moveSpeed;

                auto &voxelEngine = vox::VoxelEngine::Get();
                auto &voxelChunk = voxelEngine.voxelChunk;

                auto v0 = voxelChunk.get(camera.pos + horizontalMove);
                auto v1 = voxelChunk.get(camera.pos + horizontalMove - Float3(0, 1, 0));

                if (v0.id == BlockTypeEmpty && v1.id == BlockTypeEmpty)
                {
                    camera.pos = camera.pos + horizontalMove;
                }
            }

            // Vertical

            // Free fall
            float fallAccel = 9.8e-6f;
            fallSpeed += fallAccel * deltaTimeMs;
            camera.pos.y -= fallSpeed * deltaTimeMs;

            auto v2 = voxelChunk.get(camera.pos - Float3(0, height, 0));

            if (fallSpeed > 0.0f && v2.id != BlockTypeEmpty)
            {
                while (v2.id != BlockTypeEmpty)
                {
                    camera.pos.y += 1.0f;
                    v2 = voxelChunk.get(camera.pos - Float3(0, height, 0));
                }
                camera.pos.y = static_cast<float>(static_cast<int>(camera.pos.y - height)) + height;
                fallSpeed = 0.0f;
            }

            std::cout << "camera.pos.y = " << camera.pos.y << "\n";

            // head bump to roof
            auto v3 = voxelChunk.get(camera.pos + Float3(0, 0.49f, 0));
            if (fallSpeed < 0.0f && v3.id != BlockTypeEmpty)
            {
                fallSpeed = 0.0f;
            }
        }
    }

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

        {
            std::string sceneFileName = fileName + "vox.bin";
            ofstream myfile(sceneFileName, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
            if (myfile.is_open())
            {
                myfile.write(reinterpret_cast<char *>(vox::VoxelEngine::Get().voxelChunk.data), vox::VoxelEngine::Get().voxelChunk.size());
                myfile.close();
                cout << "Successfully saved scene to file \"" << sceneFileName << "\".\n";
            }
            else
            {
                cout << "Error: Failed to save scene to file \"" << sceneFileName << "\".\n";
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
        {
            std::string sceneFileName = fileName + "vox.bin";
            if (sceneFileName.empty())
            {
                cout << "Error: Scene file name " << sceneFileName << " is not valid.\n";
                return;
            }
            ifstream infile(sceneFileName, std::ifstream::in | std::ifstream::binary);
            if (infile.good())
            {
                infile.read(reinterpret_cast<char *>(vox::VoxelEngine::Get().voxelChunk.data), vox::VoxelEngine::Get().voxelChunk.size());
                infile.close();

                vox::VoxelEngine::Get().reload();
            }
            else
            {
                cout << "Error: Failed to read scene file \"" << sceneFileName.c_str() << "\".\n";
            }
        }
    }
}