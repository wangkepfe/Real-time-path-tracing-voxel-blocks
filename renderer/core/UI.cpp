#include "core/UI.h"
#include "core/OptixRenderer.h"
#include "core/Backend.h"
#include "core/InputHandler.h"
#include "core/GlobalSettings.h"
#include "core/RenderCamera.h"
#include "core/SceneConfig.h"
#include "core/CameraController.h"
#include "core/Character.h"
#include "voxelengine/VoxelEngine.h"

void UI::init()
{
    auto &backend = Backend::Get();

    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(backend.getWindow(), true);
    ImGui_ImplOpenGL3_Init(backend.GlslVersion.c_str());
}

UI::~UI()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void UI::update()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    auto &backend = Backend::Get();
    auto &renderer = OptixRenderer::Get();
    auto &inputHandler = InputHandler::Get();

    if (!ImGui::Begin("Render Settings", nullptr, 0))
    {
        ImGui::End();
        return;
    }

    auto &camera = RenderCamera::Get().camera;

    // Get current camera mode name
    const char* currentCameraMode = "Unknown";
    if (inputHandler.getCurrentCameraController())
    {
        if (inputHandler.getCurrentMode() == AppMode::GUI)
        {
            currentCameraMode = "GUI";
        }
        else
        {
            currentCameraMode = inputHandler.getCurrentCameraController()->getName();
        }
    }

    ImGui::Text("Camera Mode: %s", currentCameraMode);
    ImGui::Text("Frame = %d", backend.getFrameNum());
    ImGui::Text("Current FPS = %.1f", backend.getCurrentFPS());
    ImGui::Text("Current Render Width = %d", backend.getCurrentRenderWidth());
    ImGui::Text("Resolution: (%d, %d)", backend.getCurrentRenderWidth(), backend.getCurrentRenderWidth() / 16 * 9);
    ImGui::Text("Scale: %.1f %%", backend.getCurrentRenderWidth() / (float)backend.getWidth() * 100.0f);
    ImGui::Text("Camera pos=(%.2f, %.2f, %.2f)", camera.pos.x, camera.pos.y, camera.pos.z);
    ImGui::Text("Camera dir=(%.2f, %.2f, %.2f)", camera.dir.x, camera.dir.y, camera.dir.z);
    
    // Show character movement info if character exists
    if (inputHandler.getCharacter())
    {
        auto character = inputHandler.getCharacter();
        auto& movement = character->getMovement();
        auto& physics = character->getPhysics();
        
        ImGui::Separator();
        ImGui::Text("Character Speed: %.2f", movement.currentSpeed);
        ImGui::Text("Character Move Dir: (%.2f, %.2f, %.2f)", movement.moveDirection.x, movement.moveDirection.y, movement.moveDirection.z);
        ImGui::Text("Character Velocity: (%.2f, %.2f, %.2f)", physics.velocity.x, physics.velocity.y, physics.velocity.z);
        ImGui::Text("Character Grounded: %s", physics.isGrounded ? "Yes" : "No");
    }
    
    ImGui::Text("Current selected block ID = %d", inputHandler.currentSelectedBlockId);
    
    // Display center block information
    ImGui::Separator();
    ImGui::Text("Center Block Info:");
    auto& voxelEngine = VoxelEngine::Get();
    if (voxelEngine.centerBlockInfo.hasValidBlock) {
        ImGui::Text("Block ID: %d", voxelEngine.centerBlockInfo.blockId);
        ImGui::Text("Block Name: %s", voxelEngine.centerBlockInfo.blockName.c_str());
        ImGui::Text("Position: (%d, %d, %d)", 
                   voxelEngine.centerBlockInfo.position.x,
                   voxelEngine.centerBlockInfo.position.y,
                   voxelEngine.centerBlockInfo.position.z);
    } else {
        ImGui::Text("No block in center crosshair");
    }

    if (ImGui::CollapsingHeader("Temporal Denoising", 0))
    {
        DenoisingParams &denoisingParams = GlobalSettings::GetDenoisingParams();

        // Master enable/disable checkbox
        ImGui::Checkbox("Enable Denoiser", &denoisingParams.enableDenoiser);
        ImGui::Separator();

        // Boolean parameters (pass controls)
        if (ImGui::TreeNode("Pass Controls"))
        {
            for (auto &itempair : denoisingParams.GetBooleanValueList())
            {
                if (itempair.first != &denoisingParams.enableDenoiser) // Skip the master enable checkbox
                {
                    ImGui::Checkbox(itempair.second.c_str(), itempair.first);
                }
            }
            ImGui::TreePop();
        }

        // Float parameters
        if (ImGui::TreeNode("Float Parameters"))
        {
            for (auto &item : denoisingParams.GetValueList())
            {
                if (ImGui::InputFloat(item.second.c_str(), item.first))
                {
                    *item.first = max(*item.first, 0.00001f);
                }
            }
            ImGui::TreePop();
        }

        // Integer parameters
        if (ImGui::TreeNode("Integer Parameters"))
        {
            for (auto &item : denoisingParams.GetIntValueList())
            {
                ImGui::InputInt(item.second.c_str(), item.first);
                *item.first = max(*item.first, 0);
            }
            ImGui::TreePop();
        }
    }



    if (ImGui::CollapsingHeader("Post Processing", ImGuiTreeNodeFlags_None))
    {
        PostProcessParams &postProcessParams = GlobalSettings::GetPostProcessParams();
        for (auto &item : postProcessParams.GetValueList())
            ImGui::SliderFloat(
                std::get<1>(item).c_str(),
                std::get<0>(item),
                std::get<2>(item),
                std::get<3>(item),
                "%.8f",
                std::get<4>(item) ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None);
    }

    if (ImGui::CollapsingHeader("Sky", ImGuiTreeNodeFlags_None))
    {
        SkyParams &skyParams = GlobalSettings::GetSkyParams();
        for (auto &item : skyParams.GetValueList())
        {
            if (ImGui::SliderFloat(std::get<1>(item).c_str(),
                                   std::get<0>(item),
                                   std::get<2>(item),
                                   std::get<3>(item),
                                   "%.8f",
                                   std::get<4>(item) ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None))
            {
                skyParams.needRegenerate = true;
            }
        }
    }

    if (ImGui::CollapsingHeader("Character Movement", ImGuiTreeNodeFlags_None))
    {
        CharacterMovementParams &characterMovementParams = GlobalSettings::GetCharacterMovementParams();
        for (auto &item : characterMovementParams.GetValueList())
        {
            ImGui::SliderFloat(
                std::get<1>(item).c_str(),
                std::get<0>(item),
                std::get<2>(item),
                std::get<3>(item),
                "%.3f",
                std::get<4>(item) ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None);
        }
    }

    if (ImGui::CollapsingHeader("Character Animation", ImGuiTreeNodeFlags_None))
    {
        CharacterAnimationParams &characterAnimationParams = GlobalSettings::GetCharacterAnimationParams();
        for (auto &item : characterAnimationParams.GetValueList())
        {
            ImGui::SliderFloat(
                std::get<1>(item).c_str(),
                std::get<0>(item),
                std::get<2>(item),
                std::get<3>(item),
                "%.3f",
                std::get<4>(item) ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None);
        }
    }

    if (ImGui::CollapsingHeader("Camera Movement", ImGuiTreeNodeFlags_None))
    {
        CameraMovementParams &cameraMovementParams = GlobalSettings::GetCameraMovementParams();
        for (auto &item : cameraMovementParams.GetValueList())
        {
            ImGui::SliderFloat(
                std::get<1>(item).c_str(),
                std::get<0>(item),
                std::get<2>(item),
                std::get<3>(item),
                "%.6f",
                std::get<4>(item) ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None);
        }
    }

    if (ImGui::CollapsingHeader("Rendering Settings", ImGuiTreeNodeFlags_None))
    {
        RenderingParams &renderingParams = GlobalSettings::GetRenderingParams();
        
        // Float parameters
        for (auto &item : renderingParams.GetValueList())
        {
            ImGui::SliderFloat(
                std::get<1>(item).c_str(),
                std::get<0>(item),
                std::get<2>(item),
                std::get<3>(item),
                "%.1f",
                std::get<4>(item) ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None);
        }
        
        // Boolean parameters
        for (auto &item : renderingParams.GetBooleanValueList())
        {
            ImGui::Checkbox(item.second.c_str(), item.first);
        }
        
        // Integer parameters
        for (auto &item : renderingParams.GetIntValueList())
        {
            ImGui::InputInt(item.second.c_str(), item.first);
        }
    }

    if (ImGui::CollapsingHeader("Global Settings Management", ImGuiTreeNodeFlags_None))
    {
        static char globalSettingsFileName[256] = "data/settings/global_settings.yaml";

        ImGui::Text("Save/Load All Global Settings");
        ImGui::InputText("Settings File", globalSettingsFileName, sizeof(globalSettingsFileName));

        if (ImGui::Button("Save Global Settings"))
        {
            GlobalSettings::Get().SaveToYAML(std::string(globalSettingsFileName));
        }
        
        ImGui::SameLine();
        
        if (ImGui::Button("Load Global Settings"))
        {
            GlobalSettings::Get().LoadFromYAML(std::string(globalSettingsFileName));
        }

        ImGui::Separator();
        
        if (ImGui::Button("Reset to Defaults"))
        {
            // Reset all settings to default values
            GlobalSettings::Get().denoisingParams = DenoisingParams{};
            GlobalSettings::Get().postProcessParams = PostProcessParams{};
            GlobalSettings::Get().skyParams = SkyParams{};
            GlobalSettings::Get().characterMovementParams = CharacterMovementParams{};
            GlobalSettings::Get().characterAnimationParams = CharacterAnimationParams{};
            GlobalSettings::Get().cameraMovementParams = CameraMovementParams{};
            GlobalSettings::Get().renderingParams = RenderingParams{};
        }
    }

    if (ImGui::CollapsingHeader("Scene Management", ImGuiTreeNodeFlags_None))
    {
        static char sceneFileName[256] = "data/scene/scene_export.yaml";

        ImGui::Text("Save Current Camera as YAML Scene");
        ImGui::InputText("Filename", sceneFileName, sizeof(sceneFileName));

        if (ImGui::Button("Save Scene"))
        {
            // Create scene config from current camera
            SceneConfig currentScene;
            currentScene.camera.position = camera.pos;
            currentScene.camera.direction = camera.dir;
            currentScene.camera.up = Float3(0.0f, 1.0f, 0.0f); // Standard up vector
            currentScene.camera.fov = 90.0f; // Default FOV, should match camera default

            // Save to file
            SceneConfigParser::SaveToFile(std::string(sceneFileName), currentScene);
        }

        ImGui::Separator();

        static char loadFileName[256] = "data/scene/scene_export.yaml";
        ImGui::Text("Load YAML Scene");
        ImGui::InputText("Load File", loadFileName, sizeof(loadFileName));

                if (ImGui::Button("Load Scene"))
        {
            SceneConfig loadedScene;
            if (SceneConfigParser::LoadFromFile(std::string(loadFileName), loadedScene))
            {
                // Apply loaded scene to current camera
                auto &currentCamera = RenderCamera::Get().camera;
                currentCamera.pos = loadedScene.camera.position;

                // Convert direction to yaw/pitch for proper camera handling
                Float2 yawPitch = DirToYawPitch(loadedScene.camera.direction.normalize());
                currentCamera.yaw = yawPitch.x;
                currentCamera.pitch = yawPitch.y;

                currentCamera.update();
            }
        }

        ImGui::Separator();
        ImGui::Text("Current Camera Info:");
        ImGui::Text("Position: (%.2f, %.2f, %.2f)", camera.pos.x, camera.pos.y, camera.pos.z);
        ImGui::Text("Direction: (%.2f, %.2f, %.2f)", camera.dir.x, camera.dir.y, camera.dir.z);
    }

    ImGui::End();
}

void UI::render()
{
    auto &backend = Backend::Get();

    ImGui::Render();
    glViewport(0, 0, backend.getWidth(), backend.getHeight());
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}