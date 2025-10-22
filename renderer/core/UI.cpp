#include "core/UI.h"
#include "core/OptixRenderer.h"
#include "core/Backend.h"
#include "core/InputHandler.h"
#include "core/GlobalSettings.h"
#include "core/RenderCamera.h"
#include "core/WorldSceneManager.h"
#include "core/Scene.h"
#include "core/CameraController.h"
#include "core/Character.h"
#include "voxelengine/VoxelEngine.h"
#include "postprocessing/PostProcessor.h"

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
    const char *currentCameraMode = "Unknown";
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
        auto &movement = character->getMovement();
        auto &physics = character->getPhysics();

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
    auto &voxelEngine = VoxelEngine::Get();
    if (voxelEngine.centerBlockInfo.hasValidBlock)
    {
        ImGui::Text("Block ID: %d", voxelEngine.centerBlockInfo.blockId);
        ImGui::Text("Block Name: %s", voxelEngine.centerBlockInfo.blockName.c_str());
        ImGui::Text("Position: (%d, %d, %d)",
                    voxelEngine.centerBlockInfo.position.x,
                    voxelEngine.centerBlockInfo.position.y,
                    voxelEngine.centerBlockInfo.position.z);
    }
    else
    {
        ImGui::Text("No block in center crosshair");
    }

    if (ImGui::CollapsingHeader("Temporal Denoising", 0))
    {
        DenoisingParams &denoisingParams = GlobalSettings::GetDenoisingParams(); // Boolean parameters (pass controls)
        if (ImGui::TreeNode("Pass Controls"))
        {
            for (auto &itempair : denoisingParams.GetBooleanValueList())
            {
                ImGui::Checkbox(itempair.second.c_str(), itempair.first);
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
        ToneMappingParams &toneMappingParams = GlobalSettings::GetToneMappingParams();
        PostProcessingPipelineParams &pipelineParams = GlobalSettings::GetPostProcessingPipelineParams();

        // Exposure Control Section
        if (ImGui::TreeNode("Exposure Control"))
        {
            ImGui::Checkbox("Enable Auto Exposure", &pipelineParams.enableAutoExposure);

            // Real-time computed exposure display
            auto &postProcessor = PostProcessor::Get();
            float computedExposure = postProcessor.GetComputedExposure();
            ImGui::Separator();
            ImGui::Text("Real-time Exposure: %.4f", computedExposure);
            ImGui::Text("Real-time Exposure (EV): %.2f", log2f(computedExposure));
            ImGui::Separator();

            if (!pipelineParams.enableAutoExposure)
            {
                ImGui::SliderFloat("Manual Exposure", &toneMappingParams.manualExposure, 0.1f, 20.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
                ImGui::SameLine();
                ImGui::InputFloat("##manual_exposure_input", &toneMappingParams.manualExposure, 0.1f, 1.0f, "%.2f");
            }
            else
            {
                ImGui::SliderFloat("Exposure Compensation (EV)", &pipelineParams.exposureCompensation, -5.0f, 5.0f, "%.2f");
                ImGui::SameLine();
                ImGui::InputFloat("##exp_comp_input", &pipelineParams.exposureCompensation, 0.1f, 1.0f, "%.2f");

                ImGui::SliderFloat("Auto Exposure Speed", &pipelineParams.exposureSpeed, 0.1f, 5.0f, "%.2f");
                ImGui::SameLine();
                ImGui::InputFloat("##exp_speed_input", &pipelineParams.exposureSpeed, 0.1f, 0.5f, "%.2f");

                ImGui::SliderFloat("Auto Exposure Min (EV)", &pipelineParams.exposureMin, -12.0f, 0.0f, "%.1f");
                ImGui::SameLine();
                ImGui::InputFloat("##exp_min_input", &pipelineParams.exposureMin, 0.5f, 1.0f, "%.1f");

                ImGui::SliderFloat("Auto Exposure Max (EV)", &pipelineParams.exposureMax, 0.0f, 12.0f, "%.1f");
                ImGui::SameLine();
                ImGui::InputFloat("##exp_max_input", &pipelineParams.exposureMax, 0.5f, 1.0f, "%.1f");

                ImGui::SliderFloat("Histogram Min %", &pipelineParams.histogramMinPercent, 0.0f, 50.0f, "%.1f");
                ImGui::SameLine();
                ImGui::InputFloat("##hist_min_input", &pipelineParams.histogramMinPercent, 1.0f, 5.0f, "%.1f");

                ImGui::SliderFloat("Histogram Max %", &pipelineParams.histogramMaxPercent, 50.0f, 100.0f, "%.1f");
                ImGui::SameLine();
                ImGui::InputFloat("##hist_max_input", &pipelineParams.histogramMaxPercent, 1.0f, 5.0f, "%.1f");

                ImGui::SliderFloat("Target Luminance", &pipelineParams.targetLuminance, 0.05f, 0.5f, "%.2f");
                ImGui::SameLine();
                ImGui::InputFloat("##target_lum_input", &pipelineParams.targetLuminance, 0.01f, 0.05f, "%.2f");
            }
            ImGui::TreePop();
        }

        // Bloom Section
        if (ImGui::TreeNode("Bloom Effects"))
        {
            ImGui::Checkbox("Enable Bloom", &pipelineParams.enableBloom);
            if (pipelineParams.enableBloom)
            {
                ImGui::SliderFloat("Bloom Threshold", &pipelineParams.bloomThreshold, 0.0f, 5.0f, "%.2f");
                ImGui::SameLine();
                ImGui::InputFloat("##bloom_thresh_input", &pipelineParams.bloomThreshold, 0.1f, 0.5f, "%.2f");

                ImGui::SliderFloat("Bloom Intensity", &pipelineParams.bloomIntensity, 0.0f, 2.0f, "%.2f");
                ImGui::SameLine();
                ImGui::InputFloat("##bloom_intensity_input", &pipelineParams.bloomIntensity, 0.01f, 0.1f, "%.2f");

                ImGui::SliderFloat("Bloom Radius", &pipelineParams.bloomRadius, 0.5f, 5.0f, "%.2f");
                ImGui::SameLine();
                ImGui::InputFloat("##bloom_radius_input", &pipelineParams.bloomRadius, 0.1f, 0.5f, "%.2f");
            }
            ImGui::TreePop();
        }

        // Vignette Section
        if (ImGui::TreeNode("Vignette Effects"))
        {
            ImGui::Checkbox("Enable Vignette", &pipelineParams.enableVignette);
            if (pipelineParams.enableVignette)
            {
                ImGui::SliderFloat("Vignette Strength", &pipelineParams.vignetteStrength, 0.0f, 2.0f, "%.2f");
                ImGui::SameLine();
                ImGui::InputFloat("##vignette_str_input", &pipelineParams.vignetteStrength, 0.01f, 0.1f, "%.2f");

                ImGui::SliderFloat("Vignette Radius", &pipelineParams.vignetteRadius, 0.1f, 1.5f, "%.2f");
                ImGui::SameLine();
                ImGui::InputFloat("##vignette_rad_input", &pipelineParams.vignetteRadius, 0.01f, 0.1f, "%.2f");

                ImGui::SliderFloat("Vignette Smoothness", &pipelineParams.vignetteSmoothness, 0.1f, 1.0f, "%.2f");
                ImGui::SameLine();
                ImGui::InputFloat("##vignette_smooth_input", &pipelineParams.vignetteSmoothness, 0.01f, 0.1f, "%.2f");
            }
            ImGui::TreePop();
        }

        // Lens Flare Section
        if (ImGui::TreeNode("Lens Flare Effects"))
        {
            ImGui::Checkbox("Enable Lens Flare", &pipelineParams.enableLensFlare);
            if (pipelineParams.enableLensFlare)
            {
                // Use DragFloat for precise control with small values
                // Tip: Hold Ctrl while dragging for slower, more precise control
                ImGui::DragFloat("Lens Flare Intensity", &pipelineParams.lensFlareIntensity, 0.001f, 0.0f, 1.0f, "%.4f");
                ImGui::SameLine();
                ImGui::InputFloat("##intensity_input", &pipelineParams.lensFlareIntensity, 0.001f, 0.01f, "%.4f");

                ImGui::DragFloat("Ghost Spacing", &pipelineParams.lensFlareGhostSpacing, 0.001f, 0.01f, 2.0f, "%.4f");
                ImGui::SameLine();
                ImGui::InputFloat("##spacing_input", &pipelineParams.lensFlareGhostSpacing, 0.001f, 0.01f, "%.4f");

                ImGui::SliderInt("Ghost Count", &pipelineParams.lensFlareGhostCount, 1, 8);

                ImGui::DragFloat("Halo Radius", &pipelineParams.lensFlareHaloRadius, 0.001f, 0.01f, 2.0f, "%.4f");
                ImGui::SameLine();
                ImGui::InputFloat("##halo_input", &pipelineParams.lensFlareHaloRadius, 0.001f, 0.01f, "%.4f");

                ImGui::DragFloat("Sun Size", &pipelineParams.lensFlareSunSize, 0.0001f, 0.0001f, 0.1f, "%.5f");
                ImGui::SameLine();
                ImGui::InputFloat("##sun_input", &pipelineParams.lensFlareSunSize, 0.0001f, 0.001f, "%.5f");

                ImGui::DragFloat("Chromatic Aberration", &pipelineParams.lensFlareDistortion, 0.0001f, 0.0f, 0.5f, "%.5f");
                ImGui::SameLine();
                ImGui::InputFloat("##aberration_input", &pipelineParams.lensFlareDistortion, 0.0001f, 0.001f, "%.5f");

            }
            ImGui::TreePop();
        }

        // Tone Mapping Section
        if (ImGui::TreeNode("Tone Mapping"))
        {
            const char *curveNames[] = {
                "Narkowicz ACES (Fast)",
                "Uncharted 2 Filmic",
                "Reinhard"};
            int curveIndex = static_cast<int>(toneMappingParams.curve);
            if (ImGui::Combo("Tone Mapping Curve", &curveIndex, curveNames, 3))
            {
                toneMappingParams.curve = static_cast<ToneMappingParams::ToneMappingCurve>(curveIndex);
            }

            ImGui::SliderFloat("Highlight Desaturation", &toneMappingParams.highlightDesaturation, 0.0f, 1.0f, "%.2f");
            ImGui::SameLine();
            ImGui::InputFloat("##highlight_desat_input", &toneMappingParams.highlightDesaturation, 0.01f, 0.1f, "%.2f");

            ImGui::SliderFloat("White Point", &toneMappingParams.whitePoint, 1.0f, 20.0f, "%.1f");
            ImGui::SameLine();
            ImGui::InputFloat("##white_point_input", &toneMappingParams.whitePoint, 0.1f, 1.0f, "%.1f");
            ImGui::TreePop();
        }

        // Color Grading Section
        if (ImGui::TreeNode("Color Grading"))
        {
            ImGui::SliderFloat("Contrast", &toneMappingParams.contrast, 0.5f, 2.0f, "%.2f");
            ImGui::SameLine();
            ImGui::InputFloat("##contrast_input", &toneMappingParams.contrast, 0.01f, 0.1f, "%.2f");

            ImGui::SliderFloat("Saturation", &toneMappingParams.saturation, 0.0f, 2.0f, "%.2f");
            ImGui::SameLine();
            ImGui::InputFloat("##saturation_input", &toneMappingParams.saturation, 0.01f, 0.1f, "%.2f");

            ImGui::SliderFloat("Lift (Shadows)", &toneMappingParams.lift, -0.5f, 0.5f, "%.3f");
            ImGui::SameLine();
            ImGui::InputFloat("##lift_input", &toneMappingParams.lift, 0.001f, 0.01f, "%.3f");

            ImGui::SliderFloat("Gain (Highlights)", &toneMappingParams.gain, 0.5f, 2.0f, "%.2f");
            ImGui::SameLine();
            ImGui::InputFloat("##gain_input", &toneMappingParams.gain, 0.01f, 0.1f, "%.2f");

            ImGui::TreePop();
        }

        // Output Section
        if (ImGui::TreeNode("Output"))
        {
            ImGui::Text("Output: sRGB");
            ImGui::Checkbox("Enable Chromatic Adaptation", &toneMappingParams.enableChromaticAdaptation);
            ImGui::TreePop();
        }

        // Presets Section
        if (ImGui::TreeNode("Presets"))
        {
            if (ImGui::Button("Cinematic"))
            {
                // Tone mapping
                toneMappingParams.curve = ToneMappingParams::CURVE_UNCHARTED2;
                toneMappingParams.contrast = 1.1f;
                toneMappingParams.saturation = 1.05f;
                toneMappingParams.highlightDesaturation = 0.9f;
                // Pipeline
                pipelineParams.enableBloom = true;
                pipelineParams.bloomThreshold = 1.2f;
                pipelineParams.bloomIntensity = 0.4f;
                pipelineParams.bloomRadius = 3.0f;
                pipelineParams.enableAutoExposure = true;
                pipelineParams.exposureSpeed = 1.5f;
                pipelineParams.histogramMinPercent = 35.0f;
                pipelineParams.histogramMaxPercent = 85.0f;
                // Vignette
                pipelineParams.enableVignette = true;
                pipelineParams.vignetteStrength = 0.3f;
                pipelineParams.vignetteRadius = 0.8f;
                pipelineParams.vignetteSmoothness = 0.4f;
                // Lens Flare - Cinematic preset with subtle effects
                pipelineParams.enableLensFlare = true;
                pipelineParams.lensFlareIntensity = 0.0150f;
                pipelineParams.lensFlareGhostSpacing = 0.0800f;
                pipelineParams.lensFlareGhostCount = 4;
                pipelineParams.lensFlareHaloRadius = 0.1200f;
                pipelineParams.lensFlareSunSize = 0.0080f;
                pipelineParams.lensFlareDistortion = 0.0020f;
            }
            ImGui::SameLine();
            if (ImGui::Button("Vibrant"))
            {
                // Tone mapping
                toneMappingParams.curve = ToneMappingParams::CURVE_NARKOWICZ_ACES;
                toneMappingParams.contrast = 1.05f;
                toneMappingParams.saturation = 1.2f;
                toneMappingParams.highlightDesaturation = 0.7f;
                // Pipeline
                pipelineParams.enableBloom = true;
                pipelineParams.bloomThreshold = 1.5f;
                pipelineParams.bloomIntensity = 0.25f;
                pipelineParams.bloomRadius = 2.0f;
                pipelineParams.enableAutoExposure = true;
                pipelineParams.exposureSpeed = 2.0f;
                pipelineParams.histogramMinPercent = 40.0f;
                pipelineParams.histogramMaxPercent = 80.0f;
                // Vignette
                pipelineParams.enableVignette = true;
                pipelineParams.vignetteStrength = 0.4f;
                pipelineParams.vignetteRadius = 0.7f;
                pipelineParams.vignetteSmoothness = 0.5f;
                // Lens Flare - Vibrant preset with stronger effects
                pipelineParams.enableLensFlare = true;
                pipelineParams.lensFlareIntensity = 0.0250f;
                pipelineParams.lensFlareGhostSpacing = 0.1200f;
                pipelineParams.lensFlareGhostCount = 5;
                pipelineParams.lensFlareHaloRadius = 0.1800f;
                pipelineParams.lensFlareSunSize = 0.0120f;
                pipelineParams.lensFlareDistortion = 0.0040f;
            }
            if (ImGui::Button("Realistic"))
            {
                // Tone mapping
                toneMappingParams.curve = ToneMappingParams::CURVE_NARKOWICZ_ACES;
                toneMappingParams.contrast = 1.0f;
                toneMappingParams.saturation = 1.0f;
                toneMappingParams.highlightDesaturation = 0.8f;
                toneMappingParams.lift = 0.0f;
                toneMappingParams.gain = 1.0f;
                // Pipeline
                pipelineParams.enableBloom = false;
                pipelineParams.enableAutoExposure = true;
                pipelineParams.exposureSpeed = 1.0f;
                pipelineParams.histogramMinPercent = 45.0f;
                pipelineParams.histogramMaxPercent = 75.0f;
                pipelineParams.targetLuminance = 0.18f;
                // Vignette
                pipelineParams.enableVignette = false;
                pipelineParams.vignetteStrength = 0.2f;
                pipelineParams.vignetteRadius = 0.9f;
                pipelineParams.vignetteSmoothness = 0.3f;
                // Lens Flare - Realistic preset with very subtle effects
                pipelineParams.enableLensFlare = true;
                pipelineParams.lensFlareIntensity = 0.0080f;
                pipelineParams.lensFlareGhostSpacing = 0.0600f;
                pipelineParams.lensFlareGhostCount = 3;
                pipelineParams.lensFlareHaloRadius = 0.0800f;
                pipelineParams.lensFlareSunSize = 0.0050f;
                pipelineParams.lensFlareDistortion = 0.0010f;
            }
            ImGui::SameLine();

            if (ImGui::Button("Reset All"))
            {
                toneMappingParams = ToneMappingParams{};
                pipelineParams = PostProcessingPipelineParams{};
            }
            ImGui::TreePop();
        }
    }

    if (ImGui::CollapsingHeader("Sky", ImGuiTreeNodeFlags_None))
    {
        SkyParams &skyParams = GlobalSettings::GetSkyParams();
        int sky_idx = 0;
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
            ImGui::SameLine();
            std::string input_id = "##sky_input_" + std::to_string(sky_idx++);
            if (ImGui::InputFloat(input_id.c_str(), std::get<0>(item), 0.01f, 0.1f, "%.8f"))
            {
                skyParams.needRegenerate = true;
            }
        }
    }

    if (ImGui::CollapsingHeader("Character Movement", ImGuiTreeNodeFlags_None))
    {
        CharacterMovementParams &characterMovementParams = GlobalSettings::GetCharacterMovementParams();
        int char_mov_idx = 0;
        for (auto &item : characterMovementParams.GetValueList())
        {
            ImGui::SliderFloat(
                std::get<1>(item).c_str(),
                std::get<0>(item),
                std::get<2>(item),
                std::get<3>(item),
                "%.3f",
                std::get<4>(item) ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None);
            ImGui::SameLine();
            std::string input_id = "##char_mov_input_" + std::to_string(char_mov_idx++);
            ImGui::InputFloat(input_id.c_str(), std::get<0>(item), 0.01f, 0.1f, "%.3f");
        }
    }

    if (ImGui::CollapsingHeader("Character Animation", ImGuiTreeNodeFlags_None))
    {
        CharacterAnimationParams &characterAnimationParams = GlobalSettings::GetCharacterAnimationParams();
        int char_anim_idx = 0;
        for (auto &item : characterAnimationParams.GetValueList())
        {
            ImGui::SliderFloat(
                std::get<1>(item).c_str(),
                std::get<0>(item),
                std::get<2>(item),
                std::get<3>(item),
                "%.3f",
                std::get<4>(item) ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None);
            ImGui::SameLine();
            std::string input_id = "##char_anim_input_" + std::to_string(char_anim_idx++);
            ImGui::InputFloat(input_id.c_str(), std::get<0>(item), 0.01f, 0.1f, "%.3f");
        }
    }

    if (ImGui::CollapsingHeader("Camera Movement", ImGuiTreeNodeFlags_None))
    {
        CameraMovementParams &cameraMovementParams = GlobalSettings::GetCameraMovementParams();
        int cam_mov_idx = 0;
        for (auto &item : cameraMovementParams.GetValueList())
        {
            ImGui::SliderFloat(
                std::get<1>(item).c_str(),
                std::get<0>(item),
                std::get<2>(item),
                std::get<3>(item),
                "%.6f",
                std::get<4>(item) ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None);
            ImGui::SameLine();
            std::string input_id = "##cam_mov_input_" + std::to_string(cam_mov_idx++);
            ImGui::InputFloat(input_id.c_str(), std::get<0>(item), 0.0001f, 0.001f, "%.6f");
        }
    }

    if (ImGui::CollapsingHeader("Rendering Settings", ImGuiTreeNodeFlags_None))
    {
        RenderingParams &renderingParams = GlobalSettings::GetRenderingParams();

        // Float parameters
        int render_idx = 0;
        for (auto &item : renderingParams.GetValueList())
        {
            ImGui::SliderFloat(
                std::get<1>(item).c_str(),
                std::get<0>(item),
                std::get<2>(item),
                std::get<3>(item),
                "%.1f",
                std::get<4>(item) ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None);
            ImGui::SameLine();
            std::string input_id = "##render_input_" + std::to_string(render_idx++);
            ImGui::InputFloat(input_id.c_str(), std::get<0>(item), 0.1f, 1.0f, "%.1f");
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
            GlobalSettings::Get().toneMappingParams = ToneMappingParams{};
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
            WorldSceneManager::SaveScene(std::string(sceneFileName), camera, inputHandler);
        }

        ImGui::Separator();

        static char loadFileName[256] = "data/scene/scene_export.yaml";
        ImGui::Text("Load YAML Scene");
        ImGui::InputText("Load File", loadFileName, sizeof(loadFileName));

        if (ImGui::Button("Load Scene"))
        {
            WorldSceneManager::LoadScene(std::string(loadFileName), inputHandler);
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
