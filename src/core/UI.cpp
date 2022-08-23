#include "core/UI.h"
#include "core/OptixRenderer.h"
#include "core/Backend.h"
#include "core/InputHandler.h"
#include "core/GlobalSettings.h"

namespace jazzfusion
{

void UI::init()
{
    auto& backend = Backend::Get();

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
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

    auto& backend = Backend::Get();
    auto& renderer = OptixRenderer::Get();

    auto& renderPassSettings = GlobalSettings::GetRenderPassSettings();

    if (!ImGui::Begin("Render Settings", nullptr, 0))
    {
        ImGui::End();
        return;
    }

    auto& camera = renderer.getCamera();

    ImGui::Text("ms/frame: %.2f FPS: %.1f", 1000.0f / backend.getCurrentFPS(), backend.getCurrentFPS());
    ImGui::Text("Resolution: (%d, %d)", backend.getCurrentRenderWidth(), backend.getCurrentRenderWidth() / 16 * 9);
    ImGui::Text("Scale: %.1f %%", backend.getCurrentRenderWidth() / (float)backend.getWidth() * 100.0f);
    ImGui::Text("Camera pos=(%.2f, %.2f, %.2f)", camera.pos.x, camera.pos.y, camera.pos.z);
    ImGui::Text("Camera dir=(%.2f, %.2f, %.2f)", camera.dir.x, camera.dir.y, camera.dir.z);
    ImGui::Text("Accumulation Counter = %d", backend.getAccumulationCounter());

    if (ImGui::CollapsingHeader("Render Passes", 0))
    {
        auto& list = renderPassSettings.GetValueList();
        for (auto& itempair : list)
        {
            ImGui::Checkbox(itempair.second.c_str(), itempair.first);
        }
    }

    if (ImGui::CollapsingHeader("Temporal Denoising", 0))
    {
        DenoisingParams& denoisingParams = GlobalSettings::GetDenoisingParams();
        for (auto& item : denoisingParams.GetValueList())
        {
            if (ImGui::InputFloat(item.second.c_str(), item.first))
            {
                *item.first = max(*item.first, 0.00001f);
            }
        }
        for (auto& itempair : denoisingParams.GetBooleanValueList())
        {
            ImGui::Checkbox(itempair.second.c_str(), itempair.first);
        }
    }

    if (ImGui::CollapsingHeader("Post Processing", ImGuiTreeNodeFlags_None))
    {
        PostProcessParams& postProcessParams = GlobalSettings::GetPostProcessParams();
        for (auto& item : postProcessParams.GetValueList())
            ImGui::SliderFloat(
                std::get<1>(item).c_str(),
                std::get<0>(item),
                std::get<2>(item),
                std::get<3>(item),
                "%.2f",
                std::get<4>(item) ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None);
    }

    if (ImGui::CollapsingHeader("Sky", ImGuiTreeNodeFlags_None))
    {
        SkyParams& skyParams = GlobalSettings::GetSkyParams();
        for (auto& item : skyParams.GetValueList())
        {
            if (ImGui::SliderFloat(std::get<1>(item).c_str(),
                std::get<0>(item),
                std::get<2>(item),
                std::get<3>(item),
                "%.3f",
                std::get<4>(item) ? ImGuiSliderFlags_Logarithmic : ImGuiSliderFlags_None))
            {
                skyParams.needRegenerate = true;
            }
        }

        ImGui::End();
    }
}

void UI::render()
{
    auto& backend = Backend::Get();

    ImGui::Render();
    glViewport(0, 0, backend.getWidth(), backend.getHeight());
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

}