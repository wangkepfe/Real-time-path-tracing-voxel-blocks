#include "core/UI.h"
#include "core/OptixRenderer.h"
#include "core/Backend.h"
#include "core/InputHandler.h"

namespace jazzfusion
{

void UI::init()
{
    ImGui::CreateContext();
    ImGui_ImplGlfwGL3_Init(Backend::Get().getWindow(), true);
    ImGui_ImplGlfwGL3_NewFrame();
    ImGui::EndFrame();
}

void UI::clear()
{
    ImGui_ImplGlfwGL3_Shutdown();
    ImGui::DestroyContext();
}

void UI::update()
{
    ImGui_ImplGlfwGL3_NewFrame();

    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiSetCond_FirstUseEver);

    auto& backend = Backend::Get();

    if (!ImGui::Begin("Render Settings", nullptr, 0))
    {
        ImGui::End();
        return;
    }

    if (ImGui::CollapsingHeader("Tone Mapping", 0))
    {
        ImGui::SliderFloat("Gain", backend.getToneMapGain(), 0.1f, 100.0f, "%.3f", 10.0f);
        ImGui::SliderFloat("Max White", backend.getToneMapMaxWhite(), 0.1f, 100.0f, "%.3f", 10.0f);
    }

    ImGui::End();
}

void UI::render()
{
    ImGui::Render();
    ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());
}

}