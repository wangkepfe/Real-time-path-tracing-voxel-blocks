#include "core/UI.h"
#include "core/OptixRenderer.h"
#include "core/Backend.h"
#include "core/InputHandler.h"

namespace jazzfusion
{

void UI::init()
{
    auto& backend = Backend::Get();

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(backend.getWindow(), true);
    ImGui_ImplOpenGL3_Init(backend.GlslVersion);
}

void UI::clear()
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

    if (ImGui::CollapsingHeader("Tone Mapping", 0))
    {
        ImGui::SliderFloat("Gain", backend.getToneMapGain(), 0.1f, 100.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Max White", backend.getToneMapMaxWhite(), 0.1f, 100.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    }

    ImGui::End();
}

void UI::render()
{
    auto& backend = Backend::Get();

    ImGui::Render();
    glViewport(0, 0, backend.getWidth(), backend.getHeight());
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

}