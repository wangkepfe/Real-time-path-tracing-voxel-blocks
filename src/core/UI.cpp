#include "core/UI.h"
#include "core/OptixRenderer.h"
#include "core/Backend.h"

namespace jazzfusion
{

void UI::init()
{
    m_guiState = GUI_STATE_NONE;
    m_isVisibleGUI = true;
    m_mouseSpeedRatio = 10.0f;

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

    if (!m_isVisibleGUI)
    {
        return;
    }

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

void UI::eventHandler()
{
    ImGuiIO const& io = ImGui::GetIO();

    if (ImGui::IsKeyPressed(' ', false))
    {
        m_isVisibleGUI = !m_isVisibleGUI;
    }

    const ImVec2 mousePosition = ImGui::GetMousePos();
    const int x = int(mousePosition.x);
    const int y = int(mousePosition.y);

    Camera& camera = OptixRenderer::Get().getCamera();

    switch (m_guiState)
    {
    case GUI_STATE_NONE:
        if (!io.WantCaptureMouse)
        {
            if (ImGui::IsMouseDown(0))
            {
                camera.setBaseCoordinates(x, y);
                m_guiState = GUI_STATE_ORBIT;
            }
            else if (ImGui::IsMouseDown(1))
            {
                camera.setBaseCoordinates(x, y);
                m_guiState = GUI_STATE_DOLLY;
            }
            else if (ImGui::IsMouseDown(2))
            {
                camera.setBaseCoordinates(x, y);
                m_guiState = GUI_STATE_PAN;
            }
            else if (io.MouseWheel != 0.0f)
            {
                camera.zoom(io.MouseWheel);
            }
        }
        break;

    case GUI_STATE_ORBIT:
        if (ImGui::IsMouseReleased(0))
        {
            m_guiState = GUI_STATE_NONE;
        }
        else
        {
            camera.orbit(x, y);
        }
        break;

    case GUI_STATE_DOLLY:
        if (ImGui::IsMouseReleased(1))
        {
            m_guiState = GUI_STATE_NONE;
        }
        else
        {
            camera.dolly(x, y);
        }
        break;

    case GUI_STATE_PAN:
        if (ImGui::IsMouseReleased(2))
        {
            m_guiState = GUI_STATE_NONE;
        }
        else
        {
            camera.pan(x, y);
        }
        break;
    }
}

}