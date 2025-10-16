#include "ui/GameUIManager.h"

#include "ui/GameUIActionHandler.h"
#include "ui/MainMenuController.h"

#include "renderer/core/InputHandler.h"

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/FontEngineInterface.h>
#include <RmlUi/Core/Log.h>
#include <RmlUi/Core/Types.h>
#include <RmlUi/Core/Vector2.h>

#include "RmlUi_Renderer_GL3.h"
#include "RmlUi_Platform_GLFW.h"

#include <GLFW/glfw3.h>
#include <filesystem>

namespace
{
    constexpr const char* kContextName = "game_ui";
    constexpr const char* kFontRegular = "data/ui/fonts/Roboto-Regular.ttf";
    constexpr const char* kFontBold = "data/ui/fonts/Roboto-Medium.ttf";
}

GameUIManager& GameUIManager::Get()
{
    static GameUIManager instance;
    return instance;
}

bool GameUIManager::Initialize(GLFWwindow* window, int width, int height)
{
    if (m_initialized)
    {
        return true;
    }

    if (!window)
    {
        return false;
    }

    m_window = window;
    m_viewportWidth = width;
    m_viewportHeight = height;

    m_systemInterface = std::make_unique<SystemInterface_GLFW>();
    m_systemInterface->SetWindow(window);
    Rml::SetSystemInterface(m_systemInterface.get());

    if (!m_gl3Initialized)
    {
        Rml::String message;
        if (!RmlGL3::Initialize(&message))
        {
            Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to initialize RmlGL3 backend: %s", message.c_str());
            return false;
        }
        Rml::Log::Message(Rml::Log::LT_INFO, "%s", message.c_str());
        m_gl3Initialized = true;
    }

    m_renderInterface = std::make_unique<RenderInterface_GL3>();
    if (!m_renderInterface || !(*m_renderInterface))
    {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create RenderInterface_GL3.");
        return false;
    }

    Rml::SetRenderInterface(m_renderInterface.get());

    if (!Rml::Initialise())
    {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to initialise RmlUi core.");
        return false;
    }

    m_renderInterface->SetViewport(width, height);

    float dp_ratio = 1.0f;
    glfwGetWindowContentScale(window, &dp_ratio, nullptr);
    m_contentScale = dp_ratio;

    m_context = Rml::CreateContext(kContextName, Rml::Vector2i(width, height));
    if (!m_context)
    {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to create RmlUi context '%s'.", kContextName);
        return false;
    }

    m_context->SetDensityIndependentPixelRatio(dp_ratio);

    BindDefaultFonts();

    m_mainMenu = std::make_unique<MainMenuController>(*m_context, *this);
    if (!m_mainMenu->LoadDocument("data/ui/main_menu.rml"))
    {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to load main menu document.");
    }

    SetActionHandler(std::make_shared<DefaultGameUIActionHandler>(m_window));

    glfwSetInputMode(window, GLFW_LOCK_KEY_MODS, GLFW_TRUE);

    ApplyState(GameUIState::MainMenu);

    m_initialized = true;
    return true;
}

void GameUIManager::Shutdown()
{
    if (!m_initialized)
    {
        return;
    }

    m_mainMenu.reset();

    if (m_context)
    {
        Rml::RemoveContext(kContextName);
        m_context = nullptr;
    }

    Rml::SetRenderInterface(nullptr);
    Rml::SetSystemInterface(nullptr);

    Rml::Shutdown();

    m_renderInterface.reset();

    if (m_gl3Initialized)
    {
        RmlGL3::Shutdown();
        m_gl3Initialized = false;
    }

    m_systemInterface.reset();
    m_actionHandler.reset();
    m_window = nullptr;
    m_previousAppMode.reset();
    m_initialized = false;
}

void GameUIManager::Update(double)
{
    if (!m_initialized || !m_context)
    {
        return;
    }

    EnsureContextDimensions();
    m_context->Update();
}

void GameUIManager::Render()
{
    if (!m_initialized || !m_context || !m_renderInterface)
    {
        return;
    }

    m_renderInterface->BeginFrame();
    m_context->Render();
    m_renderInterface->EndFrame();
}

void GameUIManager::OnViewportChanged(int width, int height)
{
    if (!m_initialized)
    {
        return;
    }

    m_viewportWidth = width;
    m_viewportHeight = height;

    if (m_renderInterface)
    {
        m_renderInterface->SetViewport(width, height);
    }

    EnsureContextDimensions();
}

void GameUIManager::OnContentScaleChanged(float scale)
{
    if (!m_initialized)
    {
        return;
    }

    m_contentScale = scale;
    EnsureContextDimensions();
}

bool GameUIManager::OnKeyEvent(int key, int, int action, int mods)
{
    if (!m_initialized || !m_context)
    {
        return false;
    }

    m_activeModifiers = mods;
    const bool propagate = RmlGLFW::ProcessKeyCallback(m_context, key, action, mods);
    return !propagate;
}

bool GameUIManager::OnCharacterEvent(unsigned int codepoint)
{
    if (!m_initialized || !m_context)
    {
        return false;
    }

    const bool propagate = RmlGLFW::ProcessCharCallback(m_context, codepoint);
    return !propagate;
}

bool GameUIManager::OnMouseButtonEvent(int button, int action, int mods)
{
    if (!m_initialized || !m_context)
    {
        return false;
    }

    m_activeModifiers = mods;
    const bool propagate = RmlGLFW::ProcessMouseButtonCallback(m_context, button, action, mods);
    return !propagate;
}

bool GameUIManager::OnCursorPosEvent(double xpos, double ypos)
{
    if (!m_initialized || !m_context)
    {
        return false;
    }

    const bool propagate = RmlGLFW::ProcessCursorPosCallback(m_context, m_window, xpos, ypos, m_activeModifiers);
    return !propagate;
}

bool GameUIManager::OnScrollEvent(double xoffset, double yoffset)
{
    if (!m_initialized || !m_context)
    {
        return false;
    }

    (void)xoffset;

    const bool propagate = RmlGLFW::ProcessScrollCallback(m_context, yoffset, m_activeModifiers);
    return !propagate;
}

bool GameUIManager::OnCursorEnterEvent(int entered)
{
    if (!m_initialized || !m_context)
    {
        return false;
    }

    const bool propagate = RmlGLFW::ProcessCursorEnterCallback(m_context, entered);
    return !propagate;
}

bool GameUIManager::HandleEscapePressed()
{
    if (!m_initialized)
    {
        return false;
    }

    if (m_state == GameUIState::Gameplay)
    {
        RequestState(GameUIState::MainMenu);
        return true;
    }

    if (m_state == GameUIState::MainMenu)
    {
        RequestState(GameUIState::Gameplay);
        return true;
    }

    return false;
}

void GameUIManager::ToggleMainMenu()
{
    if (m_state == GameUIState::MainMenu)
    {
        RequestState(GameUIState::Gameplay);
    }
    else
    {
        RequestState(GameUIState::MainMenu);
    }
}

void GameUIManager::RequestState(GameUIState state)
{
    if (state == m_state && state == GameUIState::MainMenu)
    {
        if (m_mainMenu)
        {
            m_mainMenu->Show();
        }
        return;
    }

    if (state == m_state)
    {
        return;
    }

    ApplyState(state);
}

GameUIState GameUIManager::GetState() const
{
    return m_state;
}

bool GameUIManager::IsMainMenuVisible() const
{
    return m_state == GameUIState::MainMenu && m_mainMenu && m_mainMenu->IsVisible();
}

void GameUIManager::SetActionHandler(std::shared_ptr<GameUIActionHandler> handler)
{
    if (handler)
    {
        m_actionHandler = std::move(handler);
    }
    else
    {
        m_actionHandler = std::make_shared<DefaultGameUIActionHandler>(m_window);
    }
}

void GameUIManager::CharCallback(GLFWwindow*, unsigned int codepoint)
{
    GameUIManager::Get().OnCharacterEvent(codepoint);
}

void GameUIManager::ScrollCallback(GLFWwindow*, double xoffset, double yoffset)
{
    (void)xoffset;
    GameUIManager::Get().OnScrollEvent(xoffset, yoffset);
}

void GameUIManager::CursorEnterCallback(GLFWwindow*, int entered)
{
    GameUIManager::Get().OnCursorEnterEvent(entered);
}

void GameUIManager::FramebufferSizeCallback(GLFWwindow*, int width, int height)
{
    auto& manager = GameUIManager::Get();
    manager.OnViewportChanged(width, height);
    if (manager.m_context)
    {
        RmlGLFW::ProcessFramebufferSizeCallback(manager.m_context, width, height);
    }
}

void GameUIManager::WindowContentScaleCallback(GLFWwindow*, float xscale, float yscale)
{
    (void)yscale;
    auto& manager = GameUIManager::Get();
    manager.OnContentScaleChanged(xscale);
    if (manager.m_context)
    {
        RmlGLFW::ProcessContentScaleCallback(manager.m_context, xscale);
    }
}

void GameUIManager::ApplyState(GameUIState state)
{
    m_state = state;

    if (!m_context)
    {
        return;
    }

    auto& inputHandler = InputHandler::Get();

    if (state == GameUIState::MainMenu)
    {
        if (!m_previousAppMode.has_value())
        {
            AppMode currentMode = inputHandler.getCurrentMode();
            if (currentMode != AppMode::GUI)
            {
                m_previousAppMode = currentMode;
            }
        }

        inputHandler.setAppMode(AppMode::GUI);

        if (m_mainMenu)
        {
            m_mainMenu->Show();
        }

        if (m_window)
        {
            glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }
    else
    {
        AppMode targetMode = m_previousAppMode.value_or(AppMode::CharacterFollow);
        m_previousAppMode.reset();

        inputHandler.setAppMode(targetMode);

        if (m_mainMenu)
        {
            m_mainMenu->Hide();
        }

        if (m_window)
        {
            glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
    }
}

void GameUIManager::BindDefaultFonts()
{
    namespace fs = std::filesystem;

    if (!fs::exists(kFontRegular))
    {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Font file missing: %s", kFontRegular);
    }

    if (Rml::LoadFontFace(kFontRegular))
    {
        Rml::LoadFontFace(kFontRegular, true);
        Rml::Log::Message(Rml::Log::LT_INFO, "Loaded font face '%s' as default.", kFontRegular);
    }
    else
    {
        Rml::Log::Message(Rml::Log::LT_WARNING, "Failed to load font face '%s'.", kFontRegular);
    }
    if (!Rml::LoadFontFace(kFontBold))
    {
        Rml::Log::Message(Rml::Log::LT_WARNING, "Failed to load font face '%s'.", kFontBold);
    }
    else
    {
        Rml::Log::Message(Rml::Log::LT_INFO, "Loaded font face '%s'.", kFontBold);
    }

}

void GameUIManager::EnsureContextDimensions()
{
    if (!m_context)
    {
        return;
    }

    const Rml::Vector2i desired_dimensions(m_viewportWidth, m_viewportHeight);
    if (m_context->GetDimensions() != desired_dimensions)
    {
        m_context->SetDimensions(desired_dimensions);
    }
    m_context->SetDensityIndependentPixelRatio(m_contentScale);
}

void GameUIManager::HandleContinueRequest()
{
    if (m_actionHandler)
    {
        m_actionHandler->OnContinueRequested();
    }
    RequestState(GameUIState::Gameplay);
}

void GameUIManager::HandleNewGameRequest()
{
    if (m_actionHandler)
    {
        m_actionHandler->OnNewGameRequested();
    }
}

void GameUIManager::HandleLoadGameRequest()
{
    if (m_actionHandler)
    {
        m_actionHandler->OnLoadGameRequested();
    }
}

void GameUIManager::HandleSettingsRequest()
{
    if (m_actionHandler)
    {
        m_actionHandler->OnSettingsRequested();
    }
}

void GameUIManager::HandleQuitRequest()
{
    if (m_actionHandler)
    {
        m_actionHandler->OnQuitRequested();
    }
}
