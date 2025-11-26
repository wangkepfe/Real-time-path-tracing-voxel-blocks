#include "ui/GameUIManager.h"

#include "ui/GameUIActionHandler.h"
#include "ui/LoadGameController.h"
#include "ui/MainMenuController.h"
#include "ui/NewGameController.h"

#include "renderer/core/InputHandler.h"
#include "renderer/core/RenderCamera.h"
#include "core/WorldSceneManager.h"

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
#include <algorithm>

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

    m_newGame = std::make_unique<NewGameController>(*m_context, *this);
    if (!m_newGame->LoadDocument("data/ui/new_game.rml"))
    {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to load new game document.");
        m_newGame.reset();
    }
    else
    {
        m_newGame->Hide();
    }

    m_loadGame = std::make_unique<LoadGameController>(*m_context, *this);
    if (!m_loadGame->LoadDocument("data/ui/load_game.rml"))
    {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Failed to load load-game document.");
        m_loadGame.reset();
    }
    else
    {
        m_loadGame->Hide();
    }

    SetActionHandler(std::make_shared<DefaultGameUIActionHandler>(m_window));

    glfwSetInputMode(window, GLFW_LOCK_KEY_MODS, GLFW_TRUE);

    InitializeWorldState();
    RefreshWorldList();
    UpdateContinueButton();

    m_state = GameUIState::MainMenu;
    ApplyState(m_state);

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
    m_newGame.reset();
    m_loadGame.reset();

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

    if (m_state == GameUIState::NewGame || m_state == GameUIState::LoadGame)
    {
        RequestState(GameUIState::MainMenu);
        return true;
    }

    return false;
}

void GameUIManager::ToggleMainMenu()
{
    if (m_state == GameUIState::Gameplay)
    {
        RequestState(GameUIState::MainMenu);
        return;
    }

    RequestState(GameUIState::Gameplay);
}

void GameUIManager::RequestState(GameUIState state)
{
    if (state == m_state)
    {
        return;
    }

    m_state = state;
    ApplyState(state);
}

GameUIState GameUIManager::GetState() const
{
    return m_state;
}

bool GameUIManager::IsMainMenuVisible() const
{
    return m_state != GameUIState::Gameplay;
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
    if (!m_context)
    {
        return;
    }

    auto &inputHandler = InputHandler::Get();
    const bool menuState = (state == GameUIState::MainMenu || state == GameUIState::NewGame || state == GameUIState::LoadGame);

    if (menuState)
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
            if (state == GameUIState::MainMenu)
            {
                m_mainMenu->Show();
            }
            else
            {
                m_mainMenu->Hide();
            }
        }

        if (m_newGame)
        {
            if (state == GameUIState::NewGame)
            {
                m_newGame->Show();
            }
            else
            {
                m_newGame->Hide();
            }
        }

        if (m_loadGame)
        {
            if (state == GameUIState::LoadGame)
            {
                m_loadGame->Show();
            }
            else
            {
                m_loadGame->Hide();
            }
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

        if (m_newGame)
        {
            m_newGame->Hide();
        }

        if (m_loadGame)
        {
            m_loadGame->Hide();
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
    if (!m_hasLoadedWorld)
    {
        return;
    }

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

    if (m_newGame)
    {
        m_newGame->SetDefaultName(SuggestWorldName());
        m_newGame->FocusNameField();
        m_newGame->ClearError();
    }

    RequestState(GameUIState::NewGame);
}

void GameUIManager::HandleLoadGameRequest()
{
    if (m_actionHandler)
    {
        m_actionHandler->OnLoadGameRequested();
    }

    RefreshWorldList();
    if (m_loadGame)
    {
        m_loadGame->ClearError();
        const std::string selection = !m_pendingLoadSelection.empty() ? m_pendingLoadSelection : m_currentWorldName;
        if (!selection.empty())
        {
            m_loadGame->SetSelectedWorld(selection);
        }
    }

    RequestState(GameUIState::LoadGame);
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
    SaveActiveWorld();

    if (m_actionHandler)
    {
        m_actionHandler->OnQuitRequested();
    }
}

void GameUIManager::OnNewGameConfirmed(const std::string &worldName)
{
    if (!m_newGame)
    {
        return;
    }

    std::string normalized;
    std::string error;
    if (!WorldSceneManager::ValidateWorldName(worldName, normalized, error))
    {
        m_newGame->ShowError(error);
        return;
    }

    if (WorldSceneManager::WorldExists(normalized))
    {
        m_newGame->ShowError("A world with that name already exists.");
        return;
    }

    if (!SaveActiveWorld())
    {
        m_newGame->ShowError("Failed to save the current world.");
        return;
    }

    auto &inputHandler = InputHandler::Get();
    if (!WorldSceneManager::CreateWorld(normalized, inputHandler))
    {
        m_newGame->ShowError("Unable to create the new world.");
        return;
    }

    m_currentWorldName = normalized;
    m_hasLoadedWorld = true;
    m_pendingLoadSelection.clear();
    RefreshWorldList();
    UpdateContinueButton();
    RequestState(GameUIState::Gameplay);
}

void GameUIManager::OnNewGameCancelled()
{
    RequestState(GameUIState::MainMenu);
}

void GameUIManager::OnLoadGameSelectionChanged(const std::string &worldName)
{
    m_pendingLoadSelection = worldName;
}

void GameUIManager::OnLoadGameConfirmed(const std::string &worldName)
{
    if (!m_loadGame)
    {
        return;
    }

    if (worldName.empty())
    {
        m_loadGame->ShowError("Select a world to load.");
        return;
    }

    if (!WorldSceneManager::WorldExists(worldName))
    {
        m_loadGame->ShowError("World not found.");
        return;
    }

    if (!SaveActiveWorld())
    {
        m_loadGame->ShowError("Failed to save the current world.");
        return;
    }

    if (!LoadWorldInternal(worldName))
    {
        m_loadGame->ShowError("Failed to load the selected world.");
        return;
    }

    RefreshWorldList();
    RequestState(GameUIState::Gameplay);
}

void GameUIManager::OnLoadGameCancelled()
{
    m_pendingLoadSelection.clear();
    RequestState(GameUIState::MainMenu);
}

void GameUIManager::SaveActiveWorldToDisk()
{
    SaveActiveWorld();
}

void GameUIManager::InitializeWorldState()
{
    bool loadedExisting = false;

    if (auto lastWorld = WorldSceneManager::GetLastPlayedWorld())
    {
        loadedExisting = LoadWorldInternal(*lastWorld);
    }

    if (!loadedExisting)
    {
        auto worlds = WorldSceneManager::ListWorlds();
        if (!worlds.empty())
        {
            loadedExisting = LoadWorldInternal(worlds.front());
        }
    }

    if (!loadedExisting)
    {
        auto &inputHandler = InputHandler::Get();
        const std::string newWorld = WorldSceneManager::GenerateDefaultWorldName();
        if (WorldSceneManager::CreateWorld(newWorld, inputHandler))
        {
            LoadWorldInternal(newWorld);
        }
    }
}

void GameUIManager::RefreshWorldList()
{
    m_availableWorlds = WorldSceneManager::ListWorlds();
    if (m_loadGame)
    {
        m_loadGame->SetWorldList(m_availableWorlds);
        if (!m_currentWorldName.empty())
        {
            m_loadGame->SetSelectedWorld(m_currentWorldName);
        }
    }
}

void GameUIManager::UpdateContinueButton()
{
    if (m_mainMenu)
    {
        m_mainMenu->SetContinueEnabled(m_hasLoadedWorld);
    }
}

bool GameUIManager::SaveActiveWorld()
{
    if (m_currentWorldName.empty())
    {
        return true;
    }

    auto &camera = RenderCamera::Get().camera;
    return WorldSceneManager::SaveWorld(m_currentWorldName, camera, InputHandler::Get());
}

bool GameUIManager::LoadWorldInternal(const std::string &worldName)
{
    if (worldName.empty())
    {
        return false;
    }

    auto &inputHandler = InputHandler::Get();
    if (!WorldSceneManager::LoadWorld(worldName, inputHandler))
    {
        return false;
    }

    m_currentWorldName = worldName;
    m_hasLoadedWorld = true;
    m_pendingLoadSelection.clear();
    UpdateContinueButton();
    return true;
}

std::string GameUIManager::SuggestWorldName() const
{
    return WorldSceneManager::GenerateDefaultWorldName();
}
