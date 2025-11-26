#pragma once

#include "ui/GameUIState.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

struct GLFWwindow;

namespace Rml
{
    class Context;
}

class RenderInterface_GL3;
class SystemInterface_GLFW;

class GameUIActionHandler;
class MainMenuController;
class NewGameController;
class LoadGameController;

enum class AppMode;

class GameUIManager
{
public:
    static GameUIManager& Get();

    bool Initialize(GLFWwindow* window, int width, int height);
    void Shutdown();

    void Update(double delta_seconds);
    void Render();
    void OnViewportChanged(int width, int height);
    void OnContentScaleChanged(float scale);

    bool OnKeyEvent(int key, int scancode, int action, int mods);
    bool OnCharacterEvent(unsigned int codepoint);
    bool OnMouseButtonEvent(int button, int action, int mods);
    bool OnCursorPosEvent(double xpos, double ypos);
    bool OnScrollEvent(double xoffset, double yoffset);
    bool OnCursorEnterEvent(int entered);

    bool HandleEscapePressed();
    void ToggleMainMenu();
    void RequestState(GameUIState state);

    GameUIState GetState() const;
    bool IsMainMenuVisible() const;
    bool IsInitialized() const { return m_initialized; }

    void SetActionHandler(std::shared_ptr<GameUIActionHandler> handler);

    void OnNewGameConfirmed(const std::string& worldName);
    void OnNewGameCancelled();
    void OnLoadGameSelectionChanged(const std::string& worldName);
    void OnLoadGameConfirmed(const std::string& worldName);
    void OnLoadGameCancelled();
    void SaveActiveWorldToDisk();

    static void CharCallback(GLFWwindow* window, unsigned int codepoint);
    static void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void CursorEnterCallback(GLFWwindow* window, int entered);
    static void FramebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void WindowContentScaleCallback(GLFWwindow* window, float xscale, float yscale);

private:
    GameUIManager() = default;
    ~GameUIManager() = default;

    GameUIManager(const GameUIManager&) = delete;
    GameUIManager& operator=(const GameUIManager&) = delete;

    void ApplyState(GameUIState state);
    void BindDefaultFonts();
    void EnsureContextDimensions();
    void InitializeWorldState();
    void RefreshWorldList();
    void UpdateContinueButton();
    bool SaveActiveWorld();
    bool LoadWorldInternal(const std::string& worldName);
    std::string SuggestWorldName() const;

    void HandleContinueRequest();
    void HandleNewGameRequest();
    void HandleLoadGameRequest();
    void HandleSettingsRequest();
    void HandleQuitRequest();

    bool m_initialized = false;
    GLFWwindow* m_window = nullptr;
    std::unique_ptr<SystemInterface_GLFW> m_systemInterface;
    std::unique_ptr<RenderInterface_GL3> m_renderInterface;
    Rml::Context* m_context = nullptr;
    std::unique_ptr<MainMenuController> m_mainMenu;
    std::unique_ptr<NewGameController> m_newGame;
    std::unique_ptr<LoadGameController> m_loadGame;
    std::shared_ptr<GameUIActionHandler> m_actionHandler;
    GameUIState m_state = GameUIState::MainMenu;
    std::optional<AppMode> m_previousAppMode;
    int m_viewportWidth = 0;
    int m_viewportHeight = 0;
    int m_activeModifiers = 0;
    float m_contentScale = 1.0f;
    bool m_gl3Initialized = false;
    std::vector<std::string> m_availableWorlds;
    std::string m_currentWorldName;
    std::string m_pendingLoadSelection;
    bool m_hasLoadedWorld = false;

    friend class MainMenuController;
};
