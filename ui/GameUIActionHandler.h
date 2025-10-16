#pragma once

struct GLFWwindow;

class GameUIActionHandler
{
public:
    virtual ~GameUIActionHandler() = default;

    virtual void OnContinueRequested() {}
    virtual void OnNewGameRequested() {}
    virtual void OnLoadGameRequested() {}
    virtual void OnSettingsRequested() {}
    virtual void OnQuitRequested() {}
};

class DefaultGameUIActionHandler : public GameUIActionHandler
{
public:
    explicit DefaultGameUIActionHandler(GLFWwindow* window);
    void OnQuitRequested() override;

private:
    GLFWwindow* m_window = nullptr;
};
