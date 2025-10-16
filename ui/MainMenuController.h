#pragma once

#include <RmlUi/Core/EventListener.h>
#include <string>

namespace Rml
{
    class Context;
    class Element;
    class ElementDocument;
}

class GameUIManager;

class MainMenuController : public Rml::EventListener
{
public:
    MainMenuController(Rml::Context& context, GameUIManager& manager);
    ~MainMenuController() override;

    bool LoadDocument(const std::string& path);

    void Show();
    void Hide();
    bool IsVisible() const;

    void ProcessEvent(Rml::Event& event) override;

private:
    void BindInteractions();
    void RegisterTrigger(const char* element_id);
    void UnregisterTrigger(const char* element_id);

    Rml::Context& m_context;
    GameUIManager& m_manager;
    Rml::ElementDocument* m_document = nullptr;
};
