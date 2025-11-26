#pragma once

#include <RmlUi/Core/EventListener.h>

#include <string>
#include <vector>

namespace Rml
{
    class Context;
    class ElementDocument;
    class Element;
    class Event;
}

class GameUIManager;

class LoadGameController : public Rml::EventListener
{
public:
    LoadGameController(Rml::Context &context, GameUIManager &manager);
    ~LoadGameController() override;

    bool LoadDocument(const std::string &path);

    void Show();
    void Hide();

    void SetWorldList(const std::vector<std::string> &worlds);
    void SetSelectedWorld(const std::string &worldName);
    void ClearError();
    void ShowError(const std::string &message);

    void ProcessEvent(Rml::Event &event) override;

private:
    void BindInteractions();
    void RegisterTrigger(const char *elementId, Rml::EventId eventId);
    void UnregisterTrigger(const char *elementId, Rml::EventId eventId);
    void UpdateSelectionFromControl();
    void NotifySelectionChanged();

    Rml::Context &m_context;
    GameUIManager &m_manager;
    Rml::ElementDocument *m_document = nullptr;
    Rml::Element *m_listContainer = nullptr;
    Rml::Element *m_errorLabel = nullptr;
    std::string m_selectedWorld;
    std::vector<std::string> m_worldEntries;
};
