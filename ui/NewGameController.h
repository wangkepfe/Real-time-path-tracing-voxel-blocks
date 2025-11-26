#pragma once

#include <RmlUi/Core/EventListener.h>

#include <string>

namespace Rml
{
    class Context;
    class ElementDocument;
    class Element;
    class Event;
    class ElementFormControlInput;
}

class GameUIManager;

class NewGameController : public Rml::EventListener
{
public:
    NewGameController(Rml::Context &context, GameUIManager &manager);
    ~NewGameController() override;

    bool LoadDocument(const std::string &path);

    void Show();
    void Hide();

    void SetDefaultName(const std::string &name);
    void FocusNameField();
    void ClearError();
    void ShowError(const std::string &message);

    void ProcessEvent(Rml::Event &event) override;

private:
    void BindInteractions();
    void RegisterTrigger(const char *elementId, Rml::EventId eventId = Rml::EventId::Click);
    void UnregisterTrigger(const char *elementId, Rml::EventId eventId = Rml::EventId::Click);
    std::string GetEnteredName() const;

    Rml::Context &m_context;
    GameUIManager &m_manager;
    Rml::ElementDocument *m_document = nullptr;
    Rml::ElementFormControlInput *m_nameInput = nullptr;
    Rml::Element *m_errorLabel = nullptr;
};

