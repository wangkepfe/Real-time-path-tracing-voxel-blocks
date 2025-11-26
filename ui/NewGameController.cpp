#include "ui/NewGameController.h"

#include "ui/GameUIManager.h"

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>

namespace
{
    constexpr const char *kCreateButtonId = "new-game-create";
    constexpr const char *kCancelButtonId = "new-game-cancel";
    constexpr const char *kNameInputId = "new-game-name";
    constexpr const char *kErrorLabelId = "new-game-error";
}

NewGameController::NewGameController(Rml::Context &context, GameUIManager &manager)
    : m_context(context)
    , m_manager(manager)
{
}

NewGameController::~NewGameController()
{
    if (m_document)
    {
        UnregisterTrigger(kCreateButtonId);
        UnregisterTrigger(kCancelButtonId);
        UnregisterTrigger(kNameInputId, Rml::EventId::Change);
        m_document->Close();
        m_document = nullptr;
        m_nameInput = nullptr;
        m_errorLabel = nullptr;
    }
}

bool NewGameController::LoadDocument(const std::string &path)
{
    if (m_document)
    {
        UnregisterTrigger(kCreateButtonId);
        UnregisterTrigger(kCancelButtonId);
        UnregisterTrigger(kNameInputId, Rml::EventId::Change);
        m_document->Close();
        m_document = nullptr;
    }

    m_document = m_context.LoadDocument(path);
    if (!m_document)
    {
        return false;
    }

    m_nameInput = dynamic_cast<Rml::ElementFormControlInput *>(m_document->GetElementById(kNameInputId));
    m_errorLabel = m_document->GetElementById(kErrorLabelId);

    BindInteractions();
    return true;
}

void NewGameController::Show()
{
    if (m_document)
    {
        m_document->Show();
        m_document->PullToFront();
    }
}

void NewGameController::Hide()
{
    if (m_document)
    {
        m_document->Hide();
    }
}

void NewGameController::SetDefaultName(const std::string &name)
{
    if (m_nameInput)
    {
        m_nameInput->SetValue(name);
    }
    ClearError();
}

void NewGameController::FocusNameField()
{
    if (m_nameInput)
    {
        m_nameInput->Focus();
    }
}

void NewGameController::ClearError()
{
    if (m_errorLabel)
    {
        m_errorLabel->SetInnerRML("");
    }
}

void NewGameController::ShowError(const std::string &message)
{
    if (m_errorLabel)
    {
        m_errorLabel->SetInnerRML(message.c_str());
    }
}

void NewGameController::ProcessEvent(Rml::Event &event)
{
    const Rml::Element *target = event.GetTargetElement();
    if (!target)
    {
        return;
    }

    const Rml::String &elementId = target->GetId();

    if (event.GetId() == Rml::EventId::Click)
    {
        if (elementId == kCreateButtonId)
        {
            m_manager.OnNewGameConfirmed(GetEnteredName());
        }
        else if (elementId == kCancelButtonId)
        {
            m_manager.OnNewGameCancelled();
        }
    }
    else if (event.GetId() == Rml::EventId::Change && elementId == kNameInputId)
    {
        ClearError();
    }
}

void NewGameController::BindInteractions()
{
    RegisterTrigger(kCreateButtonId);
    RegisterTrigger(kCancelButtonId);
    RegisterTrigger(kNameInputId, Rml::EventId::Change);
}

void NewGameController::RegisterTrigger(const char *elementId, Rml::EventId eventId)
{
    if (!m_document)
    {
        return;
    }

    if (Rml::Element *element = m_document->GetElementById(elementId))
    {
        element->AddEventListener(eventId, this);
    }
}

void NewGameController::UnregisterTrigger(const char *elementId, Rml::EventId eventId)
{
    if (!m_document)
    {
        return;
    }

    if (Rml::Element *element = m_document->GetElementById(elementId))
    {
        element->RemoveEventListener(eventId, this);
    }
}

std::string NewGameController::GetEnteredName() const
{
    if (!m_nameInput)
    {
        return {};
    }
    const Rml::String value = m_nameInput->GetValue();
    return std::string(value.c_str(), value.size());
}
