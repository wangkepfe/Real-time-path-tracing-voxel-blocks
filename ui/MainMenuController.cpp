#include "ui/MainMenuController.h"

#include "ui/GameUIManager.h"

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Event.h>

#include <array>

namespace
{
    constexpr std::array<const char*, 5> kMenuElementIds = {
        "menu-continue",
        "menu-new",
        "menu-load",
        "menu-settings",
        "menu-quit"};
}

MainMenuController::MainMenuController(Rml::Context& context, GameUIManager& manager)
    : m_context(context)
    , m_manager(manager)
{
}

MainMenuController::~MainMenuController()
{
    if (m_document)
    {
        for (const char* id : kMenuElementIds)
        {
            UnregisterTrigger(id);
        }
        m_document->Close();
        m_document = nullptr;
    }
    m_continueButton = nullptr;
}

bool MainMenuController::LoadDocument(const std::string& path)
{
    if (m_document)
    {
        m_document->Close();
        m_document = nullptr;
    }

    m_document = m_context.LoadDocument(path);
    if (!m_document)
    {
        return false;
    }

    m_continueButton = m_document->GetElementById("menu-continue");
    BindInteractions();
    return true;
}

void MainMenuController::Show()
{
    if (m_document)
    {
        m_document->Show();
        m_document->PullToFront();
    }
}

void MainMenuController::Hide()
{
    if (m_document)
    {
        m_document->Hide();
    }
}

bool MainMenuController::IsVisible() const
{
    return m_document && m_document->IsVisible();
}

void MainMenuController::SetContinueEnabled(bool enabled)
{
    if (!m_continueButton)
    {
        return;
    }

    if (enabled)
    {
        m_continueButton->RemoveAttribute("disabled");
    }
    else
    {
        m_continueButton->SetAttribute("disabled", true);
    }
}

void MainMenuController::ProcessEvent(Rml::Event& event)
{
    const Rml::Element* target = event.GetTargetElement();
    if (!target)
    {
        return;
    }

    const Rml::String& element_id = target->GetId();
    if (element_id == "menu-continue")
    {
        m_manager.HandleContinueRequest();
    }
    else if (element_id == "menu-new")
    {
        m_manager.HandleNewGameRequest();
    }
    else if (element_id == "menu-load")
    {
        m_manager.HandleLoadGameRequest();
    }
    else if (element_id == "menu-settings")
    {
        m_manager.HandleSettingsRequest();
    }
    else if (element_id == "menu-quit")
    {
        m_manager.HandleQuitRequest();
    }
}

void MainMenuController::BindInteractions()
{
    for (const char* id : kMenuElementIds)
    {
        RegisterTrigger(id);
    }
}

void MainMenuController::RegisterTrigger(const char* element_id)
{
    if (!m_document)
    {
        return;
    }

    if (Rml::Element* element = m_document->GetElementById(element_id))
    {
        element->AddEventListener(Rml::EventId::Click, this);
    }
}

void MainMenuController::UnregisterTrigger(const char* element_id)
{
    if (!m_document)
    {
        return;
    }

    if (Rml::Element* element = m_document->GetElementById(element_id))
    {
        element->RemoveEventListener(Rml::EventId::Click, this);
    }
}
