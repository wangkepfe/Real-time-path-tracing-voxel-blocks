#include "ui/LoadGameController.h"

#include "ui/GameUIManager.h"

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Variant.h>

#include <utility>

namespace
{
    constexpr const char *kListContainerId = "load-game-list";
    constexpr const char *kLoadButtonId = "load-game-load";
    constexpr const char *kCancelButtonId = "load-game-cancel";
    constexpr const char *kErrorLabelId = "load-game-error";
    constexpr const char *kItemClass = "world-item";
    constexpr const char *kSelectedClass = "selected";
    constexpr const char *kIndexAttribute = "data-index";

    Rml::Element *ResolveItemElement(Rml::Element *element)
    {
        while (element && !element->HasAttribute(kIndexAttribute))
        {
            element = element->GetParentNode();
        }
        return element;
    }
}

LoadGameController::LoadGameController(Rml::Context &context, GameUIManager &manager)
    : m_context(context)
    , m_manager(manager)
{
}

LoadGameController::~LoadGameController()
{
    if (m_document)
    {
        UnregisterTrigger(kListContainerId, Rml::EventId::Click);
        UnregisterTrigger(kLoadButtonId, Rml::EventId::Click);
        UnregisterTrigger(kCancelButtonId, Rml::EventId::Click);
        m_document->Close();
        m_document = nullptr;
        m_listContainer = nullptr;
        m_errorLabel = nullptr;
        m_worldEntries.clear();
    }
}

bool LoadGameController::LoadDocument(const std::string &path)
{
    if (m_document)
    {
        UnregisterTrigger(kListContainerId, Rml::EventId::Click);
        UnregisterTrigger(kLoadButtonId, Rml::EventId::Click);
        UnregisterTrigger(kCancelButtonId, Rml::EventId::Click);
        m_document->Close();
        m_document = nullptr;
    }

    m_document = m_context.LoadDocument(path);
    if (!m_document)
    {
        return false;
    }

    m_listContainer = m_document->GetElementById(kListContainerId);
    m_errorLabel = m_document->GetElementById(kErrorLabelId);
    BindInteractions();
    return true;
}

void LoadGameController::Show()
{
    if (m_document)
    {
        m_document->Show();
        m_document->PullToFront();
    }
}

void LoadGameController::Hide()
{
    if (m_document)
    {
        m_document->Hide();
    }
}

void LoadGameController::SetWorldList(const std::vector<std::string> &worlds)
{
    m_worldEntries = worlds;
    m_selectedWorld.clear();

    if (!m_listContainer)
    {
        return;
    }

    while (Rml::Element *child = m_listContainer->GetFirstChild())
    {
        m_listContainer->RemoveChild(child);
    }

    for (size_t index = 0; index < worlds.size(); ++index)
    {
        Rml::ElementPtr button = m_document->CreateElement("button");
        if (!button)
        {
            continue;
        }

        Rml::Element *buttonElement = button.get();
        buttonElement->SetInnerRML(worlds[index].c_str());
        buttonElement->SetAttribute(kIndexAttribute, static_cast<int>(index));
        buttonElement->SetClass(kItemClass, true);
        m_listContainer->AppendChild(std::move(button));
    }

    if (!worlds.empty())
    {
        m_selectedWorld = worlds.front();
        UpdateSelectionFromControl();
        NotifySelectionChanged();
        ClearError();
    }
    else
    {
        UpdateSelectionFromControl();
        if (m_errorLabel)
        {
            m_errorLabel->SetInnerRML("No saved worlds found.");
        }
    }
}

void LoadGameController::SetSelectedWorld(const std::string &worldName)
{
    if (worldName.empty())
    {
        m_selectedWorld.clear();
        UpdateSelectionFromControl();
        NotifySelectionChanged();
        return;
    }

    for (size_t index = 0; index < m_worldEntries.size(); ++index)
    {
        if (m_worldEntries[index] == worldName)
        {
            m_selectedWorld = worldName;
            UpdateSelectionFromControl();
            NotifySelectionChanged();
            ClearError();
            return;
        }
    }
}

void LoadGameController::ClearError()
{
    if (m_errorLabel)
    {
        m_errorLabel->SetInnerRML("");
    }
}

void LoadGameController::ShowError(const std::string &message)
{
    if (m_errorLabel)
    {
        m_errorLabel->SetInnerRML(message.c_str());
    }
}

void LoadGameController::ProcessEvent(Rml::Event &event)
{
    Rml::Element *target = const_cast<Rml::Element *>(event.GetTargetElement());
    if (!target)
    {
        return;
    }

    const Rml::String &elementId = target->GetId();

    if (event.GetId() == Rml::EventId::Click)
    {
        if (elementId == kLoadButtonId)
        {
            if (!m_selectedWorld.empty())
            {
                m_manager.OnLoadGameConfirmed(m_selectedWorld);
            }
            else
            {
                ShowError("Select a world to load.");
            }
            return;
        }

        if (elementId == kCancelButtonId)
        {
            m_manager.OnLoadGameCancelled();
            return;
        }

        if (target == m_listContainer || target->GetParentNode() == nullptr)
        {
            return;
        }

        if (Rml::Element *itemElement = ResolveItemElement(target))
        {
            if (const Rml::Variant *indexVariant = itemElement->GetAttribute(kIndexAttribute))
            {
                int index = indexVariant->Get<int>();
                if (index >= 0 && static_cast<size_t>(index) < m_worldEntries.size())
                {
                    m_selectedWorld = m_worldEntries[static_cast<size_t>(index)];
                    UpdateSelectionFromControl();
                    NotifySelectionChanged();
                    ClearError();
                }
            }
        }
    }
}

void LoadGameController::BindInteractions()
{
    RegisterTrigger(kListContainerId, Rml::EventId::Click);
    RegisterTrigger(kLoadButtonId, Rml::EventId::Click);
    RegisterTrigger(kCancelButtonId, Rml::EventId::Click);
}

void LoadGameController::RegisterTrigger(const char *elementId, Rml::EventId eventId)
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

void LoadGameController::UnregisterTrigger(const char *elementId, Rml::EventId eventId)
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

void LoadGameController::UpdateSelectionFromControl()
{
    if (!m_listContainer)
    {
        return;
    }

    for (Rml::Element *child = m_listContainer->GetFirstChild(); child != nullptr; child = child->GetNextSibling())
    {
        bool isSelected = false;
        if (const Rml::Variant *indexVariant = child->GetAttribute(kIndexAttribute))
        {
            int index = indexVariant->Get<int>();
            if (index >= 0 && static_cast<size_t>(index) < m_worldEntries.size())
            {
                isSelected = (m_worldEntries[static_cast<size_t>(index)] == m_selectedWorld);
            }
        }
        child->SetClass(kSelectedClass, isSelected);
    }
}

void LoadGameController::NotifySelectionChanged()
{
    m_manager.OnLoadGameSelectionChanged(m_selectedWorld);
}
