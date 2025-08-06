#include "CameraController.h"
#include "GlobalSettings.h"

void CameraController::updateFromGlobalSettings()
{
    auto &globalCamera = GlobalSettings::GetCameraMovementParams();
    m_moveSpeed = globalCamera.moveSpeed;
    m_cursorMoveSpeed = globalCamera.cursorMoveSpeed;
}