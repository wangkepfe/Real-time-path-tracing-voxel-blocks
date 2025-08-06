#pragma once

#include "CameraController.h"
#include "../shaders/Camera.h"

/**
 * Free camera controller for unrestricted 3D movement.
 * Provides WASD + mouse controls for flying around the scene.
 */
class FreeCameraController : public CameraController
{
public:
    void updateCamera(Camera& camera, float deltaTime) override;
    void handleMouseMovement(Camera& camera, float deltax, float deltay) override;
    void setMovementInput(bool moveW, bool moveS, bool moveA, bool moveD, bool moveC, bool moveX) override;
    void setSpeedModifiers(bool slowMode, bool fastMode) { m_slowMode = slowMode; m_fastMode = fastMode; }

    void onActivate(Camera& camera) override;
    const char* getName() const override { return "FreeCamera"; }

private:
    bool m_slowMode = false;  // Ctrl modifier for 0.5x speed
    bool m_fastMode = false;  // Shift modifier for 2x speed
};