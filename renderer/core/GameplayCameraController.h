#pragma once

#include "CameraController.h"
#include "../shaders/Camera.h"

/**
 * Gameplay camera controller with terrain collision.
 * Provides WASD movement with physics (gravity, collision) for first-person gameplay.
 */
class GameplayCameraController : public CameraController
{
public:
    GameplayCameraController();
    
    void updateCamera(Camera& camera, float deltaTime) override;
    void handleMouseMovement(Camera& camera, float deltax, float deltay) override;
    void setMovementInput(bool moveW, bool moveS, bool moveA, bool moveD, bool moveC, bool moveX) override;
    void onActivate(Camera& camera) override;
    const char* getName() const override { return "Gameplay"; }

private:
    // Physics parameters
    float m_fallSpeed = 0.0f;
    float m_height = 1.5f;
};