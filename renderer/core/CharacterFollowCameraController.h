#pragma once

#include "CameraController.h"
#include "../shaders/Camera.h"

class Character;

/**
 * Character-following camera controller.
 * Provides smooth GTA5-style camera that follows the character with mouse orbit controls.
 */
class CharacterFollowCameraController : public CameraController
{
public:
    CharacterFollowCameraController();

    void updateCamera(Camera &camera, float deltaTime) override;
    void handleMouseMovement(Camera &camera, float deltax, float deltay) override;
    void setMovementInput(bool moveW, bool moveS, bool moveA, bool moveD, bool moveC, bool moveX) override;
    void setSpeedModifiers(bool slowMode, bool fastMode) override;
    void onActivate(Camera &camera) override;
    const char *getName() const override { return "CharacterFollow"; }

    // Character-specific methods
    void setCharacter(Character *character) { m_character = character; }
    Character *getCharacter() const { return m_character; }

private:
    void initializeCameraForCharacter(Camera &camera);
    void updateCameraFollowing(Camera &camera, float deltaTime);

    Character *m_character = nullptr;

    // Speed modifiers
    bool m_slowMode = false;
    bool m_fastMode = false;

    // Camera following parameters
    Float3 m_cameraOffset = Float3(0.0f, 2.5f, 5.0f); // Default behind and above
    Float3 m_cameraTarget = Float3(0.0f, 0.0f, 0.0f);
    float m_cameraFollowSpeed = 5.0f;
    bool m_cameraInitialized = false;
};