#include "FreeCameraController.h"
#include "../shaders/Camera.h"
#include "../shaders/LinearMath.h"

void FreeCameraController::updateCamera(Camera& camera, float deltaTime)
{
    // Update movement speeds from global settings
    updateFromGlobalSettings();
    
    // Calculate movement based on input
    if (m_moveW || m_moveS || m_moveA || m_moveD || m_moveC || m_moveX)
    {
        Float3 movingDir{0};
        
        // Use actual camera direction for forward/backward movement
        Float3 forwardDir = camera.dir; // Use full 3D camera direction
        
        // For strafing, use horizontal-only direction to maintain level movement
        Float3 horizontalDir = camera.dir;
        horizontalDir.y = 0.0f;
        horizontalDir = normalize(horizontalDir);
        Float3 strafeDir = cross(horizontalDir, Float3(0, 1, 0)).normalize();

        if (m_moveW) movingDir += forwardDir;        // Forward in actual look direction
        if (m_moveS) movingDir -= forwardDir;        // Backward in actual look direction
        if (m_moveA) movingDir -= strafeDir;         // Left
        if (m_moveD) movingDir += strafeDir;         // Right
        if (m_moveC) movingDir += Float3(0, 1, 0);   // Up
        if (m_moveX) movingDir -= Float3(0, 1, 0);   // Down

        // Apply speed modifiers
        float effectiveSpeed = m_moveSpeed;
        if (m_slowMode) effectiveSpeed *= 0.5f;      // Ctrl for 0.5x speed
        else if (m_fastMode) effectiveSpeed *= 2.0f; // Shift for 2x speed
        
        // Apply movement directly to position
        Float3 movement = movingDir * deltaTime * effectiveSpeed;
        camera.pos = camera.pos + movement;
    }
    
    // Ensure posDelta stays zero (we manage position directly)
    camera.posDelta = Float3(0.0f, 0.0f, 0.0f);
    
    // Update camera direction and matrices
    camera.updateMatrices();
}

void FreeCameraController::handleMouseMovement(Camera& camera, float deltax, float deltay)
{
    // Update movement speeds from global settings
    updateFromGlobalSettings();
    
    camera.yaw -= deltax * m_cursorMoveSpeed;
    camera.pitch -= deltay * m_cursorMoveSpeed;
    camera.pitch = clampf(camera.pitch, -PI_OVER_2 + 0.1f, PI_OVER_2 - 0.1f);
}

void FreeCameraController::setMovementInput(bool moveW, bool moveS, bool moveA, bool moveD, bool moveC, bool moveX)
{
    m_moveW = moveW;
    m_moveS = moveS;
    m_moveA = moveA;
    m_moveD = moveD;
    m_moveC = moveC;
    m_moveX = moveX;
}

void FreeCameraController::onActivate(Camera& camera)
{
    // Ensure camera is in a clean state
    camera.posDelta = Float3(0.0f, 0.0f, 0.0f);
}