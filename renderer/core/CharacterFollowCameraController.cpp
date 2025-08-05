#include "CharacterFollowCameraController.h"
#include "../shaders/Camera.h"
#include "../shaders/LinearMath.h"
#include "Character.h"
#include "GlobalSettings.h"
#include <iostream>

CharacterFollowCameraController::CharacterFollowCameraController()
{
    // Movement speeds will be updated from global settings
}

void CharacterFollowCameraController::updateCamera(Camera& camera, float deltaTime)
{
    // Update movement speeds and camera follow parameters from global settings
    updateFromGlobalSettings();
    auto &globalCamera = GlobalSettings::GetCameraMovementParams();
    m_cameraFollowSpeed = globalCamera.followSpeed;
    m_cameraOffset.z = globalCamera.followDistance;
    m_cameraOffset.y = globalCamera.followHeight;
    
    if (!m_character)
    {
        std::cerr << "CharacterFollowCameraController: No character set!" << std::endl;
        return;
    }

    if (!m_cameraInitialized)
    {
        initializeCameraForCharacter(camera);
        return;
    }

    // Convert camera-relative input to world-relative movement for character
    Float3 cameraForward = camera.dir;
    cameraForward.y = 0.0f; // Remove vertical component
    cameraForward = normalize(cameraForward);
    
    Float3 cameraRight = cross(cameraForward, Float3(0.0f, 1.0f, 0.0f));
    cameraRight = normalize(cameraRight);
    
    // Calculate world-space movement direction based on camera orientation
    Float3 moveDirection = Float3(0.0f, 0.0f, 0.0f);
    bool hasInput = m_moveW || m_moveS || m_moveA || m_moveD;
    
    if (hasInput)
    {
        if (m_moveW) moveDirection = moveDirection + cameraForward;
        if (m_moveS) moveDirection = moveDirection - cameraForward;
        if (m_moveA) moveDirection = moveDirection - cameraRight;
        if (m_moveD) moveDirection = moveDirection + cameraRight;
        
    }
    else
    {
    }
    
    // Set the character movement direction in world space
    m_character->setMovementDirection(moveDirection);
    
    // Determine running mode: active when fast mode is on and there's movement input
    bool hasMovementInput = m_moveW || m_moveS || m_moveA || m_moveD;
    bool isRunning = m_fastMode && hasMovementInput;
    
    // Update character with input
    m_character->setMovementInput(m_moveW, m_moveS, m_moveA, m_moveD, m_moveC, isRunning);
    
    // Update character
    m_character->update(deltaTime);
    
    // Update camera to follow character
    updateCameraFollowing(camera, deltaTime);
}

void CharacterFollowCameraController::handleMouseMovement(Camera& camera, float deltax, float deltay)
{
    // Update movement speeds from global settings
    updateFromGlobalSettings();
    
    if (!m_character || !m_cameraInitialized)
        return;
        
    // Update camera yaw/pitch for orbiting around character
    camera.yaw -= deltax * m_cursorMoveSpeed;
    camera.pitch -= deltay * m_cursorMoveSpeed;
    camera.pitch = clampf(camera.pitch, -PI_OVER_2 + 0.3f, PI_OVER_2 - 0.3f);
}

void CharacterFollowCameraController::setMovementInput(bool moveW, bool moveS, bool moveA, bool moveD, bool moveC, bool moveX)
{
    m_moveW = moveW;
    m_moveS = moveS;
    m_moveA = moveA;
    m_moveD = moveD;
    m_moveC = moveC;
    // Note: moveX (down) is not used in character mode
    
}

void CharacterFollowCameraController::setSpeedModifiers(bool slowMode, bool fastMode)
{
    m_slowMode = slowMode;
    m_fastMode = fastMode;
}

void CharacterFollowCameraController::onActivate(Camera& camera)
{
    
    // Clear any movement input
    setMovementInput(false, false, false, false, false, false);
    
    // Initialize camera for character if we have one
    if (m_character)
    {
        initializeCameraForCharacter(camera);
    }
    else
    {
        m_cameraInitialized = false;
    }
}

void CharacterFollowCameraController::initializeCameraForCharacter(Camera& camera)
{
    
    if (!m_character)
    {
        return;
    }
        
    // Get character position
    EntityTransform characterTransform = m_character->getTransform();
    Float3 characterPos = characterTransform.position;
    
    
    // Check if character position has valid values
    if (isnan(characterPos.x) || isnan(characterPos.y) || isnan(characterPos.z))
    {
        return;
    }
    
    // Position camera behind and above the character
    Float3 newCameraPos = characterPos + m_cameraOffset;
    
    // Set camera position directly (no posDelta corruption)
    camera.pos = newCameraPos;
    camera.posDelta = Float3(0.0f, 0.0f, 0.0f);
    
    // Make camera look at character
    Float3 targetPos = m_character->getCameraTargetPosition();
    Float3 lookVector = targetPos - camera.pos;
    
    // Validate lookVector components first
    if (isnan(lookVector.x) || isnan(lookVector.y) || isnan(lookVector.z))
    {
        camera.dir = Float3(0.0f, 0.0f, -1.0f);
        camera.yaw = 0.0f;
        camera.pitch = 0.0f;
    }
    else
    {
        float lookLength = length(lookVector);
        
        // Proper NaN and zero-length checking
        if (isnan(lookLength) || lookLength < 0.001f)
        {
            camera.dir = Float3(0.0f, 0.0f, -1.0f);
            camera.yaw = 0.0f;
            camera.pitch = 0.0f;
        }
        else
        {
            Float3 lookDirection = lookVector / lookLength;  // Manual normalize
            
            // Validate normalized direction
            if (isnan(lookDirection.x) || isnan(lookDirection.y) || isnan(lookDirection.z))
            {
                camera.dir = Float3(0.0f, 0.0f, -1.0f);
                camera.yaw = 0.0f;
                camera.pitch = 0.0f;
            }
            else
            {
                camera.dir = lookDirection;
                
                // Calculate initial yaw and pitch from the look direction
                camera.yaw = atan2(-lookDirection.x, -lookDirection.z);
                camera.pitch = asin(clampf(lookDirection.y, -1.0f, 1.0f)); // Clamp input to asin
                camera.pitch = clampf(camera.pitch, -PI_OVER_2 + 0.1f, PI_OVER_2 - 0.1f);
                
                // Validate final yaw/pitch values
                if (isnan(camera.yaw) || isnan(camera.pitch))
                {
                    camera.yaw = 0.0f;
                    camera.pitch = 0.0f;
                }
            }
        }
    }
    
    // Update camera matrices
    camera.updateMatrices();
    
    // Store initial target
    m_cameraTarget = targetPos;
    
    // Mark camera as initialized
    m_cameraInitialized = true;
    
}

void CharacterFollowCameraController::updateCameraFollowing(Camera& camera, float deltaTime)
{
    if (!m_character || !m_cameraInitialized)
        return;
        
    // Get character position and orientation
    EntityTransform characterTransform = m_character->getTransform();
    Float3 characterPos = characterTransform.position;
    Float3 characterTarget = m_character->getCameraTargetPosition();
    
    // Calculate desired camera position based on yaw/pitch
    float distance = length(m_cameraOffset);
    Float3 cameraDirection = YawPitchToDir(camera.yaw, camera.pitch);
    Float3 desiredCameraPos = characterTarget - cameraDirection * distance;
    
    // Smooth camera position interpolation (GTA5-style)
    Float3 currentCameraPos = camera.pos;
    
    // Protect against corrupted camera position
    if (isnan(currentCameraPos.x) || isnan(currentCameraPos.y) || isnan(currentCameraPos.z) ||
        abs(currentCameraPos.x) > 1e10f || abs(currentCameraPos.y) > 1e10f || abs(currentCameraPos.z) > 1e10f)
    {
        currentCameraPos = characterPos + m_cameraOffset;
        camera.pos = currentCameraPos;
    }
    
    Float3 positionDelta = desiredCameraPos - currentCameraPos;
    
    // Validate position delta before distance calculation
    if (isnan(positionDelta.x) || isnan(positionDelta.y) || isnan(positionDelta.z))
    {
        camera.pos = characterPos + m_cameraOffset;
        camera.posDelta = Float3(0.0f, 0.0f, 0.0f);
        return;
    }
    
    // Use different smoothing speeds based on distance
    float followSpeed = m_cameraFollowSpeed;
    float distanceToTarget = length(positionDelta);
    
    // Validate distance calculation
    if (isnan(distanceToTarget) || distanceToTarget < 0.0f)
    {
        return;
    }
    
    // Faster following when far away, slower when close (for smoothness)
    if (distanceToTarget > 5.0f)
        followSpeed *= 2.0f;
    else if (distanceToTarget < 1.0f)
        followSpeed *= 0.5f;
    
    // Apply smooth following
    Float3 smoothedDelta = positionDelta * followSpeed * deltaTime;
    
    // Validate smoothed delta
    if (isnan(smoothedDelta.x) || isnan(smoothedDelta.y) || isnan(smoothedDelta.z))
    {
        return;
    }
    
    // Update camera position directly (no posDelta corruption)
    Float3 newCameraPos = currentCameraPos + smoothedDelta;
    camera.pos = newCameraPos;
    camera.posDelta = Float3(0.0f, 0.0f, 0.0f); // Always keep this at zero
    
    // Update camera direction to look at character
    Float3 lookDirection = normalize(characterTarget - newCameraPos);
    camera.dir = lookDirection;
    
    // Update matrices
    camera.updateMatrices();
    
    // Store current target for next frame
    m_cameraTarget = characterTarget;
    
    // Update character orientation based on movement direction only (not camera)
    const CharacterMovement& movement = m_character->getMovement();
    // Let Character::updateSmoothRotation() handle the rotation smoothly
    // Remove immediate setYaw() call to prevent snapping
}