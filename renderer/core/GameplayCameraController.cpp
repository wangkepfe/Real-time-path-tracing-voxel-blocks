#include "GameplayCameraController.h"
#include "../shaders/Camera.h"
#include "../shaders/LinearMath.h"
#include "GlobalSettings.h"
#include "../../voxelengine/VoxelEngine.h"
#include "voxelengine/BlockType.h"

GameplayCameraController::GameplayCameraController()
{
    // Movement speeds will be updated from global settings
}

void GameplayCameraController::updateCamera(Camera& camera, float deltaTime)
{
    // Update movement speeds and gameplay parameters from global settings
    updateFromGlobalSettings();
    auto &globalCamera = GlobalSettings::GetCameraMovementParams();
    m_height = globalCamera.gameplayHeight;
    
    auto &voxelEngine = VoxelEngine::Get();

    // Horizontal movement
    if (m_moveW || m_moveS || m_moveA || m_moveD)
    {
        Float3 movingDir{0};

        Float3 upDir = Float3(0, 1, 0);
        Float3 strafeRightDir = cross(camera.dir, upDir).normalize();
        Float3 frontDir = cross(upDir, strafeRightDir).normalize();

        if (m_moveW) movingDir += frontDir;
        if (m_moveS) movingDir -= frontDir;
        if (m_moveA) movingDir -= strafeRightDir;
        if (m_moveD) movingDir += strafeRightDir;

        Float3 horizontalMove = movingDir * deltaTime * m_moveSpeed;

        // Use multi-chunk collision detection
        Float3 newPos = camera.pos + horizontalMove;
        auto v0 = voxelEngine.getVoxelAtGlobal((unsigned int)newPos.x, (unsigned int)newPos.y, (unsigned int)newPos.z);
        auto v1 = voxelEngine.getVoxelAtGlobal((unsigned int)newPos.x, (unsigned int)(newPos.y - 1), (unsigned int)newPos.z);

        if (v0.id == BlockTypeEmpty && v1.id == BlockTypeEmpty)
        {
            camera.pos = camera.pos + horizontalMove;
        }
    }

    // Vertical movement (gravity and jumping)
    
    // Free fall
    float fallAccel = 9.8e-6f;
    m_fallSpeed += fallAccel * deltaTime;
    Float3 verticalMove = Float3(0.0f, -m_fallSpeed * deltaTime, 0.0f);
    
    Float3 fallCheckPos = camera.pos + verticalMove - Float3(0, m_height, 0);
    auto v2 = voxelEngine.getVoxelAtGlobal((unsigned int)fallCheckPos.x, (unsigned int)fallCheckPos.y, (unsigned int)fallCheckPos.z);

    if (m_fallSpeed > 0.0f && v2.id != BlockTypeEmpty)
    {
        while (v2.id != BlockTypeEmpty)
        {
            verticalMove.y += 1.0f;
            fallCheckPos = camera.pos + verticalMove - Float3(0, m_height, 0);
            v2 = voxelEngine.getVoxelAtGlobal((unsigned int)fallCheckPos.x, (unsigned int)fallCheckPos.y, (unsigned int)fallCheckPos.z);
        }
        verticalMove.y += static_cast<float>(static_cast<int>((camera.pos.y + verticalMove.y) - m_height)) + m_height - (camera.pos.y + verticalMove.y);
        m_fallSpeed = 0.0f;
    }
    
    // Apply vertical movement
    camera.pos = camera.pos + verticalMove;

    // Handle head bump to roof
    Float3 headPos = camera.pos + Float3(0, 0.49f, 0);
    auto v3 = voxelEngine.getVoxelAtGlobal((unsigned int)headPos.x, (unsigned int)headPos.y, (unsigned int)headPos.z);
    if (m_fallSpeed < 0.0f && v3.id != BlockTypeEmpty)
    {
        m_fallSpeed = 0.0f;
    }
    
    // Jumping
    if (m_moveC) // Space key for jumping
    {
        m_fallSpeed = -0.008f;
    }
    
    // Ensure posDelta stays zero (we manage position directly)
    camera.posDelta = Float3(0.0f, 0.0f, 0.0f);
    
    // Update camera matrices
    camera.updateMatrices();
}

void GameplayCameraController::handleMouseMovement(Camera& camera, float deltax, float deltay)
{
    // Update movement speeds from global settings
    updateFromGlobalSettings();
    
    camera.yaw -= deltax * m_cursorMoveSpeed;
    camera.pitch -= deltay * m_cursorMoveSpeed;
    camera.pitch = clampf(camera.pitch, -PI_OVER_2 + 0.1f, PI_OVER_2 - 0.1f);
}

void GameplayCameraController::setMovementInput(bool moveW, bool moveS, bool moveA, bool moveD, bool moveC, bool moveX)
{
    m_moveW = moveW;
    m_moveS = moveS;
    m_moveA = moveA;
    m_moveD = moveD;
    m_moveC = moveC;
    // Note: moveX (down) is not used in gameplay mode
}

void GameplayCameraController::onActivate(Camera& camera)
{
    // Clear any movement input
    setMovementInput(false, false, false, false, false, false);
    
    // Reset physics state
    m_fallSpeed = 0.0f;
    
    // Ensure camera is in a clean state
    camera.posDelta = Float3(0.0f, 0.0f, 0.0f);
}