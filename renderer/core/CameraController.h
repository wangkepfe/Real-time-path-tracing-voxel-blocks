#pragma once

#include "../shaders/LinearMath.h"

class Character;
struct Camera;

/**
 * Abstract base class for camera control strategies.
 * Each camera mode (free move, character follow, gameplay) should inherit from this.
 * 
 * DESIGN PRINCIPLE: Only CameraController implementations should modify camera state.
 * This centralizes all camera update logic and prevents scattered camera.update() calls.
 */
class CameraController
{
public:
    virtual ~CameraController() = default;
    
    /**
     * Update camera position and orientation based on input and deltaTime.
     * This is the ONLY method that should modify camera state.
     * 
     * @param camera Camera to update (modified in-place)
     * @param deltaTime Frame time in milliseconds
     */
    virtual void updateCamera(Camera& camera, float deltaTime) = 0;
    
    /**
     * Handle mouse movement for camera control.
     * 
     * @param camera Camera to update
     * @param deltax Mouse movement in X direction
     * @param deltay Mouse movement in Y direction
     */
    virtual void handleMouseMovement(Camera& camera, float deltax, float deltay) = 0;
    
    /**
     * Set movement input state.
     * Different controllers may interpret these differently.
     */
    virtual void setMovementInput(bool moveW, bool moveS, bool moveA, bool moveD, bool moveC, bool moveX) = 0;
    
    /**
     * Set speed modifier state (for running, sprinting, etc.)
     * Default implementation does nothing - controllers can override if needed.
     */
    virtual void setSpeedModifiers(bool slowMode, bool fastMode) {}
    
    /**
     * Called when switching to this controller.
     * Used for initialization and smooth transitions.
     */
    virtual void onActivate(Camera& camera) {}
    
    /**
     * Called when switching away from this controller.
     * Used for cleanup.
     */
    virtual void onDeactivate(Camera& camera) {}
    
    /**
     * Returns a human-readable name for debugging.
     */
    virtual const char* getName() const = 0;

protected:
    // Common movement input state
    bool m_moveW = false;
    bool m_moveS = false;
    bool m_moveA = false;
    bool m_moveD = false;
    bool m_moveC = false;
    bool m_moveX = false;
    
    // Movement speeds - these will be updated from global settings
    float m_moveSpeed = 0.003f;
    float m_cursorMoveSpeed = 0.001f;
    
    // Update movement parameters from global settings
    void updateFromGlobalSettings();
};