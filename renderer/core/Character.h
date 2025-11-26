#pragma once

#include "Entity.h"
#include "../shaders/LinearMath.h"

// Character physics properties
struct CharacterPhysics
{
    Float3 velocity = Float3(0.0f, 0.0f, 0.0f);
    Float3 acceleration = Float3(0.0f, 0.0f, 0.0f);

    // Collision cylinder dimensions
    float radius = 0.3f; // 0.6 tiles wide -> 0.3 radius
    float height = 1.8f; // 1.8 tiles height

    // Physics constants
    float gravity = -9.81f;
    float friction = 0.8f;
    float maxSpeed = 5.0f;
    float jumpForce = 6.0f;
    bool isGrounded = false;
    bool canJump = true;
};

// Character movement state
struct CharacterMovement
{
    bool moveForward = false;
    bool moveBackward = false;
    bool moveLeft = false;
    bool moveRight = false;
    bool jump = false;
    bool isRunning = false; // Whether running mode is active
    bool isSneaking = false; // Whether sneaking mode is active

    Float3 moveDirection = Float3(0.0f, 0.0f, 0.0f);
    float moveSpeed = 3.0f;

    // Movement tracking for animation
    float currentSpeed = 0.0f; // Current horizontal movement speed

    // Character orientation (separate from camera)
    float yaw = 0.0f;           // Current character rotation around Y axis
    float targetYaw = 0.0f;     // Target yaw for smooth rotation
    float rotationSpeed = 8.0f; // How fast character rotates (radians per second)
};

// Animation states for the character
enum class CharacterAnimationState
{
    Idle,    // Standing still
    Walking, // Walking/running
    Blending // Transitioning between states
};

// Three-animation system: idle, walk, run
struct CharacterAnimation
{
    int idleClipIndex = -1;  // Index of idle animation clip
    int walkClipIndex = -1;  // Index of walk animation clip
    int runClipIndex = -1;   // Index of run animation clip
    int placeClipIndex = -1; // Index of place animation clip
    int sneakClipIndex = -1; // Index of sneak animation clip

    // Animation blending ratios
    float blendRatio = 0.0f;     // Blend ratio for current animation pair
    float animationSpeed = 1.0f; // Animation playback speed

    // Thresholds (loaded from global settings)
    float walkSpeedThreshold = 0.1f;
    float mediumSpeedThreshold = 2.5f;
    float runSpeedThreshold = 0.2f;
    float runMediumSpeedThreshold = 4.0f;

    // Animation time tracking
    float idleTime = 0.0f;
    float walkTime = 0.0f;
    float runTime = 0.0f;

    // Mode switching detection
    bool previousRunningMode = false;
};

class Character : public Entity
{
public:
    Character(const EntityTransform &transform);
    ~Character();

    // Override Entity update to include character-specific logic
    void update(float deltaTime) override;

    // Character movement control
    void setMovementInput(bool forward, bool backward, bool left, bool right, bool jump, bool isRunning, bool isSneaking = false);
    void setMovementDirection(const Float3 &direction);
    void setYaw(float yaw);
    void updateSmoothRotation(float deltaTime);

    // Physics
    void applyForce(const Float3 &force);
    void checkGroundCollision();
    bool checkCylinderCollision(const Float3 &newPosition);

    // Getters
    const CharacterPhysics &getPhysics() const { return m_physics; }
    const CharacterMovement &getMovement() const { return m_movement; }
    Float3 getForwardVector() const;
    Float3 getRightVector() const;
    bool isGrounded() const { return m_physics.isGrounded; }

    // Character camera following
    Float3 getCameraTargetPosition() const;

    // Animation control
    void initializeAnimationClips();
    void updateTwoStageAnimation(float deltaTime);
    void updateAnimationTimes(float deltaTime);
    void blendTwoAnimations(int anim1Index, int anim2Index, float ratio); // ratio: 0.0 = full anim1, 1.0 = full anim2
    void setAnimationSpeed(float speed);
    float getBlendRatio() const { return m_animation.blendRatio; }

    // Place animation control
    void triggerPlaceAnimation();

private:
    CharacterPhysics m_physics;
    CharacterMovement m_movement;
    CharacterAnimation m_animation;

    // Cache for consistent ground height calculations
    float m_lastCalculatedGroundHeight = 0.0f;
    bool m_groundHeightValid = false;

    // Internal physics methods
    void updatePhysics(float deltaTime);
    void updateMovement(float deltaTime);
    void resolveCollisions(Float3 &newPosition);

    // Terrain interaction
    float getTerrainHeightAt(const Float3 &worldPos);
    bool isPositionValid(const Float3 &position);
    bool hasGroundSupport(const Float3 &position, float groundHeight);
    bool ensureStandingClearance(Float3 &position);
};
