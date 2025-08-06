#include "Character.h"
#include "../util/DebugUtils.h"
#include "../voxelengine/VoxelEngine.h"
#include "../voxelengine/Voxel.h"
#include "../voxelengine/Block.h"
#include "GlobalSettings.h"
#include <iostream>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Character::Character(const EntityTransform &transform)
    : Entity(EntityTypeMinecraftCharacter, transform)
{
    // Initialize rotation values from transform
    m_movement.yaw = transform.rotation.y;
    m_movement.targetYaw = transform.rotation.y;

    // Initialize character at ground level
    checkGroundCollision();

    // Initialize animation clips after a short delay to ensure entity is loaded
    initializeAnimationClips();
}

Character::~Character()
{
}

void Character::update(float deltaTime)
{
    // Call parent Entity update for animation
    Entity::update(deltaTime);

    // Update character-specific logic
    updateMovement(deltaTime);
    updateSmoothRotation(deltaTime);
    updatePhysics(deltaTime);

    // Update two-stage animation system
    updateTwoStageAnimation(deltaTime);

    // Update animation times
    updateAnimationTimes(deltaTime);
}

void Character::setMovementInput(bool forward, bool backward, bool left, bool right, bool jump, bool isRunning, bool isSneaking)
{
    m_movement.moveForward = forward;
    m_movement.moveBackward = backward;
    m_movement.moveLeft = left;
    m_movement.moveRight = right;
    m_movement.jump = jump && m_physics.canJump && m_physics.isGrounded;
    
    // Sneaking and running are mutually exclusive
    if (isSneaking && isRunning)
    {
        m_movement.isSneaking = isSneaking; // Prioritize sneaking
        m_movement.isRunning = false;
    }
    else
    {
        m_movement.isRunning = isRunning;
        m_movement.isSneaking = isSneaking;
    }
}

void Character::setMovementDirection(const Float3 &direction)
{
    // Only normalize if there's actual direction input
    if (length(direction) > 0.01f)
    {
        m_movement.moveDirection = normalize(direction);
    }
    else
    {
        // No input - set to zero direction (don't normalize zero vector)
        m_movement.moveDirection = Float3(0.0f, 0.0f, 0.0f);
    }
}

void Character::setYaw(float yaw)
{
    m_movement.yaw = yaw;
    m_movement.targetYaw = yaw; // Also set target yaw to avoid smooth rotation

    // Update entity rotation
    EntityTransform transform = getTransform();
    transform.rotation.y = yaw;
    setTransform(transform);
}

void Character::updateSmoothRotation(float deltaTime)
{
    // Calculate the shortest angular distance between current and target yaw
    float angleDiff = m_movement.targetYaw - m_movement.yaw;

    // Normalize angle difference to [-π, π] range
    while (angleDiff > M_PI)
        angleDiff -= 2.0f * M_PI;
    while (angleDiff < -M_PI)
        angleDiff += 2.0f * M_PI;

    // Only rotate if there's a significant difference
    if (std::abs(angleDiff) > 0.01f)
    {
        // Calculate rotation step based on rotation speed
        float rotationStep = m_movement.rotationSpeed * deltaTime;

        // Clamp rotation step to not overshoot
        if (std::abs(angleDiff) < rotationStep)
        {
            m_movement.yaw = m_movement.targetYaw;
        }
        else
        {
            // Rotate towards target
            m_movement.yaw += (angleDiff > 0 ? rotationStep : -rotationStep);
        }

        // Normalize current yaw to [0, 2π] range
        while (m_movement.yaw < 0.0f)
            m_movement.yaw += 2.0f * M_PI;
        while (m_movement.yaw >= 2.0f * M_PI)
            m_movement.yaw -= 2.0f * M_PI;

        // Update entity rotation
        EntityTransform transform = getTransform();
        transform.rotation.y = m_movement.yaw;
        setTransform(transform);
    }
}

void Character::applyForce(const Float3 &force)
{
    m_physics.acceleration = m_physics.acceleration + force;
}

Float3 Character::getForwardVector() const
{
    float yaw = m_movement.yaw;
    return Float3(sin(yaw), 0.0f, cos(yaw));
}

Float3 Character::getRightVector() const
{
    float yaw = m_movement.yaw;
    return Float3(cos(yaw), 0.0f, -sin(yaw));
}

Float3 Character::getCameraTargetPosition() const
{
    EntityTransform transform = getTransform();
    return Float3(transform.position.x,
                  transform.position.y + 2.4f,
                  transform.position.z);
}

void Character::updateMovement(float deltaTime)
{
    // Update movement parameters from global settings
    auto &globalMovement = GlobalSettings::GetCharacterMovementParams();
    m_movement.rotationSpeed = globalMovement.rotationSpeed;
    m_physics.jumpForce = globalMovement.jumpForce;
    m_physics.gravity = globalMovement.gravity;
    m_physics.friction = globalMovement.friction;
    m_physics.radius = globalMovement.radius;
    m_physics.height = globalMovement.height;

    // Use the world-space movement direction set by InputHandler
    Float3 inputDirection = m_movement.moveDirection;

    // Check if there's any input
    bool hasMovementInput = m_movement.moveForward || m_movement.moveBackward || m_movement.moveLeft || m_movement.moveRight;
    float directionLength = length(inputDirection);

    // Only apply movement if there's actual input
    if (hasMovementInput && directionLength > 0.01f)
    {
        inputDirection = normalize(inputDirection);

        // Set target yaw based on desired movement direction
        m_movement.targetYaw = atan2(inputDirection.x, inputDirection.z);

        // Character moves in the direction it's currently facing, not the input direction
        Float3 currentFacingDirection = getForwardVector();

        // Use appropriate force based on movement mode (sneak < walk < run)
        float moveForce;
        if (m_movement.isSneaking)
        {
            moveForce = GlobalSettings::GetCharacterMovementParams().walkMoveForce * 0.4f; // Sneak is slower than walk
        }
        else if (m_movement.isRunning)
        {
            moveForce = GlobalSettings::GetCharacterMovementParams().runMoveForce;
        }
        else
        {
            moveForce = GlobalSettings::GetCharacterMovementParams().walkMoveForce;
        }

        Float3 forceVector = currentFacingDirection * moveForce;
        applyForce(forceVector);
    }

    // Handle jumping
    if (m_movement.jump && m_physics.isGrounded)
    {
        m_physics.velocity.y = m_physics.jumpForce;
        m_physics.isGrounded = false;
        m_physics.canJump = false;
    }
}

void Character::updatePhysics(float deltaTime)
{
    // Validate deltaTime (should now be in seconds, not milliseconds)
    if (isnan(deltaTime) || deltaTime < 0.0f)
    {
        return;
    }

    EntityTransform transform = getTransform();
    Float3 currentPos = transform.position;

    // Validate current position
    if (isnan(currentPos.x) || isnan(currentPos.y) || isnan(currentPos.z))
    {
        currentPos = Float3(32.0f, 10.0f, 38.0f); // Default spawn position
        transform.position = currentPos;
        setTransform(transform);
        return;
    }

    // Apply gravity
    if (!m_physics.isGrounded)
    {
        applyForce(Float3(0.0f, m_physics.gravity, 0.0f));
    }

    // Update velocity based on acceleration
    Float3 velocityDelta = m_physics.acceleration * deltaTime;

    // Validate velocity delta
    if (isnan(velocityDelta.x) || isnan(velocityDelta.y) || isnan(velocityDelta.z))
    {
        return;
    }

    m_physics.velocity = m_physics.velocity + velocityDelta;

    // Validate velocity after acceleration
    if (isnan(m_physics.velocity.x) || isnan(m_physics.velocity.y) || isnan(m_physics.velocity.z))
    {
        m_physics.velocity = Float3(0.0f, 0.0f, 0.0f);
        return;
    }

    // Apply friction on horizontal movement
    m_physics.velocity.x *= (1.0f - m_physics.friction * deltaTime);
    m_physics.velocity.z *= (1.0f - m_physics.friction * deltaTime);

    // Validate velocity after friction
    if (isnan(m_physics.velocity.x) || isnan(m_physics.velocity.y) || isnan(m_physics.velocity.z))
    {
        m_physics.velocity = Float3(0.0f, 0.0f, 0.0f);
        return;
    }

    // Clamp horizontal speed based on movement mode
    auto &globalMovement = GlobalSettings::GetCharacterMovementParams();
    float currentMaxSpeed;
    if (m_movement.isSneaking)
    {
        currentMaxSpeed = globalMovement.walkMaxSpeed * 0.4f; // Sneak is slower than walk
    }
    else if (m_movement.isRunning)
    {
        currentMaxSpeed = globalMovement.runMaxSpeed;
    }
    else
    {
        currentMaxSpeed = globalMovement.walkMaxSpeed;
    }

    Float3 horizontalVel = Float3(m_physics.velocity.x, 0.0f, m_physics.velocity.z);
    float horizontalSpeed = length(horizontalVel);
    if (horizontalSpeed > currentMaxSpeed)
    {
        horizontalVel = normalize(horizontalVel) * currentMaxSpeed;
        m_physics.velocity.x = horizontalVel.x;
        m_physics.velocity.z = horizontalVel.z;
    }

    // Calculate new position
    Float3 positionDelta = m_physics.velocity * deltaTime;

    // Validate position delta
    if (isnan(positionDelta.x) || isnan(positionDelta.y) || isnan(positionDelta.z))
    {
        return;
    }

    Float3 newPosition = currentPos + positionDelta;

    // Sneak mode: proactively constrain movement to stay on safe ground
    if (m_movement.isSneaking && m_physics.isGrounded)
    {
        // Step-by-step movement validation - move in small increments to stay safe
        Float3 originalMovement = newPosition - currentPos;
        float totalDistance = length(originalMovement);
        
        if (totalDistance > 0.001f)
        {
            Float3 moveDirection = normalize(originalMovement);
            const float stepSize = 0.05f; // Small step size for precision
            int maxSteps = static_cast<int>(totalDistance / stepSize) + 1;
            
            Float3 safePosition = currentPos;
            
            for (int step = 0; step < maxSteps; step++)
            {
                float currentStepSize = std::min(stepSize, totalDistance - (step * stepSize));
                if (currentStepSize <= 0) break;
                
                Float3 testPosition = safePosition + moveDirection * currentStepSize;
                
                // Check safety for the entire character footprint, not just center
                bool isStepSafe = true;
                float radius = m_physics.radius;
                
                // Check multiple points within character's circular footprint
                const int footprintChecks = 8;
                for (int i = 0; i < footprintChecks && isStepSafe; i++)
                {
                    float angle = (2.0f * M_PI * i) / footprintChecks;
                    Float3 footprintOffset = Float3(cos(angle) * radius, 0, sin(angle) * radius);
                    Float3 footprintPos = testPosition + footprintOffset;
                    
                    float footprintGroundHeight = getTerrainHeightAt(footprintPos);
                    float footprintDropDistance = footprintPos.y - footprintGroundHeight;
                    
                    if (footprintDropDistance > 1.0f)
                    {
                        isStepSafe = false;
                    }
                }
                
                // Also check the center
                float centerGroundHeight = getTerrainHeightAt(testPosition);
                float centerDropDistance = testPosition.y - centerGroundHeight;
                if (centerDropDistance > 1.0f)
                {
                    isStepSafe = false;
                }
                
                if (isStepSafe)
                {
                    // This step is safe, accept it
                    safePosition = testPosition;
                }
                else
                {
                    // This step would be unsafe, stop here and try edge sliding
                    // Find the edge direction by testing perpendicular movements
                    Float3 edgeDirection = Float3(0, 0, 0);
                    bool foundEdge = false;
                    
                    // Test perpendicular directions to find edge
                    Float3 perpendicular1 = Float3(-moveDirection.z, 0, moveDirection.x);
                    Float3 perpendicular2 = Float3(moveDirection.z, 0, -moveDirection.x);
                    
                    for (Float3 perpDir : {perpendicular1, perpendicular2})
                    {
                        Float3 edgeTestPos = safePosition + perpDir * currentStepSize;
                        
                        // Check safety for entire footprint
                        bool isEdgeSafe = true;
                        
                        // Check footprint points
                        for (int i = 0; i < footprintChecks && isEdgeSafe; i++)
                        {
                            float angle = (2.0f * M_PI * i) / footprintChecks;
                            Float3 footprintOffset = Float3(cos(angle) * radius, 0, sin(angle) * radius);
                            Float3 footprintPos = edgeTestPos + footprintOffset;
                            
                            float footprintGroundHeight = getTerrainHeightAt(footprintPos);
                            float footprintDropDistance = footprintPos.y - footprintGroundHeight;
                            
                            if (footprintDropDistance > 1.0f)
                            {
                                isEdgeSafe = false;
                            }
                        }
                        
                        // Check center
                        float edgeGroundHeight = getTerrainHeightAt(edgeTestPos);
                        float edgeDropDistance = edgeTestPos.y - edgeGroundHeight;
                        if (edgeDropDistance > 1.0f)
                        {
                            isEdgeSafe = false;
                        }
                        
                        if (isEdgeSafe)
                        {
                            edgeDirection = perpDir;
                            foundEdge = true;
                            break;
                        }
                    }
                    
                    if (foundEdge)
                    {
                        // Move along the edge instead - already validated as safe above
                        safePosition = safePosition + edgeDirection * currentStepSize;
                    }
                    break; // Stop stepping forward
                }
            }
            
            newPosition = safePosition;
        }
    }

    // Check and resolve collisions
    resolveCollisions(newPosition);

    // Validate final position and apply bounds checking
    if (isnan(newPosition.x) || isnan(newPosition.y) || isnan(newPosition.z))
    {
        newPosition = Float3(32.0f, 10.0f, 38.0f); // Reset to spawn
    }

    // Apply reasonable world bounds (prevent character from going to extreme coordinates)
    const float maxWorldCoord = 10000.0f;
    if (std::abs(newPosition.x) > maxWorldCoord || std::abs(newPosition.y) > maxWorldCoord || std::abs(newPosition.z) > maxWorldCoord)
    {
        newPosition = Float3(32.0f, 10.0f, 38.0f);     // Reset to spawn
        m_physics.velocity = Float3(0.0f, 0.0f, 0.0f); // Stop movement
    }

    // Update transform

    transform.position = newPosition;
    setTransform(transform);

    // Reset acceleration for next frame
    m_physics.acceleration = Float3(0.0f, 0.0f, 0.0f);

    // Update grounded state
    checkGroundCollision();

    // Update movement tracking for animation
    Float3 currentVelocity = Float3(m_physics.velocity.x, 0.0f, m_physics.velocity.z);
    m_movement.currentSpeed = length(currentVelocity);
}

void Character::resolveCollisions(Float3 &newPosition)
{
    EntityTransform transform = getTransform();
    Float3 currentPos = transform.position;

    // Check terrain collision and adjust Y position
    float groundHeight = getTerrainHeightAt(newPosition);
    float characterBottom = newPosition.y;

    // Cache the ground height for consistent use in checkGroundCollision
    m_lastCalculatedGroundHeight = groundHeight;
    m_groundHeightValid = true;

    // CLIFF EDGE FIX: Check if character is close enough to ground to be considered grounded
    // This handles the case where character is floating slightly above cliff edge
    float distanceToGround = characterBottom - groundHeight;
    bool shouldBeGrounded = false;
    bool isCliffEdgeCase = false;
    bool hasSupport = false;

    if (characterBottom <= groundHeight)
    {
        // Standard case: character is at or below ground level
        shouldBeGrounded = true;
    }
    else if (distanceToGround < 0.2f && m_physics.velocity.y <= -2.0f)
    {
        // CLIFF EDGE CASE: Character is floating close to ground and falling FAST
        // Only apply ground support check when character is already falling significantly
        // This prevents interference with normal cliff walking
        isCliffEdgeCase = true;
        hasSupport = hasGroundSupport(newPosition, groundHeight);
        shouldBeGrounded = hasSupport;
    }

    if (shouldBeGrounded)
    {
        float oldY = newPosition.y;
        newPosition.y = groundHeight + 0.01f; // Small offset to ensure character is above ground

        if (m_physics.velocity.y <= 0.0f)
        {
            m_physics.velocity.y = 0.0f;
            m_physics.isGrounded = true;
            m_physics.canJump = true;
        }
    }
    else
    {
        // Character should fall or is jumping
        if (distanceToGround > 0.3f) // Only set airborne if significantly above ground
        {
            m_physics.isGrounded = false;
        }
    }

    // Check ceiling/roof collision when moving upward
    if (m_physics.velocity.y > 0.0f) // Only check when moving upward
    {
        float characterTop = newPosition.y + m_physics.height;

        // Check if character's head would hit a solid block
        int blockX = static_cast<int>(floor(newPosition.x));
        int blockZ = static_cast<int>(floor(newPosition.z));
        int ceilingY = static_cast<int>(floor(characterTop));

        Voxel ceilingVoxel = VoxelEngine::Get().getVoxelAtGlobal(blockX, ceilingY, blockZ);
        if (ceilingVoxel.id != BlockTypeEmpty) // Hit solid block above
        {
            // Stop upward movement and position character just below the ceiling
            newPosition.y = static_cast<float>(ceilingY) - m_physics.height - 0.01f;
            m_physics.velocity.y = 0.0f; // Stop upward velocity
        }
    }

    // Check horizontal cylinder collision with terrain
    bool positionValid = isPositionValid(newPosition);

    if (!positionValid)
    {
        // Calculate movement delta
        Float3 movementDelta = newPosition - currentPos;
        Float3 bestPosition = newPosition; // FIXED: Preserve Y position from vertical collision resolution
        bestPosition.x = currentPos.x;     // Only reset horizontal position
        bestPosition.z = currentPos.z;
        bool foundValidPosition = false;

        // STEP 1: Try pure axis sliding (X and Z independently)
        Float3 slideX = Float3(newPosition.x, newPosition.y, currentPos.z);
        Float3 slideZ = Float3(currentPos.x, newPosition.y, newPosition.z);

        if (isPositionValid(slideX))
        {
            bestPosition = slideX;
            foundValidPosition = true;
            m_physics.velocity.z *= 0.3f; // Reduce blocked direction
        }
        else if (isPositionValid(slideZ))
        {
            bestPosition = slideZ;
            foundValidPosition = true;
            m_physics.velocity.x *= 0.3f; // Reduce blocked direction
        }

        // STEP 2: Try step-up for stairs/small obstacles
        // CLIFF EDGE FIX: Only allow step-up when character is trying to move UP, not when falling DOWN
        if (!foundValidPosition)
        {
            // Check if character is trying to move upward (going up stairs) vs downward (falling off cliffs)
            float currentGroundHeight = getTerrainHeightAt(currentPos);
            float newGroundHeight = getTerrainHeightAt(newPosition);
            bool movingUpward = newGroundHeight > currentGroundHeight + 0.1f; // Moving to higher ground
            bool isFalling = m_physics.velocity.y < -1.0f;                    // Falling with significant downward velocity

            if (movingUpward && !isFalling)
            {
                // Character is trying to go up stairs - allow step-up
                for (float stepHeight = 0.25f; stepHeight <= 1.0f; stepHeight += 0.25f)
                {
                    Float3 stepUp = Float3(newPosition.x, newPosition.y + stepHeight, newPosition.z);
                    if (isPositionValid(stepUp))
                    {
                        bestPosition = stepUp;
                        foundValidPosition = true;
                        break;
                    }
                }
            }
        }

        // STEP 3: Try diagonal movement alternatives (corner escape)
        if (!foundValidPosition)
        {
            // Try moving at different angles around the obstacle
            float angles[] = {0.25f, -0.25f, 0.5f, -0.5f, 0.75f, -0.75f, 1.0f, -1.0f};
            float movementMagnitude = length(movementDelta);

            for (float angleOffset : angles)
            {
                Float3 alternativeDir = Float3(
                    movementDelta.x * cos(angleOffset) - movementDelta.z * sin(angleOffset),
                    movementDelta.y,
                    movementDelta.x * sin(angleOffset) + movementDelta.z * cos(angleOffset));
                Float3 alternativePos = currentPos + alternativeDir;

                if (isPositionValid(alternativePos))
                {
                    bestPosition = alternativePos;
                    foundValidPosition = true;
                    break;
                }
            }
        }

        // STEP 4: Push-out mechanism (move away from closest obstacle)
        if (!foundValidPosition)
        {
            // Try small movements in cardinal directions to escape
            Float3 escapeDirections[] = {
                Float3(0.1f, 0, 0), Float3(-0.1f, 0, 0),
                Float3(0, 0, 0.1f), Float3(0, 0, -0.1f),
                Float3(0.1f, 0, 0.1f), Float3(-0.1f, 0, 0.1f),
                Float3(0.1f, 0, -0.1f), Float3(-0.1f, 0, -0.1f)};

            for (Float3 escapeDir : escapeDirections)
            {
                Float3 escapePos = currentPos + escapeDir;
                if (isPositionValid(escapePos))
                {
                    bestPosition = escapePos;
                    foundValidPosition = true;
                    // Add escape velocity
                    m_physics.velocity = m_physics.velocity + escapeDir * 2.0f;
                    break;
                }
            }
        }

        // Apply the best position found
        Float3 oldPosition = newPosition;
        newPosition = bestPosition;

        if (!foundValidPosition)
        {
            // Gradually reduce velocity to prevent infinite attempts
            m_physics.velocity.x *= 0.8f;
            m_physics.velocity.z *= 0.8f;
        }
    }
}

void Character::checkGroundCollision()
{
    EntityTransform transform = getTransform();
    Float3 position = transform.position;

    // Use cached ground height if available to ensure consistency with resolveCollisions
    float groundHeight;
    bool usedCache = false;
    if (m_groundHeightValid)
    {
        groundHeight = m_lastCalculatedGroundHeight;
        m_groundHeightValid = false; // Reset for next frame
        usedCache = true;
    }
    else
    {
        // Fallback: recalculate if cache not available
        groundHeight = getTerrainHeightAt(position);
    }

    float characterBottom = position.y;
    float distance = std::abs(characterBottom - groundHeight);
    bool wasGrounded = m_physics.isGrounded;

    if (distance < 0.1f)
    {
        m_physics.isGrounded = true;
        m_physics.canJump = true;
    }
    else
    {
        m_physics.isGrounded = false;
    }
}

bool Character::checkCylinderCollision(const Float3 &newPosition)
{
    return !isPositionValid(newPosition);
}

float Character::getTerrainHeightAt(const Float3 &worldPos)
{
    // NEW DESIGN: Find the appropriate floor surface based on character's current position
    // This supports multi-level structures: buildings with floors, caves, underground areas
    // Instead of finding the topmost block, we find the nearest valid floor below/above the character

    int blockX = static_cast<int>(floor(worldPos.x));
    int blockZ = static_cast<int>(floor(worldPos.z));
    int currentY = static_cast<int>(floor(worldPos.y));

    // First, look downward from current position to find the floor below
    for (int y = currentY; y >= 0; y--)
    {
        Voxel voxel = VoxelEngine::Get().getVoxelAtGlobal(blockX, y, blockZ);
        if (voxel.id != BlockTypeEmpty) // Found solid block
        {
            // Check if there's air space above this block for character to stand
            Voxel aboveVoxel = VoxelEngine::Get().getVoxelAtGlobal(blockX, y + 1, blockZ);
            if (aboveVoxel.id == BlockTypeEmpty) // Air space above
            {
                return static_cast<float>(y + 1); // Top surface of the solid block
            }
        }
    }

    // If no floor found below, look slightly upward (for cases where character is falling through)
    for (int y = currentY + 1; y <= currentY + 3 && y < 256; y++)
    {
        Voxel voxel = VoxelEngine::Get().getVoxelAtGlobal(blockX, y, blockZ);
        if (voxel.id != BlockTypeEmpty) // Found solid block
        {
            // Check if there's air space above this block
            Voxel aboveVoxel = VoxelEngine::Get().getVoxelAtGlobal(blockX, y + 1, blockZ);
            if (aboveVoxel.id == BlockTypeEmpty) // Air space above
            {
                return static_cast<float>(y + 1); // Top surface of the solid block
            }
        }
    }

    return 0.0f; // Fallback ground level
}

bool Character::isPositionValid(const Float3 &position)
{
    // SIMPLE FIX: If character is grounded and stable, allow horizontal movement
    bool isGrounded = m_physics.isGrounded;
    bool isStable = std::abs(m_physics.velocity.y) < 0.1f; // Not moving vertically much

    if (isGrounded && isStable)
    {
        // Character is stable on ground - allow horizontal movement
        // Only check for collisions at head level to prevent walking through walls
        float radius = m_physics.radius;
        float height = m_physics.height;
        float headLevel = position.y + height * 0.8f; // Check near the top of character

        const int checkPoints = 8;
        for (int i = 0; i < checkPoints; i++)
        {
            float angle = (2.0f * M_PI * i) / checkPoints;
            float checkX = position.x + radius * cos(angle);
            float checkZ = position.z + radius * sin(angle);

            int blockX = static_cast<int>(floor(checkX));
            int blockY = static_cast<int>(floor(headLevel));
            int blockZ = static_cast<int>(floor(checkZ));

            Voxel voxel = VoxelEngine::Get().getVoxelAtGlobal(blockX, blockY, blockZ);
            if (voxel.id != BlockTypeEmpty)
            {
                return false; // Head would hit something
            }
        }

        return true; // No head collisions - movement allowed
    }

    // Character is not stable - use normal collision detection
    float radius = m_physics.radius;
    float height = m_physics.height;
    float checkStartY = position.y + 0.1f;

    // When falling, be more permissive
    if (m_physics.velocity.y < -1.0f)
    {
        checkStartY = position.y + height * 0.5f;
    }

    const int checkPoints = 8;
    for (int i = 0; i < checkPoints; i++)
    {
        float angle = (2.0f * M_PI * i) / checkPoints;
        float checkX = position.x + radius * cos(angle);
        float checkZ = position.z + radius * sin(angle);

        for (float checkY = checkStartY; checkY < position.y + height; checkY += 0.5f)
        {
            int blockX = static_cast<int>(floor(checkX));
            int blockY = static_cast<int>(floor(checkY));
            int blockZ = static_cast<int>(floor(checkZ));

            Voxel voxel = VoxelEngine::Get().getVoxelAtGlobal(blockX, blockY, blockZ);
            if (voxel.id != BlockTypeEmpty)
            {
                return false;
            }
        }
    }

    return true;
}

bool Character::hasGroundSupport(const Float3 &position, float groundHeight)
{
    // Check if character has sufficient ground support within their footprint
    // This prevents cliff-edge oscillation by ensuring character either has solid support or falls

    float radius = m_physics.radius;
    int supportingBlocks = 0;
    int totalChecks = 0;

    // Check multiple points within character's circular footprint
    const int checkPoints = 8;
    for (int i = 0; i < checkPoints; i++)
    {
        float angle = (2.0f * M_PI * i) / checkPoints;
        float checkX = position.x + radius * 0.7f * cos(angle); // Check at 70% of radius
        float checkZ = position.z + radius * 0.7f * sin(angle);

        int blockX = static_cast<int>(floor(checkX));
        int blockZ = static_cast<int>(floor(checkZ));
        int blockY = static_cast<int>(floor(groundHeight)) - 1; // Block that should provide support

        Voxel supportBlock = VoxelEngine::Get().getVoxelAtGlobal(blockX, blockY, blockZ);
        totalChecks++;

        if (supportBlock.id != BlockTypeEmpty)
        {
            supportingBlocks++;
        }
        else
        {
        }
    }

    // Also check center point
    int centerX = static_cast<int>(floor(position.x));
    int centerZ = static_cast<int>(floor(position.z));
    int centerY = static_cast<int>(floor(groundHeight)) - 1;

    Voxel centerBlock = VoxelEngine::Get().getVoxelAtGlobal(centerX, centerY, centerZ);
    totalChecks++;
    if (centerBlock.id != BlockTypeEmpty)
    {
        supportingBlocks++;
    }
    else
    {
    }

    // Character needs at least 50% ground support to stay on cliff edge
    float supportRatio = static_cast<float>(supportingBlocks) / static_cast<float>(totalChecks);
    bool hasSupport = supportRatio >= 0.5f;

    return hasSupport;
}

void Character::initializeAnimationClips()
{
    if (!hasAnimation() || !getAnimationManager())
        return;

    // Find animation clip indices by name
    // GLTF has animations: "Walk", "Idle", "Run", "Place", "Sneak"
    m_animation.walkClipIndex = 0;  // First animation is "Walk"
    m_animation.idleClipIndex = 1;  // Second animation is "Idle"
    m_animation.runClipIndex = 2;   // Third animation is "Run"
    m_animation.placeClipIndex = 3; // Fourth animation is "Place"
    m_animation.sneakClipIndex = 4; // Fifth animation is "Sneak"

    // Start with idle animation
    m_animation.blendRatio = 0.0f;
    m_animation.animationSpeed = 1.0f;

    // Start manual blending between idle and walk (default state)
    getAnimationManager()->startManualBlend(m_animation.idleClipIndex, m_animation.walkClipIndex);
}

// THREE-ANIMATION SYSTEM: idle, walk, run

void Character::updateTwoStageAnimation(float deltaTime)
{
    if (!hasAnimation() || !getAnimationManager())
        return;

    // Update animation parameters from global settings
    auto &globalAnimation = GlobalSettings::GetCharacterAnimationParams();
    m_animation.walkSpeedThreshold = globalAnimation.walkSpeedThreshold;
    m_animation.mediumSpeedThreshold = globalAnimation.mediumSpeedThreshold;
    m_animation.runSpeedThreshold = globalAnimation.runSpeedThreshold;
    m_animation.runMediumSpeedThreshold = globalAnimation.runMediumSpeedThreshold;
    m_animation.animationSpeed = globalAnimation.animationSpeed;

    const float currentSpeed = m_movement.currentSpeed;
    // Use appropriate max speed based on running mode
    auto &globalMovement = GlobalSettings::GetCharacterMovementParams();
    const float maxSpeed = m_movement.isRunning ? globalMovement.runMaxSpeed : globalMovement.walkMaxSpeed;

    // Check if running mode has changed - restart blend if it has
    bool modeChanged = (m_movement.isRunning != m_animation.previousRunningMode);
    if (modeChanged)
    {
        // Mode switched - need to restart animation blend
        if (m_movement.isRunning)
        {
            // Switched to running mode - start idle-run blend
            getAnimationManager()->startManualBlend(m_animation.idleClipIndex, m_animation.runClipIndex);
        }
        else
        {
            // Switched to walking mode - start idle-walk blend
            getAnimationManager()->startManualBlend(m_animation.idleClipIndex, m_animation.walkClipIndex);
        }
        m_animation.previousRunningMode = m_movement.isRunning;
    }

    if (m_movement.isRunning)
    {
        // RUNNING MODE: Use idle-run blending with run thresholds
        if (currentSpeed < m_animation.runSpeedThreshold)
        {
            // Below run threshold: Full idle
            blendTwoAnimations(m_animation.idleClipIndex, m_animation.runClipIndex, 0.0f);
            m_animation.blendRatio = 0.0f;
            setAnimationSpeed(m_animation.animationSpeed);
        }
        else if (currentSpeed <= m_animation.runMediumSpeedThreshold)
        {
            // STAGE 1: Blend from idle to run based on speed
            float stage1Progress = (currentSpeed - m_animation.runSpeedThreshold) /
                                   (m_animation.runMediumSpeedThreshold - m_animation.runSpeedThreshold);
            stage1Progress = std::min(1.0f, std::max(0.0f, stage1Progress));

            blendTwoAnimations(m_animation.idleClipIndex, m_animation.runClipIndex, stage1Progress);
            m_animation.blendRatio = stage1Progress;
            setAnimationSpeed(m_animation.animationSpeed);
        }
        else
        {
            // STAGE 2: Full run animation, scale speed based on movement
            blendTwoAnimations(m_animation.idleClipIndex, m_animation.runClipIndex, 1.0f);
            m_animation.blendRatio = 1.0f;

            float stage2Progress = (currentSpeed - m_animation.runMediumSpeedThreshold) /
                                   (maxSpeed - m_animation.runMediumSpeedThreshold);
            stage2Progress = std::min(1.0f, std::max(0.0f, stage2Progress));

            // Scale animation speed for walking
            float scaledSpeed = m_animation.animationSpeed + (stage2Progress * 0.5f);
            setAnimationSpeed(scaledSpeed);
        }
    }
    else
    {
        // WALKING MODE: Use idle-walk blending with walk thresholds
        if (currentSpeed < m_animation.walkSpeedThreshold)
        {
            // Below threshold: Full idle
            blendTwoAnimations(m_animation.idleClipIndex, m_animation.walkClipIndex, 0.0f);
            m_animation.blendRatio = 0.0f;
            setAnimationSpeed(m_animation.animationSpeed);
        }
        else if (currentSpeed <= m_animation.mediumSpeedThreshold)
        {
            // STAGE 1: Blend from idle to walk based on speed
            float stage1Progress = (currentSpeed - m_animation.walkSpeedThreshold) /
                                   (m_animation.mediumSpeedThreshold - m_animation.walkSpeedThreshold);
            stage1Progress = std::min(1.0f, std::max(0.0f, stage1Progress));

            blendTwoAnimations(m_animation.idleClipIndex, m_animation.walkClipIndex, stage1Progress);
            m_animation.blendRatio = stage1Progress;
            setAnimationSpeed(m_animation.animationSpeed);
        }
        else
        {
            // STAGE 2: Full walk animation, scale speed based on movement
            blendTwoAnimations(m_animation.idleClipIndex, m_animation.walkClipIndex, 1.0f);
            m_animation.blendRatio = 1.0f;

            float stage2Progress = (currentSpeed - m_animation.mediumSpeedThreshold) /
                                   (maxSpeed - m_animation.mediumSpeedThreshold);
            stage2Progress = std::min(1.0f, std::max(0.0f, stage2Progress));

            // Scale animation speed for walking
            float scaledSpeed = m_animation.animationSpeed + (stage2Progress * 0.5f);
            setAnimationSpeed(scaledSpeed);
        }
    }
    
    // Handle sneaking animation as additive
    if (m_movement.isSneaking)
    {
        // Start sneaking additive animation if not already active
        if (!getAnimationManager()->hasMultipleAdditiveAnimation(m_animation.sneakClipIndex))
        {
            getAnimationManager()->startMultipleAdditiveAnimation(m_animation.sneakClipIndex, 1.0f);
        }
    }
    else
    {
        // Stop sneaking additive animation if active
        if (getAnimationManager()->hasMultipleAdditiveAnimation(m_animation.sneakClipIndex))
        {
            getAnimationManager()->stopMultipleAdditiveAnimation(m_animation.sneakClipIndex);
        }
    }
}

void Character::updateAnimationTimes(float deltaTime)
{
    // Update animation times based on playback speed
    m_animation.idleTime += deltaTime * m_animation.animationSpeed;
    m_animation.walkTime += deltaTime * m_animation.animationSpeed;
    m_animation.runTime += deltaTime * m_animation.animationSpeed;

    // Handle looping
    AnimationClip *idleClip = getAnimationManager()->getAnimationClip(m_animation.idleClipIndex);
    AnimationClip *walkClip = getAnimationManager()->getAnimationClip(m_animation.walkClipIndex);
    AnimationClip *runClip = getAnimationManager()->getAnimationClip(m_animation.runClipIndex);

    if (idleClip && m_animation.idleTime > idleClip->duration)
    {
        m_animation.idleTime = fmod(m_animation.idleTime, idleClip->duration);
    }

    if (walkClip && m_animation.walkTime > walkClip->duration)
    {
        m_animation.walkTime = fmod(m_animation.walkTime, walkClip->duration);
    }

    if (runClip && m_animation.runTime > runClip->duration)
    {
        m_animation.runTime = fmod(m_animation.runTime, runClip->duration);
    }
}

void Character::blendTwoAnimations(int anim1Index, int anim2Index, float ratio)
{
    if (!hasAnimation() || !getAnimationManager())
        return;

    // Determine the correct time values based on which animations are being blended
    float anim1Time = m_animation.idleTime; // Default to idle time for first animation
    float anim2Time = m_animation.walkTime; // Default to walk time for second animation

    // Update times based on the specific animation indices
    if (anim1Index == m_animation.idleClipIndex)
        anim1Time = m_animation.idleTime;
    else if (anim1Index == m_animation.walkClipIndex)
        anim1Time = m_animation.walkTime;
    else if (anim1Index == m_animation.runClipIndex)
        anim1Time = m_animation.runTime;

    if (anim2Index == m_animation.idleClipIndex)
        anim2Time = m_animation.idleTime;
    else if (anim2Index == m_animation.walkClipIndex)
        anim2Time = m_animation.walkTime;
    else if (anim2Index == m_animation.runClipIndex)
        anim2Time = m_animation.runTime;

    // Update the manual blend state in AnimationManager
    getAnimationManager()->updateManualBlend(ratio, anim1Time, anim2Time);
}

void Character::setAnimationSpeed(float speed)
{
    m_animation.animationSpeed = std::max(0.1f, speed);

    if (hasAnimation() && getAnimationManager())
    {
        getAnimationManager()->setPlaybackSpeed(m_animation.animationSpeed);
    }
}

void Character::triggerPlaceAnimation()
{
    if (!hasAnimation() || !getAnimationManager())
        return;

    // Don't start a new place animation if one is already playing
    if (getAnimationManager()->hasMultipleAdditiveAnimation(m_animation.placeClipIndex))
        return;

    // Get place animation speed from global settings
    auto &globalAnimation = GlobalSettings::GetCharacterAnimationParams();
    float placeSpeed = globalAnimation.placeAnimationSpeed;

    // Start the additive place animation using multiple additive system
    getAnimationManager()->startMultipleAdditiveAnimation(m_animation.placeClipIndex, placeSpeed);
}