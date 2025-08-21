#pragma once

#include "../shaders/LinearMath.h"
#include "../shaders/SystemParameter.h"
#include "../animation/Animation.h"
#include "../animation/AnimationManager.h"
#include "../animation/Skeleton.h"
#include <string>
#include <vector>
#include <memory>

// Entity type enumeration - separate from Block types
enum EntityType
{
    EntityTypeMinecraftCharacter,
    EntityTypeCount
};

// Entity rendering constants for OptiX integration
namespace EntityConstants
{
    // Instance ID offset for entities to avoid conflicts with block instance IDs
    static constexpr unsigned int ENTITY_INSTANCE_ID_OFFSET = 100000;

    // Material index offset for entities to enable future entity-specific materials
    static constexpr unsigned int ENTITY_MATERIAL_INDEX_OFFSET = 1000;
}

struct EntityTransform
{
    Float3 position = Float3(0.0f, 0.0f, 0.0f);
    Float3 rotation = Float3(0.0f, 0.0f, 0.0f); // Euler angles in radians
    Float3 scale = Float3(1.0f, 1.0f, 1.0f);

    // Get 4x3 transform matrix for OptiX
    void getTransformMatrix(float matrix[12]) const;
};

class Entity
{
public:
    Entity(EntityType type, const EntityTransform& transform);
    ~Entity();

    // Getters
    EntityType getType() const { return m_type; }
    const EntityTransform& getTransform() const { return m_transform; }
    VertexAttributes* getAttributes() const { return m_d_attributes; }
    unsigned int* getIndices() const { return m_d_indices; }
    unsigned int getAttributeSize() const { return m_attributeSize; }
    unsigned int getIndicesSize() const { return m_indicesSize; }

    // Setters
    void setTransform(const EntityTransform& transform) { m_transform = transform; }
    void setPosition(const Float3& position) { m_transform.position = position; }
    void setRotation(const Float3& rotation) { m_transform.rotation = rotation; }
    void setScale(const Float3& scale) { m_transform.scale = scale; }

    // Load geometry for this entity type
    bool loadGeometry();

    // Update entity (for animation, etc.)
    virtual void update(float deltaTime);

    // Animation support
    bool hasAnimation() const { return m_animationManager != nullptr; }
    AnimationManager* getAnimationManager() const { return m_animationManager.get(); }

private:
    EntityType m_type;
    EntityTransform m_transform;

    // GPU geometry data
    VertexAttributes* m_d_attributes = nullptr;           // Current (possibly skinned) vertices
    VertexAttributes* m_d_originalAttributes = nullptr;   // Original unskinned vertices for animation
    VertexSkinningData* m_d_skinningData = nullptr;       // Joint indices and weights for animation
    unsigned int* m_d_indices = nullptr;
    unsigned int m_attributeSize = 0;
    unsigned int m_indicesSize = 0;

    // Animation system
    std::unique_ptr<AnimationManager> m_animationManager;
    std::vector<AnimationClip> m_animationClips;
    Skeleton m_skeleton;

};