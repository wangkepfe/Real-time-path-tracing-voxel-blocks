#include "Entity.h"
#include "../util/DebugUtils.h"
#include "../util/ModelUtils.h"
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

void EntityTransform::getTransformMatrix(float matrix[12]) const
{
    // Create rotation matrices
    float cosX = cos(rotation.x), sinX = sin(rotation.x);
    float cosY = cos(rotation.y), sinY = sin(rotation.y);
    float cosZ = cos(rotation.z), sinZ = sin(rotation.z);

    // Combined rotation matrix (ZYX order)
    float r00 = cosY * cosZ;
    float r01 = -cosY * sinZ;
    float r02 = sinY;
    float r10 = sinX * sinY * cosZ + cosX * sinZ;
    float r11 = -sinX * sinY * sinZ + cosX * cosZ;
    float r12 = -sinX * cosY;
    float r20 = -cosX * sinY * cosZ + sinX * sinZ;
    float r21 = cosX * sinY * sinZ + sinX * cosZ;
    float r22 = cosX * cosY;

    // Apply scale and translation
    matrix[0] = r00 * scale.x;
    matrix[1] = r01 * scale.y;
    matrix[2] = r02 * scale.z;
    matrix[3] = position.x;

    matrix[4] = r10 * scale.x;
    matrix[5] = r11 * scale.y;
    matrix[6] = r12 * scale.z;
    matrix[7] = position.y;

    matrix[8] = r20 * scale.x;
    matrix[9] = r21 * scale.y;
    matrix[10] = r22 * scale.z;
    matrix[11] = position.z;
}

Entity::Entity(EntityType type, const EntityTransform& transform)
    : m_type(type), m_transform(transform)
{
    loadGeometry();
}

Entity::~Entity()
{
    // Note: m_d_attributes and m_d_indices are owned and freed by OptixRenderer
    // through the m_geometries array. Do not free them here to avoid double-free.
    m_d_attributes = nullptr;
    m_d_indices = nullptr;
}

bool Entity::loadGeometry()
{
    switch (m_type)
    {
        case EntityTypeMinecraftCharacter:
            return loadMinecraftCharacterGeometry();
        default:
            std::cerr << "Unknown entity type: " << m_type << std::endl;
            return false;
    }
}

bool Entity::loadMinecraftCharacterGeometry()
{
    // Load Minecraft character from GLTF file
    const std::string gltfFile = "data/models/minecraft_char.gltf";

    // Use the regular model loading function from ModelUtils
    loadModel(&m_d_attributes, &m_d_indices, m_attributeSize, m_indicesSize, gltfFile);

    if (m_d_attributes && m_d_indices && m_attributeSize > 0 && m_indicesSize > 0) {
        return true;
    }

    return false;
}