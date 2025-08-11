#include "Entity.h"
#include "../util/DebugUtils.h"
#include "../util/ModelUtils.h"
#include "../util/GLTFUtils.h"
#include "../animation/VertexSkinning.h"
#include "../animation/Animation.h"
#include "../animation/AnimationManager.h"
#include "../animation/Skeleton.h"
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <utility>
#include <iomanip>

// Helper function to calculate AABB for vertices influenced by a specific joint
std::pair<Float3, Float3> calculateJointAABB(const VertexAttributes *vertices, int numVertices, int jointIndex, float weightThreshold = 0.1f)
{
    float maxFloat = std::numeric_limits<float>::max();
    float minFloat = std::numeric_limits<float>::lowest();
    Float3 minBounds(maxFloat, maxFloat, maxFloat);
    Float3 maxBounds(minFloat, minFloat, minFloat);
    bool foundVertex = false;

    for (int i = 0; i < numVertices; ++i)
    {
        const VertexAttributes &vertex = vertices[i];

        // Check if this vertex is influenced by the target joint
        bool influenced = false;
        if ((vertex.jointIndices.x == jointIndex && vertex.jointWeights.x >= weightThreshold) ||
            (vertex.jointIndices.y == jointIndex && vertex.jointWeights.y >= weightThreshold) ||
            (vertex.jointIndices.z == jointIndex && vertex.jointWeights.z >= weightThreshold) ||
            (vertex.jointIndices.w == jointIndex && vertex.jointWeights.w >= weightThreshold))
        {
            influenced = true;
        }

        if (influenced)
        {
            foundVertex = true;
            const Float3 &pos = vertex.vertex;

            // Update min bounds
            minBounds.x = std::min(minBounds.x, pos.x);
            minBounds.y = std::min(minBounds.y, pos.y);
            minBounds.z = std::min(minBounds.z, pos.z);

            // Update max bounds
            maxBounds.x = std::max(maxBounds.x, pos.x);
            maxBounds.y = std::max(maxBounds.y, pos.y);
            maxBounds.z = std::max(maxBounds.z, pos.z);
        }
    }

    if (!foundVertex)
    {
        // Return default bounds if no vertices found
        return std::pair<Float3, Float3>(Float3(0.0f, 0.0f, 0.0f), Float3(0.0f, 0.0f, 0.0f));
    }

    return std::pair<Float3, Float3>(minBounds, maxBounds);
}

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

Entity::Entity(EntityType type, const EntityTransform &transform)
    : m_type(type), m_transform(transform)
{
    bool success = loadGeometry();
    if (!success)
    {
        std::cout << "Geometry loading FAILED" << std::endl;
    }
}

Entity::~Entity()
{
    // Note: m_d_attributes and m_d_indices are owned and freed by OptixRenderer
    // through the m_geometries array. Do not free them here to avoid double-free.

    // Free original vertices for animation (these are owned by Entity)
    if (m_d_originalAttributes)
    {
        CUDA_CHECK(cudaFree(m_d_originalAttributes));
        m_d_originalAttributes = nullptr;
    }

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

void Entity::update(float deltaTime)
{
    static int updateCount = 0;
    updateCount++;

    assert(m_animationManager != nullptr);

    // Update animation
    m_animationManager->update(deltaTime);

    // Apply vertex skinning if we have original vertices
    if (m_d_originalAttributes && m_d_attributes && m_attributeSize > 0)
    {
        // Validate pointers
        if (!m_d_originalAttributes || !m_d_attributes)
        {
            return;
        }

        // Get skinning data from animation manager's skeleton (which has the GPU memory)
        SkinningData skinningData;
        Skeleton *animSkeleton = m_animationManager->getSkeleton();
        if (!animSkeleton)
        {
            return;
        }

        skinningData.jointMatrices = (Mat4 *)m_animationManager->getJointMatricesGPU();
        skinningData.numJoints = m_animationManager->getJointCount();
        skinningData.enabled = true;

        // Validate joint matrix pointer
        if (!skinningData.jointMatrices)
        {
            return;
        }

        // Apply vertex skinning
        applySkinningToVertices(m_d_attributes, m_d_originalAttributes,
                                m_attributeSize, skinningData);
    }
}

bool Entity::loadMinecraftCharacterGeometry()
{
    // Load Minecraft 1.8+ character with tessellated mesh from GLTF file with animation support
    const std::string gltfFile = "data/models/character-pink-smoothie.gltf";

    std::cout << "Loading animated Minecraft character from: " << gltfFile << std::endl;

    // TODO: Integrate with ModelManager once animated entity support is added
    // For now, continue using GLTFUtils directly for animated entities
    if (GLTFUtils::loadAnimatedGLTFModel(&m_d_attributes, &m_d_indices, m_attributeSize, m_indicesSize,
                                         m_skeleton, m_animationClips, gltfFile))
    {
        std::cout << "Successfully loaded animated character with " << m_animationClips.size() << " animations" << std::endl;

        // Allocate and copy original vertices for animation skinning
        const size_t vertexBufferSize = m_attributeSize * sizeof(VertexAttributes);
        CUDA_CHECK(cudaMalloc((void **)&m_d_originalAttributes, vertexBufferSize));
        CUDA_CHECK(cudaMemcpy(m_d_originalAttributes, m_d_attributes, vertexBufferSize, cudaMemcpyDeviceToDevice));

        // Create and initialize animation manager
        m_animationManager = std::make_unique<AnimationManager>();
        m_animationManager->setSkeleton(m_skeleton);

        // Add all animation clips to the manager
        for (const auto &clip : m_animationClips)
        {
            m_animationManager->addAnimationClip(clip);
        }

        return true;
    }
    else
    {
        std::cerr << "Failed to load animated GLTF model, falling back to static model" << std::endl;

        // Fallback to regular model loading
        loadModel(&m_d_attributes, &m_d_indices, m_attributeSize, m_indicesSize, gltfFile);

        if (m_d_attributes && m_d_indices && m_attributeSize > 0 && m_indicesSize > 0)
        {
            return true;
        }
    }

    return false;
}