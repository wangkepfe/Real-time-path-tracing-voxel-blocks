#include "Entity.h"
#include "../util/DebugUtils.h"
#include "../assets/ModelManager.h"
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
    // Note: m_d_attributes are owned and freed by ModelManager through the m_geometries array.
    // Do not free them here to avoid double-free.

    // Free indices (owned by Entity after conversion from Int3 to unsigned int)
    if (m_d_indices)
    {
        cudaError_t err = cudaFree(m_d_indices);
        if (err != cudaSuccess && err != cudaErrorInvalidValue)
        {
            CUDA_CHECK(err);
        }
        m_d_indices = nullptr;
    }

    // Free original vertices for animation (these are owned by Entity)
    if (m_d_originalAttributes)
    {
        CUDA_CHECK(cudaFree(m_d_originalAttributes));
        m_d_originalAttributes = nullptr;
    }

    // Free skinning data for animation (these are owned by Entity)
    if (m_d_skinningData)
    {
        CUDA_CHECK(cudaFree(m_d_skinningData));
        m_d_skinningData = nullptr;
    }

    m_d_attributes = nullptr;
}

bool Entity::loadGeometry()
{
    // Use ModelManager to get geometry for this entity type
    auto &modelManager = Assets::ModelManager::Get();
    const Assets::LoadedGeometry *geometry = modelManager.getGeometryForEntity(m_type);

    if (!geometry)
    {
        std::cerr << "No geometry found for entity type: " << m_type << std::endl;
        return false;
    }

    // Copy geometry pointers (ModelManager owns the data)
    m_d_attributes = geometry->d_attributes;
    m_attributeSize = geometry->attributeSize;

    // Convert Int3 indices to unsigned int format for Entity interface compatibility
    m_indicesSize = geometry->indicesSize * 3; // Int3 count * 3 = unsigned int count
    size_t indicesBufferSize = m_indicesSize * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc((void **)&m_d_indices, indicesBufferSize));

    // Convert Int3* to unsigned int* on GPU
    std::vector<unsigned int> tempIndices(m_indicesSize);
    std::vector<Int3> int3Indices(geometry->indicesSize);

    // Copy Int3 data from GPU to host
    CUDA_CHECK(cudaMemcpy(int3Indices.data(), geometry->d_indices, geometry->indicesSize * sizeof(Int3), cudaMemcpyDeviceToHost));

    // Convert Int3 to flat unsigned int array
    for (size_t i = 0; i < geometry->indicesSize; ++i)
    {
        tempIndices[i * 3 + 0] = int3Indices[i].x;
        tempIndices[i * 3 + 1] = int3Indices[i].y;
        tempIndices[i * 3 + 2] = int3Indices[i].z;
    }

    // Copy converted indices back to GPU
    CUDA_CHECK(cudaMemcpy(m_d_indices, tempIndices.data(), indicesBufferSize, cudaMemcpyHostToDevice));

    // Handle animation if present
    if (geometry->hasAnimation)
    {
        m_isAnimated = true;
        std::cout << "Setting up animation for entity with " << geometry->animationClips.size() << " animations" << std::endl;

        // Copy skeleton and animation clips
        m_skeleton = geometry->skeleton;
        m_animationClips = geometry->animationClips;

        // Allocate and copy original vertices for animation skinning
        const size_t vertexBufferSize = m_attributeSize * sizeof(VertexAttributes);
        CUDA_CHECK(cudaMalloc((void **)&m_d_originalAttributes, vertexBufferSize));
        CUDA_CHECK(cudaMemcpy(m_d_originalAttributes, m_d_attributes, vertexBufferSize, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMalloc((void **)&m_d_prevAttributes, vertexBufferSize));
        CUDA_CHECK(cudaMemcpy(m_d_prevAttributes, m_d_attributes, vertexBufferSize, cudaMemcpyDeviceToDevice));

        // Copy skinning data from the geometry if available
        if (geometry->d_skinningData)
        {
            const size_t skinningBufferSize = m_attributeSize * sizeof(VertexSkinningData);
            CUDA_CHECK(cudaMalloc((void **)&m_d_skinningData, skinningBufferSize));
            CUDA_CHECK(cudaMemcpy(m_d_skinningData, geometry->d_skinningData, skinningBufferSize, cudaMemcpyDeviceToDevice));
        }
        else
        {
            // Create empty skinning data array for non-animated models
            const size_t skinningBufferSize = m_attributeSize * sizeof(VertexSkinningData);
            CUDA_CHECK(cudaMalloc((void **)&m_d_skinningData, skinningBufferSize));
            
            // Initialize with default values (no skinning)
            std::vector<VertexSkinningData> tempSkinningData(m_attributeSize);
            for (unsigned int i = 0; i < m_attributeSize; ++i)
            {
                tempSkinningData[i].jointIndices = Int4(0, 0, 0, 0);
                tempSkinningData[i].jointWeights = Float4(1.0f, 0.0f, 0.0f, 0.0f);
            }
            CUDA_CHECK(cudaMemcpy(m_d_skinningData, tempSkinningData.data(), skinningBufferSize, cudaMemcpyHostToDevice));
        }

        // Create and initialize animation manager
        m_animationManager = std::make_unique<AnimationManager>();
        m_animationManager->setSkeleton(m_skeleton);

        // Add all animation clips to the manager
        for (const auto &clip : m_animationClips)
        {
            m_animationManager->addAnimationClip(clip);
        }

        std::cout << "Successfully set up animated entity with " << m_animationClips.size() << " animations" << std::endl;
    }
    else
    {
        m_isAnimated = false;
        m_d_prevAttributes = nullptr;
    }

    return true;
}

void Entity::update(float deltaTime)
{
    if (!hasAnimation())
    {
        return;
    }

    static int updateCount = 0;
    updateCount++;

    // Update animation
    m_animationManager->update(deltaTime);

    // Apply vertex skinning if we have original vertices
    if (m_d_originalAttributes && m_d_attributes && m_d_prevAttributes && m_attributeSize > 0)
    {
        const size_t vertexBufferSize = m_attributeSize * sizeof(VertexAttributes);
        CUDA_CHECK(cudaMemcpy(m_d_prevAttributes, m_d_attributes, vertexBufferSize, cudaMemcpyDeviceToDevice));

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
                                m_d_skinningData, m_attributeSize, skinningData);
    }
}
