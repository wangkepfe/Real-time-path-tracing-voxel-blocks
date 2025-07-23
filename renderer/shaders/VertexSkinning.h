#pragma once

#include "LinearMath.h"
#include "SystemParameter.h"

// Maximum number of joints per skeleton (must match Animation.h)
#define MAX_SKINNING_JOINTS 128

// GPU vertex skinning data structure
struct SkinningData
{
    float* jointMatrices;        // Device pointer to joint matrices (MAX_SKINNING_JOINTS * 16)
    int numJoints;               // Number of active joints
    bool enabled;                // Whether skinning is enabled
};

// CUDA kernel for vertex skinning
__global__ void applyVertexSkinning(
    VertexAttributes* vertices,
    const VertexAttributes* originalVertices,
    const float* jointMatrices,
    int numVertices,
    int numJoints
);

// Host functions for managing skinning
void initVertexSkinning();
void cleanupVertexSkinning();
void updateSkinningMatrices(const float* hostMatrices, int numJoints);
void applySkinningToVertices(VertexAttributes* d_vertices, const VertexAttributes* d_originalVertices,
                           int numVertices, const SkinningData& skinningData);

// Utility functions
__device__ Float3 transformPoint(const Float3& point, const float* matrix);
__device__ void multiplyFloat4x4(const float* a, const float* b, float* result);
__device__ Float3 blendTransforms(const Float3& vertex, const Int4& joints, const Float4& weights,
                                  const float* jointMatrices, int numJoints);