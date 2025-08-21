#pragma once

#include "../shaders/LinearMath.h"
#include "../shaders/SystemParameter.h"

// Maximum number of joints per skeleton (must match Animation.h)
#define MAX_SKINNING_JOINTS 128

// GPU vertex skinning data structure
struct SkinningData
{
    Mat4* jointMatrices;        // Device pointer to joint matrices (MAX_SKINNING_JOINTS Mat4s)
    int numJoints;               // Number of active joints
    bool enabled;                // Whether skinning is enabled
};

// CUDA kernel for vertex skinning
__global__ void applyVertexSkinning(
    VertexAttributes* vertices,
    const VertexAttributes* originalVertices,
    const VertexSkinningData* skinningData,
    const Mat4* jointMatrices,
    int numVertices,
    int numJoints
);

// Host functions for managing skinning
void initVertexSkinning();
void cleanupVertexSkinning();
void updateSkinningMatrices(const float* hostMatrices, int numJoints);
void applySkinningToVertices(VertexAttributes* d_vertices, const VertexAttributes* d_originalVertices,
                           const VertexSkinningData* d_skinningData, int numVertices, const SkinningData& skinningData);

// Utility functions
__device__ Float3 blendTransforms(const Float3& vertex, const VertexSkinningData& skinningData,
                                  const Mat4* jointMatrices, int numJoints);