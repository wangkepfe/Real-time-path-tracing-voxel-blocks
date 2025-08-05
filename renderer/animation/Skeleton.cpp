#include "Skeleton.h"
#include "../util/DebugUtils.h"
#include <cuda_runtime.h>
#include <iostream>

void Skeleton::updateJointTransforms()
{
    // Update transforms in hierarchical order
    for (size_t i = 0; i < joints.size(); ++i)
    {
        Joint &joint = joints[i];

        // Build local matrix from TRS
        joint.localMatrix = Mat4(joint.position, joint.rotation, joint.scale);
        

        // Calculate global matrix
        if (joint.parentIndex == -1)
        {
            joint.globalMatrix = joint.localMatrix;
        }
        else
        {
            const Joint &parent = joints[joint.parentIndex];
            joint.globalMatrix = parent.globalMatrix * joint.localMatrix;
        }
    }
}

void Skeleton::uploadToGPU()
{
    if (!d_jointMatrices)
    {
        CUDA_CHECK(cudaMalloc(&d_jointMatrices, MAX_JOINTS * sizeof(Mat4)));
    }

    // Prepare matrix data for upload
    Mat4 jointMatrices[MAX_JOINTS];

    // Zero out arrays
    for (int i = 0; i < MAX_JOINTS; i++)
    {
        jointMatrices[i] = Mat4(); // Initialize to identity
    }

    // Copy joint data
    for (size_t i = 0; i < joints.size() && i < MAX_JOINTS; ++i)
    {
        jointMatrices[i] = joints[i].globalMatrix * joints[i].inverseBindMatrix;
    }

    // Upload joint matrices to GPU
    CUDA_CHECK(cudaMemcpy(d_jointMatrices, jointMatrices, MAX_JOINTS * sizeof(Mat4), cudaMemcpyHostToDevice));

    // Synchronize after uploads
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Skeleton::cleanup()
{
    if (d_jointMatrices)
    {
        CUDA_CHECK(cudaFree(d_jointMatrices));
        d_jointMatrices = nullptr;
    }
}