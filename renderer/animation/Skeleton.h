#pragma once

#include "../shaders/LinearMath.h"
#include "Animation.h"
#include <vector>
#include <string>
#include <unordered_map>

// Maximum number of joints supported per skeleton
#define MAX_JOINTS 128

// Forward declaration
struct Joint;

// Skeletal structure for an animated model
struct Skeleton
{
    std::vector<Joint> joints;                             // All joints in the skeleton
    std::unordered_map<std::string, int> jointNameToIndex; // Name to index mapping

    // GPU data (device pointers)
    Mat4 *d_jointMatrices = nullptr; // Device joint matrices (MAX_JOINTS Mat4s)

    // Default constructor
    Skeleton() = default;

    // Copy constructor - Each skeleton manages its own GPU memory
    Skeleton(const Skeleton &other)
        : joints(other.joints), jointNameToIndex(other.jointNameToIndex)
    {
        d_jointMatrices = nullptr;
    }

    // Move constructor
    Skeleton(Skeleton &&other) noexcept
        : joints(std::move(other.joints)), jointNameToIndex(std::move(other.jointNameToIndex)),
          d_jointMatrices(other.d_jointMatrices)
    {
        other.d_jointMatrices = nullptr;
    }

    // Copy assignment operator
    Skeleton &operator=(const Skeleton &other)
    {
        if (this != &other)
        {
            cleanup(); // Clean up existing GPU resources
            joints = other.joints;
            jointNameToIndex = other.jointNameToIndex;
            d_jointMatrices = nullptr;
        }
        return *this;
    }

    // Move assignment operator
    Skeleton &operator=(Skeleton &&other) noexcept
    {
        if (this != &other)
        {
            cleanup(); // Clean up existing GPU resources
            joints = std::move(other.joints);
            jointNameToIndex = std::move(other.jointNameToIndex);
            d_jointMatrices = other.d_jointMatrices;
            other.d_jointMatrices = nullptr;
        }
        return *this;
    }

    // Destructor
    ~Skeleton()
    {
        cleanup();
    }

    // Add a joint to the skeleton
    int addJoint(const std::string &name, int parentIndex = -1)
    {
        Joint joint;
        joint.parentIndex = parentIndex;
        joint.name = name;
        
        int index = static_cast<int>(joints.size());
        joints.push_back(joint);
        jointNameToIndex[name] = index;
        
        return index;
    }

    // Find joint index by name
    int findJoint(const std::string &name) const
    {
        auto it = jointNameToIndex.find(name);
        return (it != jointNameToIndex.end()) ? it->second : -1;
    }

    // Update joint transforms in hierarchical order
    void updateJointTransforms();

    // Upload matrices to GPU
    void uploadToGPU();

    // Get device joint matrices pointer for vertex skinning
    float *getDeviceJointMatrices() const
    {
        return (float *)d_jointMatrices;
    }

    // Cleanup GPU resources
    void cleanup();
};