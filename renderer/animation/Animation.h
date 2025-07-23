#pragma once

#include "shaders/LinearMath.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>

// Maximum number of joints supported per skeleton
#define MAX_JOINTS 128

// Joint representation for skeletal animation
struct Joint
{
    int parentIndex = -1;                    // Index of parent joint (-1 for root)
    Float3 position = Float3(0.0f);         // Local position relative to parent
    Float4 rotation = Float4(0.0f, 0.0f, 0.0f, 1.0f);  // Quaternion rotation
    Float3 scale = Float3(1.0f);             // Local scale

    // Computed transform matrices (updated during animation)
    float localMatrix[16];                   // Local transform matrix
    float globalMatrix[16];                  // World space transform matrix
    float inverseBindMatrix[16];             // Inverse bind pose matrix

    std::string name;                        // Joint name for debugging
};

// Animation keyframe for different interpolation types
template<typename T>
struct AnimationKeyframe
{
    float time;                              // Time in seconds
    T value;                                 // Keyframe value
};

// Animation sampler for different property types
struct AnimationSampler
{
    enum InterpolationType {
        STEP,
        LINEAR,
        CUBICSPLINE
    };

    InterpolationType interpolation = LINEAR;
    std::vector<float> inputTimes;           // Time values
    std::vector<Float3> outputTranslations; // Translation keyframes
    std::vector<Float4> outputRotations;    // Rotation keyframes (quaternions)
    std::vector<Float3> outputScales;       // Scale keyframes
};

// Animation channel that targets a specific joint property
struct AnimationChannel
{
    enum TargetPath {
        TRANSLATION,
        ROTATION,
        SCALE
    };

    int targetJoint;                         // Index of target joint
    TargetPath targetPath;                   // Which property to animate
    int samplerIndex;                        // Index into animation's samplers array
};

// Complete animation clip
struct AnimationClip
{
    std::string name;                        // Animation name
    float duration;                          // Animation duration in seconds
    std::vector<AnimationSampler> samplers;  // Animation samplers
    std::vector<AnimationChannel> channels;  // Animation channels

    bool isLooping = true;                   // Whether animation loops
    float playbackSpeed = 1.0f;              // Playback speed multiplier
};

// Skeletal structure for an animated model
struct Skeleton
{
    std::vector<Joint> joints;               // All joints in the skeleton
    std::unordered_map<std::string, int> jointNameToIndex; // Name to index mapping

    // GPU data (device pointers)
    float* d_jointMatrices = nullptr;        // Device joint matrices (MAX_JOINTS * 16 floats)
    float* d_inverseBindMatrices = nullptr;  // Device inverse bind matrices

    // Default constructor
    Skeleton() = default;

            // Copy constructor - Each skeleton manages its own GPU memory
    Skeleton(const Skeleton& other)
        : joints(other.joints), jointNameToIndex(other.jointNameToIndex) {
        // Each skeleton manages its own GPU memory - will be allocated in uploadToGPU()
        d_jointMatrices = nullptr;
        d_inverseBindMatrices = nullptr;
    }

    // Assignment operator - Each skeleton manages its own GPU resources
    Skeleton& operator=(const Skeleton& other) {
        if (this != &other) {
            // Clean up existing GPU resources
            cleanup();

            // Copy CPU data
            joints = other.joints;
            jointNameToIndex = other.jointNameToIndex;

            // Reset GPU pointers - each skeleton manages its own GPU memory
            d_jointMatrices = nullptr;
            d_inverseBindMatrices = nullptr;
        }
        return *this;
    }

    // Destructor
    ~Skeleton() {
        cleanup();
    }

    // Update all joint transforms
    void updateJointTransforms();

    // Upload matrices to GPU
    void uploadToGPU();

    // Get device joint matrices pointer for vertex skinning
    float* getDeviceJointMatrices() const {
        return d_jointMatrices;
    }

    // Cleanup GPU resources
    void cleanup();
};

// Animation state for a playing animation
struct AnimationState
{
    int clipIndex = -1;                      // Index of current animation clip
    float currentTime = 0.0f;                // Current time in animation
    float weight = 1.0f;                     // Blend weight (for animation blending)
    bool isPlaying = false;                  // Whether animation is currently playing
    bool hasFinished = false;                // Whether animation has finished (for non-looping)
};

// High-performance animation manager with state-of-the-art features
class AnimationManager
{
public:
    AnimationManager();
    ~AnimationManager();

    // Animation management
    int addAnimationClip(const AnimationClip& clip);
    void removeAnimationClip(int clipIndex);
    AnimationClip* getAnimationClip(int clipIndex);

    // Skeleton management
    void setSkeleton(const Skeleton& skeleton);
    Skeleton* getSkeleton() { return &m_skeleton; }

    // Animation playback
    void playAnimation(int clipIndex, bool loop = true, float blendTime = 0.0f);
    void stopAnimation();
    void pauseAnimation();
    void resumeAnimation();
    void setPlaybackSpeed(float speed);

    // Animation blending (for smooth transitions)
    void blendToAnimation(int clipIndex, float blendDuration, bool loop = true);
    void addAdditiveAnimation(int clipIndex, float weight);
    void clearAdditiveAnimations();

    // Update animation (call every frame)
    void update(float deltaTime);

    // GPU data access
    float* getJointMatricesGPU() const { return m_skeleton.d_jointMatrices; }
    int getJointCount() const { return static_cast<int>(m_skeleton.joints.size()); }

    // Performance optimization settings
    void enableBoneCulling(bool enable) { m_boneCullingEnabled = enable; }
    void setLODDistances(float highLOD, float mediumLOD, float lowLOD);

private:
    // Core data
    std::vector<AnimationClip> m_animationClips;
    Skeleton m_skeleton;

    // Animation state
    std::vector<AnimationState> m_animationStates;  // For animation blending
    AnimationState m_primaryAnimation;              // Main animation

    // Blending state
    struct BlendState {
        int fromClipIndex = -1;
        int toClipIndex = -1;
        float blendTime = 0.0f;
        float blendDuration = 0.0f;
        bool isBlending = false;
    } m_blendState;

    // Additive animations for layering
    struct AdditiveAnimation {
        int clipIndex;
        float weight;
        float currentTime;
    };
    std::vector<AdditiveAnimation> m_additiveAnimations;

    // Performance optimization
    bool m_boneCullingEnabled = true;
    float m_highLODDistance = 10.0f;
    float m_mediumLODDistance = 50.0f;
    float m_lowLODDistance = 100.0f;

    // Internal methods
    void updateAnimation(AnimationState& state, const AnimationClip& clip, float deltaTime);
    void evaluateAnimation(const AnimationClip& clip, float time, std::vector<Joint>& joints);
    void blendAnimations();
    void applyAdditiveAnimations();

    // Interpolation methods
    Float3 interpolateTranslation(const AnimationSampler& sampler, float time);
    Float4 interpolateRotation(const AnimationSampler& sampler, float time);
    Float3 interpolateScale(const AnimationSampler& sampler, float time);

    // Utility methods
    void buildJointMatrices();
    int findKeyframeIndex(const std::vector<float>& times, float time);
    Float4 slerpQuaternions(const Float4& q1, const Float4& q2, float t);
    void multiplyMatrices(const float* a, const float* b, float* result);
    void matrixFromTRS(const Float3& translation, const Float4& rotation, const Float3& scale, float* matrix);
};

// Utility functions for matrix operations (optimized with SIMD when available)
namespace AnimationMath {
    void multiplyMatrix4x4(const float* a, const float* b, float* result);
    void transposeMatrix4x4(const float* input, float* output);
    void invertMatrix4x4(const float* input, float* output);
    Float4 quaternionSlerp(const Float4& q1, const Float4& q2, float t);
    Float4 quaternionMultiply(const Float4& q1, const Float4& q2);
    void matrixFromQuaternion(const Float4& q, float* matrix);
    Float4 matrixToQuaternion(const float* matrix);
}