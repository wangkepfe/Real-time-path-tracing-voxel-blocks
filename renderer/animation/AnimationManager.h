#pragma once

#include "../shaders/LinearMath.h"
#include "Animation.h"
#include "Skeleton.h"
#include <vector>

// Forward declarations
struct AnimationClip;
struct AnimationSampler;


// Blend state for transitioning between animations
struct BlendState
{
    int fromClipIndex = -1;   // Source animation clip
    int toClipIndex = -1;     // Target animation clip
    float blendTime = 0.0f;   // Current blend time
    float blendDuration = 0.0f; // Total blend duration
    bool isBlending = false;  // Whether blending is active
};

// Additive animation layer
struct AdditiveAnimation
{
    int clipIndex = -1;      // Animation clip index
    float weight = 1.0f;     // Blend weight
    float currentTime = 0.0f; // Current playback time
};

// Animation manager class for handling skeletal animation
class AnimationManager
{
public:
    AnimationManager();
    ~AnimationManager();

    // Animation clip management
    int addAnimationClip(const AnimationClip &clip);
    void removeAnimationClip(int clipIndex);
    AnimationClip *getAnimationClip(int clipIndex);

    // Skeleton management
    void setSkeleton(const Skeleton &skeleton);
    Skeleton *getSkeleton() { return &m_skeleton; }

    // Animation playback
    void playAnimation(int clipIndex, bool loop = true, float blendTime = 0.0f);
    void stopAnimation();
    void pauseAnimation();
    void resumeAnimation();
    void setPlaybackSpeed(float speed);

    // Animation blending
    void blendToAnimation(int clipIndex, float blendDuration, bool loop = true);
    void addAdditiveAnimation(int clipIndex, float weight);
    void clearAdditiveAnimations();

    // Update animation (call every frame)
    void update(float deltaTime);

    // GPU data access
    float *getJointMatricesGPU() const { return (float *)m_skeleton.d_jointMatrices; }
    int getJointCount() const { return static_cast<int>(m_skeleton.joints.size()); }

private:
    // Core data
    std::vector<AnimationClip> m_animationClips;
    Skeleton m_skeleton;

    // Animation state
    AnimationState m_primaryAnimation;
    BlendState m_blendState;
    std::vector<AdditiveAnimation> m_additiveAnimations;

    // Internal methods
    void updateAnimation(AnimationState &state, const AnimationClip &clip, float deltaTime);
    void evaluateAnimation(const AnimationClip &clip, float time, std::vector<Joint> &joints);
    void blendAnimations();
    void applyAdditiveAnimations();

    // Interpolation methods
    Float3 interpolateTranslation(const AnimationSampler &sampler, float time);
    Float4 interpolateRotation(const AnimationSampler &sampler, float time);
    Float3 interpolateScale(const AnimationSampler &sampler, float time);

    // Utility methods
    void buildJointMatrices();
    int findKeyframeIndex(const std::vector<float> &times, float time);
    Float4 slerpQuaternions(const Float4 &q1, const Float4 &q2, float t);
    void multiplyMatrices(const float *a, const float *b, float *result);
};