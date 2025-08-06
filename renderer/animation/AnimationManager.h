#pragma once

#include "../shaders/LinearMath.h"
#include "Animation.h"
#include "Skeleton.h"
#include <vector>

// Forward declarations
struct AnimationClip;
struct AnimationSampler;

// Manual blend state for our two-animation system
struct ManualBlendState
{
    bool isActive = false;
    int anim1Index = -1;
    int anim2Index = -1;
    float blendRatio = 0.0f;   // 0.0 = full anim1, 1.0 = full anim2
    float anim1Time = 0.0f;
    float anim2Time = 0.0f;
};

// Additive animation state for place animation
struct AdditiveAnimationState
{
    bool isActive = false;
    int animationIndex = -1;
    float currentTime = 0.0f;
    float duration = 0.0f;
    float speed = 1.0f;
};

// Multiple additive animations support
struct MultipleAdditiveAnimations
{
    std::vector<AdditiveAnimationState> animations;
    
    void addAnimation(int animationIndex, float speed = 1.0f);
    void removeAnimation(int animationIndex);
    void clearAll();
    bool hasAnimation(int animationIndex) const;
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

    // Animation playback - simplified
    void setPlaybackSpeed(float speed);
    
    // New manual blending system
    void startManualBlend(int anim1Index, int anim2Index);
    void updateManualBlend(float ratio, float anim1Time, float anim2Time);
    void stopManualBlend();

    // Additive animation system for place animation (legacy single additive)
    void startAdditiveAnimation(int animationIndex, float speed = 1.0f);
    void stopAdditiveAnimation();
    bool isAdditiveAnimationActive() const { return m_additiveAnimation.isActive; }
    
    // Multiple additive animations system
    void startMultipleAdditiveAnimation(int animationIndex, float speed = 1.0f);
    void stopMultipleAdditiveAnimation(int animationIndex);
    void stopAllMultipleAdditiveAnimations();
    bool hasMultipleAdditiveAnimation(int animationIndex) const;

    // Update animation (call every frame)
    void update(float deltaTime);

    // GPU data access
    float *getJointMatricesGPU() const { return (float *)m_skeleton.d_jointMatrices; }
    int getJointCount() const { return static_cast<int>(m_skeleton.joints.size()); }

private:
    // Core data
    std::vector<AnimationClip> m_animationClips;
    Skeleton m_skeleton;

    // Animation state - simplified
    ManualBlendState m_manualBlend;
    AdditiveAnimationState m_additiveAnimation;
    MultipleAdditiveAnimations m_multipleAdditiveAnimations;
    float m_playbackSpeed = 1.0f;

    // Internal methods
    void evaluateAnimation(const AnimationClip &clip, float time, std::vector<Joint> &joints);
    void blendTwoEvaluatedAnimations(const std::vector<Joint> &joints1, const std::vector<Joint> &joints2, float ratio);
    void applyAdditiveAnimation(const AnimationClip &clip, float time, std::vector<Joint> &joints);
    void applyMultipleAdditiveAnimations(std::vector<Joint> &joints);

    // Interpolation methods
    Float3 interpolateTranslation(const AnimationSampler &sampler, float time);
    Float4 interpolateRotation(const AnimationSampler &sampler, float time);
    Float3 interpolateScale(const AnimationSampler &sampler, float time);

    // Utility methods
    int findKeyframeIndex(const std::vector<float> &times, float time);
};