#include "AnimationManager.h"
#include "../util/DebugUtils.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <iostream>

AnimationManager::AnimationManager()
{
    // Initialize with identity matrices
    m_primaryAnimation.clipIndex = -1;
    m_primaryAnimation.isPlaying = false;
}

AnimationManager::~AnimationManager()
{
    m_skeleton.cleanup();
}

int AnimationManager::addAnimationClip(const AnimationClip &clip)
{
    m_animationClips.push_back(clip);
    return static_cast<int>(m_animationClips.size() - 1);
}

void AnimationManager::removeAnimationClip(int clipIndex)
{
    if (clipIndex >= 0 && clipIndex < static_cast<int>(m_animationClips.size()))
    {
        m_animationClips.erase(m_animationClips.begin() + clipIndex);
    }
}

AnimationClip *AnimationManager::getAnimationClip(int clipIndex)
{
    if (clipIndex >= 0 && clipIndex < static_cast<int>(m_animationClips.size()))
    {
        return &m_animationClips[clipIndex];
    }
    return nullptr;
}

void AnimationManager::setSkeleton(const Skeleton &skeleton)
{
    m_skeleton.cleanup();
    m_skeleton = skeleton;
    m_skeleton.updateJointTransforms(); // Initialize matrices first!
    m_skeleton.uploadToGPU();
}

void AnimationManager::playAnimation(int clipIndex, bool loop, float blendTime)
{
    if (clipIndex < 0 || clipIndex >= static_cast<int>(m_animationClips.size()))
        return;

    if (blendTime > 0.0f && m_primaryAnimation.isPlaying)
    {
        blendToAnimation(clipIndex, blendTime, loop);
    }
    else
    {
        m_primaryAnimation.clipIndex = clipIndex;
        m_primaryAnimation.currentTime = 0.0f;
        m_primaryAnimation.isPlaying = true;
        m_primaryAnimation.hasFinished = false;

        AnimationClip *clip = getAnimationClip(clipIndex);
        if (clip)
        {
            clip->isLooping = loop;
        }
    }
}

void AnimationManager::stopAnimation()
{
    m_primaryAnimation.isPlaying = false;
    m_primaryAnimation.currentTime = 0.0f;
    m_blendState.isBlending = false;
}

void AnimationManager::pauseAnimation()
{
    m_primaryAnimation.isPlaying = false;
}

void AnimationManager::resumeAnimation()
{
    if (m_primaryAnimation.clipIndex != -1)
    {
        m_primaryAnimation.isPlaying = true;
    }
}

void AnimationManager::setPlaybackSpeed(float speed)
{
    AnimationClip *clip = getAnimationClip(m_primaryAnimation.clipIndex);
    if (clip)
    {
        clip->playbackSpeed = speed;
    }
}

void AnimationManager::blendToAnimation(int clipIndex, float blendDuration, bool loop)
{
    if (clipIndex < 0 || clipIndex >= static_cast<int>(m_animationClips.size()))
        return;

    m_blendState.fromClipIndex = m_primaryAnimation.clipIndex;
    m_blendState.toClipIndex = clipIndex;
    m_blendState.blendTime = 0.0f;
    m_blendState.blendDuration = blendDuration;
    m_blendState.isBlending = true;

    // Set up target animation
    AnimationClip *targetClip = getAnimationClip(clipIndex);
    if (targetClip)
    {
        targetClip->isLooping = loop;
    }
}

void AnimationManager::addAdditiveAnimation(int clipIndex, float weight)
{
    if (clipIndex < 0 || clipIndex >= static_cast<int>(m_animationClips.size()))
        return;

    AdditiveAnimation additive;
    additive.clipIndex = clipIndex;
    additive.weight = weight;
    additive.currentTime = 0.0f;

    m_additiveAnimations.push_back(additive);
}

void AnimationManager::clearAdditiveAnimations()
{
    m_additiveAnimations.clear();
}

void AnimationManager::update(float deltaTime)
{
    static int animUpdateCount = 0;
    animUpdateCount++;

    if (m_skeleton.joints.empty())
    {
        if (animUpdateCount % 120 == 0)
        {
            std::cout << "ANIM_DEBUG: AnimationManager::update " << animUpdateCount
                      << " - NO SKELETON JOINTS!" << std::endl;
        }
        return;
    }

    // Update primary animation
    if (m_primaryAnimation.isPlaying && m_primaryAnimation.clipIndex != -1)
    {
        AnimationClip *clip = getAnimationClip(m_primaryAnimation.clipIndex);
        if (clip)
        {
            updateAnimation(m_primaryAnimation, *clip, deltaTime);
        }
    }

    // Update blend state
    if (m_blendState.isBlending)
    {
        m_blendState.blendTime += deltaTime;

        if (m_blendState.blendTime >= m_blendState.blendDuration)
        {
            // Blend complete - switch to target animation
            m_primaryAnimation.clipIndex = m_blendState.toClipIndex;
            m_blendState.isBlending = false;
        }
    }

    // Update additive animations
    for (auto &additive : m_additiveAnimations)
    {
        AnimationClip *clip = getAnimationClip(additive.clipIndex);
        if (clip)
        {
            additive.currentTime += deltaTime * clip->playbackSpeed;
            if (clip->isLooping && additive.currentTime > clip->duration)
            {
                additive.currentTime = fmod(additive.currentTime, clip->duration);
            }
        }
    }

    // Evaluate and blend all animations
    if (m_primaryAnimation.isPlaying || m_blendState.isBlending || !m_additiveAnimations.empty())
    {
        blendAnimations();
        m_skeleton.updateJointTransforms();
        m_skeleton.uploadToGPU();
    }
}

void AnimationManager::updateAnimation(AnimationState &state, const AnimationClip &clip, float deltaTime)
{
    state.currentTime += deltaTime * clip.playbackSpeed;

    if (state.currentTime > clip.duration)
    {
        if (clip.isLooping)
        {
            state.currentTime = fmod(state.currentTime, clip.duration);
        }
        else
        {
            state.currentTime = clip.duration;
            state.hasFinished = true;
            state.isPlaying = false;
        }
    }
}

void AnimationManager::evaluateAnimation(const AnimationClip &clip, float time, std::vector<Joint> &joints)
{
    // Reset all joints to bind pose first
    for (auto &joint : joints)
    {
        // Reset to stored bind pose
        joint.position = joint.bindPosition;
        joint.rotation = joint.bindRotation;
        joint.scale = joint.bindScale;
    }

    // Apply all animation channels
    for (const auto &channel : clip.channels)
    {
        if (channel.targetJoint >= 0 && channel.targetJoint < static_cast<int>(joints.size()) &&
            channel.samplerIndex >= 0 && channel.samplerIndex < static_cast<int>(clip.samplers.size()))
        {
            const AnimationSampler &sampler = clip.samplers[channel.samplerIndex];
            Joint &joint = joints[channel.targetJoint];

            switch (channel.targetPath)
            {
            case AnimationChannel::TRANSLATION:
                // Only apply translation to root joint (hip)
                // Other joints should maintain their bind pose positions
                if (joint.parentIndex == -1 || joint.name == "hip")
                {
                    joint.position = interpolateTranslation(sampler, time);
                }
                // For non-root joints, keep bind pose position
                break;
            case AnimationChannel::ROTATION:
                joint.rotation = interpolateRotation(sampler, time);
                break;
            case AnimationChannel::SCALE:
                joint.scale = interpolateScale(sampler, time);
                break;
            }
        }
    }
}

void AnimationManager::blendAnimations()
{
    if (m_blendState.isBlending)
    {
        // Evaluate both animations
        std::vector<Joint> fromJoints = m_skeleton.joints;
        std::vector<Joint> toJoints = m_skeleton.joints;

        if (m_blendState.fromClipIndex >= 0)
        {
            AnimationClip *fromClip = getAnimationClip(m_blendState.fromClipIndex);
            if (fromClip)
                evaluateAnimation(*fromClip, m_primaryAnimation.currentTime, fromJoints);
        }

        if (m_blendState.toClipIndex >= 0)
        {
            AnimationClip *toClip = getAnimationClip(m_blendState.toClipIndex);
            if (toClip)
                evaluateAnimation(*toClip, m_blendState.blendTime, toJoints);
        }

        // Blend between animations
        float blendFactor = m_blendState.blendTime / m_blendState.blendDuration;
        blendFactor = std::clamp(blendFactor, 0.0f, 1.0f);

        for (size_t i = 0; i < m_skeleton.joints.size(); ++i)
        {
            Joint &joint = m_skeleton.joints[i];
            const Joint &from = fromJoints[i];
            const Joint &to = toJoints[i];

            // Linear interpolation for position and scale
            joint.position = from.position * (1.0f - blendFactor) + to.position * blendFactor;
            joint.scale = from.scale * (1.0f - blendFactor) + to.scale * blendFactor;

            // Spherical interpolation for rotation
            joint.rotation = slerp(from.rotation, to.rotation, blendFactor);
        }
    }
    else if (m_primaryAnimation.isPlaying && m_primaryAnimation.clipIndex != -1)
    {
        // Just evaluate primary animation
        AnimationClip *clip = getAnimationClip(m_primaryAnimation.clipIndex);
        if (clip)
        {
            evaluateAnimation(*clip, m_primaryAnimation.currentTime, m_skeleton.joints);
        }
    }

    // Apply additive animations
    if (!m_additiveAnimations.empty())
    {
        applyAdditiveAnimations();
    }
}

void AnimationManager::applyAdditiveAnimations()
{
    for (const auto &additive : m_additiveAnimations)
    {
        AnimationClip *clip = getAnimationClip(additive.clipIndex);
        if (!clip)
            continue;

        std::vector<Joint> additiveJoints = m_skeleton.joints;
        evaluateAnimation(*clip, additive.currentTime, additiveJoints);

        // Apply additive transformation
        for (size_t i = 0; i < m_skeleton.joints.size(); ++i)
        {
            Joint &joint = m_skeleton.joints[i];
            const Joint &additiveJoint = additiveJoints[i];

            // Additive blending
            joint.position = joint.position + additiveJoint.position * additive.weight;
            joint.scale = joint.scale + (additiveJoint.scale - Float3(1.0f)) * additive.weight;

            // For rotation, use quaternion multiplication for additive
            Float4 additiveRot = slerp(
                Float4(0.0f, 0.0f, 0.0f, 1.0f), additiveJoint.rotation, additive.weight);
            joint.rotation = quaternionMultiply(joint.rotation, additiveRot);
        }
    }
}

Float3 AnimationManager::interpolateTranslation(const AnimationSampler &sampler, float time)
{
    if (sampler.inputTimes.empty() || sampler.outputTranslations.empty())
        return Float3(0.0f);

    if (time <= sampler.inputTimes[0])
        return sampler.outputTranslations[0];

    if (time >= sampler.inputTimes.back())
        return sampler.outputTranslations.back();

    int keyIndex = findKeyframeIndex(sampler.inputTimes, time);

    if (keyIndex < 0 || keyIndex >= static_cast<int>(sampler.inputTimes.size() - 1))
        return sampler.outputTranslations[0];

    float t0 = sampler.inputTimes[keyIndex];
    float t1 = sampler.inputTimes[keyIndex + 1];
    float factor = (time - t0) / (t1 - t0);

    const Float3 &v0 = sampler.outputTranslations[keyIndex];
    const Float3 &v1 = sampler.outputTranslations[keyIndex + 1];

    switch (sampler.interpolation)
    {
    case AnimationSampler::STEP:
        return v0;
    case AnimationSampler::LINEAR:
        return v0 * (1.0f - factor) + v1 * factor;
    case AnimationSampler::CUBICSPLINE:
        // TODO: Implement cubic spline interpolation
        return v0 * (1.0f - factor) + v1 * factor;
    default:
        return v0;
    }
}

Float4 AnimationManager::interpolateRotation(const AnimationSampler &sampler, float time)
{
    if (sampler.inputTimes.empty() || sampler.outputRotations.empty())
        return Float4(0.0f, 0.0f, 0.0f, 1.0f);

    if (time <= sampler.inputTimes[0])
        return sampler.outputRotations[0];

    if (time >= sampler.inputTimes.back())
        return sampler.outputRotations.back();

    int keyIndex = findKeyframeIndex(sampler.inputTimes, time);

    if (keyIndex < 0 || keyIndex >= static_cast<int>(sampler.inputTimes.size() - 1))
        return sampler.outputRotations[0];

    float t0 = sampler.inputTimes[keyIndex];
    float t1 = sampler.inputTimes[keyIndex + 1];
    float factor = (time - t0) / (t1 - t0);

    const Float4 &q0 = sampler.outputRotations[keyIndex];
    const Float4 &q1 = sampler.outputRotations[keyIndex + 1];

    switch (sampler.interpolation)
    {
    case AnimationSampler::STEP:
        return q0;
    case AnimationSampler::LINEAR:
    case AnimationSampler::CUBICSPLINE: // Fallback to linear for now
        return slerp(q0, q1, factor);
    default:
        return q0;
    }
}

Float3 AnimationManager::interpolateScale(const AnimationSampler &sampler, float time)
{
    // Same implementation as translation
    if (sampler.inputTimes.empty() || sampler.outputScales.empty())
        return Float3(1.0f);

    if (time <= sampler.inputTimes[0])
        return sampler.outputScales[0];

    if (time >= sampler.inputTimes.back())
        return sampler.outputScales.back();

    int keyIndex = findKeyframeIndex(sampler.inputTimes, time);

    if (keyIndex < 0 || keyIndex >= static_cast<int>(sampler.inputTimes.size() - 1))
        return sampler.outputScales[0];

    float t0 = sampler.inputTimes[keyIndex];
    float t1 = sampler.inputTimes[keyIndex + 1];
    float factor = (time - t0) / (t1 - t0);

    const Float3 &v0 = sampler.outputScales[keyIndex];
    const Float3 &v1 = sampler.outputScales[keyIndex + 1];

    switch (sampler.interpolation)
    {
    case AnimationSampler::STEP:
        return v0;
    case AnimationSampler::LINEAR:
        return v0 * (1.0f - factor) + v1 * factor;
    case AnimationSampler::CUBICSPLINE:
        // TODO: Implement cubic spline interpolation
        return v0 * (1.0f - factor) + v1 * factor;
    default:
        return v0;
    }
}

int AnimationManager::findKeyframeIndex(const std::vector<float> &times, float time)
{
    // Binary search for efficiency
    int left = 0;
    int right = static_cast<int>(times.size()) - 1;

    while (left <= right)
    {
        int mid = (left + right) / 2;

        if (times[mid] <= time && (mid == right || times[mid + 1] > time))
        {
            return mid;
        }
        else if (times[mid] < time)
        {
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }

    return std::max(0, right);
}