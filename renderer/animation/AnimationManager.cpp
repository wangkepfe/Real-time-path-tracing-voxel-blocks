#include "AnimationManager.h"
#include "../util/DebugUtils.h"
#include "../shaders/LinearMath.h"  // For slerp function
#include <algorithm>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <iostream>

AnimationManager::AnimationManager()
{
    // Initialize manual blend state
    m_manualBlend.isActive = false;
    // Initialize additive animation state
    m_additiveAnimation.isActive = false;
    m_playbackSpeed = 1.0f;
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

void AnimationManager::setPlaybackSpeed(float speed)
{
    m_playbackSpeed = std::max(0.1f, speed);
}

void AnimationManager::startManualBlend(int anim1Index, int anim2Index)
{
    if (anim1Index < 0 || anim1Index >= static_cast<int>(m_animationClips.size()) ||
        anim2Index < 0 || anim2Index >= static_cast<int>(m_animationClips.size()))
        return;
        
    m_manualBlend.isActive = true;
    m_manualBlend.anim1Index = anim1Index;
    m_manualBlend.anim2Index = anim2Index;
    m_manualBlend.blendRatio = 0.0f;
    m_manualBlend.anim1Time = 0.0f;
    m_manualBlend.anim2Time = 0.0f;
}

void AnimationManager::updateManualBlend(float ratio, float anim1Time, float anim2Time)
{
    if (!m_manualBlend.isActive)
        return;
        
    m_manualBlend.blendRatio = std::max(0.0f, std::min(1.0f, ratio));
    m_manualBlend.anim1Time = anim1Time;
    m_manualBlend.anim2Time = anim2Time;
}

void AnimationManager::stopManualBlend()
{
    m_manualBlend.isActive = false;
}

void AnimationManager::startAdditiveAnimation(int animationIndex, float speed)
{
    if (animationIndex < 0 || animationIndex >= static_cast<int>(m_animationClips.size()))
        return;
        
    m_additiveAnimation.isActive = true;
    m_additiveAnimation.animationIndex = animationIndex;
    m_additiveAnimation.currentTime = 0.0f;
    m_additiveAnimation.duration = m_animationClips[animationIndex].duration;
    m_additiveAnimation.speed = speed;
}

void AnimationManager::stopAdditiveAnimation()
{
    m_additiveAnimation.isActive = false;
}

void AnimationManager::update(float deltaTime)
{
    if (m_skeleton.joints.empty())
        return;
    
    bool needsUpdate = false;
        
    // Handle manual blending (base animation)
    if (m_manualBlend.isActive)
    {
        // Get the animation clips
        AnimationClip *anim1 = getAnimationClip(m_manualBlend.anim1Index);
        AnimationClip *anim2 = getAnimationClip(m_manualBlend.anim2Index);
        
        if (anim1 && anim2)
        {
            
            // Evaluate both animations at their specified times
            std::vector<Joint> joints1 = m_skeleton.joints;
            std::vector<Joint> joints2 = m_skeleton.joints;
            
            evaluateAnimation(*anim1, m_manualBlend.anim1Time, joints1);
            evaluateAnimation(*anim2, m_manualBlend.anim2Time, joints2);
            
            // Blend the results into skeleton
            blendTwoEvaluatedAnimations(joints1, joints2, m_manualBlend.blendRatio);
            needsUpdate = true;
        }
    }
    
    // Handle additive animation (place animation)
    if (m_additiveAnimation.isActive)
    {
        // Update additive animation time
        m_additiveAnimation.currentTime += deltaTime * m_additiveAnimation.speed;
        
        // Check if animation is complete
        if (m_additiveAnimation.currentTime >= m_additiveAnimation.duration)
        {
            m_additiveAnimation.isActive = false;
        }
        else
        {
            // Apply additive animation on top of current skeleton state
            AnimationClip *additiveClip = getAnimationClip(m_additiveAnimation.animationIndex);
            if (additiveClip)
            {
                applyAdditiveAnimation(*additiveClip, m_additiveAnimation.currentTime, m_skeleton.joints);
                needsUpdate = true;
            }
        }
    }
    
    // Handle multiple additive animations (including sneak animation)
    if (!m_multipleAdditiveAnimations.animations.empty())
    {
        // Update all active multiple additive animations
        for (auto it = m_multipleAdditiveAnimations.animations.begin(); it != m_multipleAdditiveAnimations.animations.end();)
        {
            if (it->isActive)
            {
                // Update animation time
                it->currentTime += deltaTime * it->speed;
                
                // For looping animations (like sneak), wrap around
                AnimationClip *clip = getAnimationClip(it->animationIndex);
                if (clip && it->currentTime >= it->duration)
                {
                    if (clip->isLooping)
                    {
                        it->currentTime = fmod(it->currentTime, it->duration);
                    }
                    else
                    {
                        // Non-looping animation finished - remove it completely
                        it = m_multipleAdditiveAnimations.animations.erase(it);
                        needsUpdate = true;
                        continue;
                    }
                }
                
                needsUpdate = true;
                ++it;
            }
            else
            {
                // Remove inactive animations
                it = m_multipleAdditiveAnimations.animations.erase(it);
            }
        }
        
        // Apply all active multiple additive animations
        if (!m_multipleAdditiveAnimations.animations.empty())
        {
            applyMultipleAdditiveAnimations(m_skeleton.joints);
        }
    }
    
    // Update joint transforms and upload to GPU if needed
    if (needsUpdate)
    {
        m_skeleton.updateJointTransforms();
        m_skeleton.uploadToGPU();
    }
}

void AnimationManager::blendTwoEvaluatedAnimations(const std::vector<Joint> &joints1, const std::vector<Joint> &joints2, float ratio)
{
    // Blend the two animation results
    for (size_t i = 0; i < m_skeleton.joints.size(); ++i)
    {
        Joint &joint = m_skeleton.joints[i];
        const Joint &joint1 = joints1[i];
        const Joint &joint2 = joints2[i];
        
        // Linear interpolation for position and scale
        joint.position = joint1.position * (1.0f - ratio) + joint2.position * ratio;
        joint.scale = joint1.scale * (1.0f - ratio) + joint2.scale * ratio;
        
        
        // Spherical interpolation for rotation
        joint.rotation = slerp(joint1.rotation, joint2.rotation, ratio);
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
                    
                    Float3 translation = interpolateTranslation(sampler, time);
                    joint.position = translation;
                    
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


Float3 AnimationManager::interpolateTranslation(const AnimationSampler &sampler, float time)
{
    if (sampler.inputTimes.empty() || sampler.outputTranslations.empty()) {
        std::cout << "interpolateTranslation: empty data - inputTimes: " << sampler.inputTimes.size() 
                  << ", outputTranslations: " << sampler.outputTranslations.size() << std::endl;
        return Float3(0.0f);
    }

    if (time <= sampler.inputTimes[0])
    {
        // For CUBICSPLINE, return the value part of first keyframe
        if (sampler.interpolation == AnimationSampler::CUBICSPLINE)
            return sampler.outputTranslations[1]; // Index 1 is the value (in_tangent=0, value=1, out_tangent=2)
        else
            return sampler.outputTranslations[0];
    }

    if (time >= sampler.inputTimes.back())
    {
        // For CUBICSPLINE, return the value part of last keyframe
        if (sampler.interpolation == AnimationSampler::CUBICSPLINE)
        {
            int lastIndex = static_cast<int>(sampler.inputTimes.size()) - 1;
            return sampler.outputTranslations[lastIndex * 3 + 1]; // Last value
        }
        else
            return sampler.outputTranslations.back();
    }

    int keyIndex = findKeyframeIndex(sampler.inputTimes, time);

    if (keyIndex < 0 || keyIndex >= static_cast<int>(sampler.inputTimes.size() - 1))
    {
        if (sampler.interpolation == AnimationSampler::CUBICSPLINE)
            return sampler.outputTranslations[1]; // First value
        else
            return sampler.outputTranslations[0];
    }

    float t0 = sampler.inputTimes[keyIndex];
    float t1 = sampler.inputTimes[keyIndex + 1];
    float factor = (time - t0) / (t1 - t0);

    switch (sampler.interpolation)
    {
    case AnimationSampler::STEP:
        return sampler.outputTranslations[keyIndex];
    case AnimationSampler::LINEAR:
    {
        const Float3 &v0 = sampler.outputTranslations[keyIndex];
        const Float3 &v1 = sampler.outputTranslations[keyIndex + 1];
        Float3 result = v0 * (1.0f - factor) + v1 * factor;
        
        
        return result;
    }
    case AnimationSampler::CUBICSPLINE:
    {
        // GLTF CUBICSPLINE interpolation using Hermite splines
        // Data format: [in_tangent, value, out_tangent] for each keyframe
        // So for keyframe i: data[i*3] = in_tangent, data[i*3+1] = value, data[i*3+2] = out_tangent
        
        if (sampler.outputTranslations.size() < (keyIndex + 1) * 3 + 2)
            return sampler.outputTranslations[keyIndex * 3 + 1]; // Fallback to current value
            
        const Float3 &p0 = sampler.outputTranslations[keyIndex * 3 + 1];     // Current value
        const Float3 &m0 = sampler.outputTranslations[keyIndex * 3 + 2];     // Current out-tangent
        const Float3 &p1 = sampler.outputTranslations[(keyIndex + 1) * 3 + 1]; // Next value
        const Float3 &m1 = sampler.outputTranslations[(keyIndex + 1) * 3];     // Next in-tangent
        
        float dt = t1 - t0;
        
        // Hermite basis functions
        float t2 = factor * factor;
        float t3 = t2 * factor;
        
        float h00 = 2 * t3 - 3 * t2 + 1;  // (2t³ - 3t² + 1)
        float h10 = t3 - 2 * t2 + factor; // (t³ - 2t² + t)
        float h01 = -2 * t3 + 3 * t2;     // (-2t³ + 3t²)
        float h11 = t3 - t2;              // (t³ - t²)
        
        // Hermite interpolation: p(t) = h00*p0 + h10*dt*m0 + h01*p1 + h11*dt*m1
        return p0 * h00 + m0 * (h10 * dt) + p1 * h01 + m1 * (h11 * dt);
    }
    default:
        if (sampler.interpolation == AnimationSampler::CUBICSPLINE)
            return sampler.outputTranslations[keyIndex * 3 + 1];
        else
            return sampler.outputTranslations[keyIndex];
    }
}

Float4 AnimationManager::interpolateRotation(const AnimationSampler &sampler, float time)
{
    if (sampler.inputTimes.empty() || sampler.outputRotations.empty())
        return Float4(0.0f, 0.0f, 0.0f, 1.0f);

    if (time <= sampler.inputTimes[0])
    {
        // For CUBICSPLINE, return the value part of first keyframe
        if (sampler.interpolation == AnimationSampler::CUBICSPLINE)
            return sampler.outputRotations[1]; // Index 1 is the value
        else
            return sampler.outputRotations[0];
    }

    if (time >= sampler.inputTimes.back())
    {
        // For CUBICSPLINE, return the value part of last keyframe
        if (sampler.interpolation == AnimationSampler::CUBICSPLINE)
        {
            int lastIndex = static_cast<int>(sampler.inputTimes.size()) - 1;
            return sampler.outputRotations[lastIndex * 3 + 1]; // Last value
        }
        else
            return sampler.outputRotations.back();
    }

    int keyIndex = findKeyframeIndex(sampler.inputTimes, time);

    if (keyIndex < 0 || keyIndex >= static_cast<int>(sampler.inputTimes.size() - 1))
    {
        if (sampler.interpolation == AnimationSampler::CUBICSPLINE)
            return sampler.outputRotations[1]; // First value
        else
            return sampler.outputRotations[0];
    }

    float t0 = sampler.inputTimes[keyIndex];
    float t1 = sampler.inputTimes[keyIndex + 1];
    float factor = (time - t0) / (t1 - t0);

    switch (sampler.interpolation)
    {
    case AnimationSampler::STEP:
        return sampler.outputRotations[keyIndex];
    case AnimationSampler::LINEAR:
    {
        const Float4 &q0 = sampler.outputRotations[keyIndex];
        const Float4 &q1 = sampler.outputRotations[keyIndex + 1];
        return slerp(q0, q1, factor);
    }
    case AnimationSampler::CUBICSPLINE:
    {
        // For quaternions in CUBICSPLINE, we still use SLERP between the value points
        // The tangent data for quaternions is more complex and often we just interpolate values
        // This is a simplified approach - for full CUBICSPLINE quaternion support, 
        // we'd need squad interpolation which is quite complex
        
        if (sampler.outputRotations.size() < (keyIndex + 1) * 3 + 2)
            return sampler.outputRotations[keyIndex * 3 + 1]; // Fallback to current value
            
        const Float4 &q0 = sampler.outputRotations[keyIndex * 3 + 1];     // Current value
        const Float4 &q1 = sampler.outputRotations[(keyIndex + 1) * 3 + 1]; // Next value
        
        // Use SLERP for now - could be enhanced with SQUAD for true cubic interpolation
        return slerp(q0, q1, factor);
    }
    default:
        if (sampler.interpolation == AnimationSampler::CUBICSPLINE)
            return sampler.outputRotations[keyIndex * 3 + 1];
        else
            return sampler.outputRotations[keyIndex];
    }
}

Float3 AnimationManager::interpolateScale(const AnimationSampler &sampler, float time)
{
    if (sampler.inputTimes.empty() || sampler.outputScales.empty())
        return Float3(1.0f);

    if (time <= sampler.inputTimes[0])
    {
        // For CUBICSPLINE, return the value part of first keyframe
        if (sampler.interpolation == AnimationSampler::CUBICSPLINE)
            return sampler.outputScales[1]; // Index 1 is the value (in_tangent=0, value=1, out_tangent=2)
        else
            return sampler.outputScales[0];
    }

    if (time >= sampler.inputTimes.back())
    {
        // For CUBICSPLINE, return the value part of last keyframe
        if (sampler.interpolation == AnimationSampler::CUBICSPLINE)
        {
            int lastIndex = static_cast<int>(sampler.inputTimes.size()) - 1;
            return sampler.outputScales[lastIndex * 3 + 1]; // Last value
        }
        else
            return sampler.outputScales.back();
    }

    int keyIndex = findKeyframeIndex(sampler.inputTimes, time);

    if (keyIndex < 0 || keyIndex >= static_cast<int>(sampler.inputTimes.size() - 1))
    {
        if (sampler.interpolation == AnimationSampler::CUBICSPLINE)
            return sampler.outputScales[1]; // First value
        else
            return sampler.outputScales[0];
    }

    float t0 = sampler.inputTimes[keyIndex];
    float t1 = sampler.inputTimes[keyIndex + 1];
    float factor = (time - t0) / (t1 - t0);

    switch (sampler.interpolation)
    {
    case AnimationSampler::STEP:
        return sampler.outputScales[keyIndex];
    case AnimationSampler::LINEAR:
    {
        const Float3 &v0 = sampler.outputScales[keyIndex];
        const Float3 &v1 = sampler.outputScales[keyIndex + 1];
        return v0 * (1.0f - factor) + v1 * factor;
    }
    case AnimationSampler::CUBICSPLINE:
    {
        // GLTF CUBICSPLINE interpolation using Hermite splines
        // Data format: [in_tangent, value, out_tangent] for each keyframe
        // So for keyframe i: data[i*3] = in_tangent, data[i*3+1] = value, data[i*3+2] = out_tangent
        
        if (sampler.outputScales.size() < (keyIndex + 1) * 3 + 2)
            return sampler.outputScales[keyIndex * 3 + 1]; // Fallback to current value
            
        const Float3 &p0 = sampler.outputScales[keyIndex * 3 + 1];     // Current value
        const Float3 &m0 = sampler.outputScales[keyIndex * 3 + 2];     // Current out-tangent
        const Float3 &p1 = sampler.outputScales[(keyIndex + 1) * 3 + 1]; // Next value
        const Float3 &m1 = sampler.outputScales[(keyIndex + 1) * 3];     // Next in-tangent
        
        float dt = t1 - t0;
        
        // Hermite basis functions
        float t2 = factor * factor;
        float t3 = t2 * factor;
        
        float h00 = 2 * t3 - 3 * t2 + 1;  // (2t³ - 3t² + 1)
        float h10 = t3 - 2 * t2 + factor; // (t³ - 2t² + t)
        float h01 = -2 * t3 + 3 * t2;     // (-2t³ + 3t²)
        float h11 = t3 - t2;              // (t³ - t²)
        
        // Hermite interpolation: p(t) = h00*p0 + h10*dt*m0 + h01*p1 + h11*dt*m1
        return p0 * h00 + m0 * (h10 * dt) + p1 * h01 + m1 * (h11 * dt);
    }
    default:
        if (sampler.interpolation == AnimationSampler::CUBICSPLINE)
            return sampler.outputScales[keyIndex * 3 + 1];
        else
            return sampler.outputScales[keyIndex];
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

void AnimationManager::applyAdditiveAnimation(const AnimationClip &clip, float time, std::vector<Joint> &joints)
{
    // Apply additive animation - add rotations to current joint rotations
    for (const auto &channel : clip.channels)
    {
        if (channel.targetJoint >= 0 && channel.targetJoint < static_cast<int>(joints.size()) &&
            channel.samplerIndex >= 0 && channel.samplerIndex < static_cast<int>(clip.samplers.size()))
        {
            const AnimationSampler &sampler = clip.samplers[channel.samplerIndex];
            Joint &joint = joints[channel.targetJoint];

            switch (channel.targetPath)
            {
            case AnimationChannel::ROTATION:
            {
                // Get the additive rotation from the animation
                Float4 additiveRotation = interpolateRotation(sampler, time);
                
                // Apply additive rotation by multiplying quaternions
                // Result = currentRotation * additiveRotation
                joint.rotation = quaternionMultiply(joint.rotation, additiveRotation);
                break;
            }
            case AnimationChannel::TRANSLATION:
            {
                // For additive translation, add the translation offset
                Float3 additiveTranslation = interpolateTranslation(sampler, time);
                joint.position = joint.position + additiveTranslation;
                break;
            }
            case AnimationChannel::SCALE:
            {
                // For additive scale, multiply scales
                Float3 additiveScale = interpolateScale(sampler, time);
                joint.scale = joint.scale * additiveScale;
                break;
            }
            }
        }
    }
}

// MultipleAdditiveAnimations implementation
void MultipleAdditiveAnimations::addAnimation(int animationIndex, float speed)
{
    // Check if animation already exists
    for (auto &anim : animations)
    {
        if (anim.animationIndex == animationIndex)
        {
            // Update existing animation - restart it
            anim.speed = speed;
            anim.currentTime = 0.0f;
            anim.isActive = true;
            // Duration will be updated by AnimationManager::startMultipleAdditiveAnimation
            return;
        }
    }
    
    // Add new animation
    AdditiveAnimationState newAnim;
    newAnim.isActive = true;
    newAnim.animationIndex = animationIndex;
    newAnim.currentTime = 0.0f;
    newAnim.duration = 0.0f; // Will be set by AnimationManager
    newAnim.speed = speed;
    animations.push_back(newAnim);
}

void MultipleAdditiveAnimations::removeAnimation(int animationIndex)
{
    animations.erase(std::remove_if(animations.begin(), animations.end(),
        [animationIndex](const AdditiveAnimationState &anim) {
            return anim.animationIndex == animationIndex;
        }), animations.end());
}

void MultipleAdditiveAnimations::clearAll()
{
    animations.clear();
}

bool MultipleAdditiveAnimations::hasAnimation(int animationIndex) const
{
    for (const auto &anim : animations)
    {
        if (anim.animationIndex == animationIndex && anim.isActive)
        {
            return true;
        }
    }
    return false;
}

// AnimationManager multiple additive animations methods
void AnimationManager::startMultipleAdditiveAnimation(int animationIndex, float speed)
{
    if (animationIndex < 0 || animationIndex >= static_cast<int>(m_animationClips.size()))
        return;
        
    // Set duration for the animation
    AdditiveAnimationState newAnim;
    newAnim.duration = m_animationClips[animationIndex].duration;
    
    m_multipleAdditiveAnimations.addAnimation(animationIndex, speed);
    
    // Update duration in the added animation
    for (auto &anim : m_multipleAdditiveAnimations.animations)
    {
        if (anim.animationIndex == animationIndex)
        {
            anim.duration = newAnim.duration;
            break;
        }
    }
}

void AnimationManager::stopMultipleAdditiveAnimation(int animationIndex)
{
    m_multipleAdditiveAnimations.removeAnimation(animationIndex);
}

void AnimationManager::stopAllMultipleAdditiveAnimations()
{
    m_multipleAdditiveAnimations.clearAll();
}

bool AnimationManager::hasMultipleAdditiveAnimation(int animationIndex) const
{
    return m_multipleAdditiveAnimations.hasAnimation(animationIndex);
}

void AnimationManager::applyMultipleAdditiveAnimations(std::vector<Joint> &joints)
{
    for (auto &anim : m_multipleAdditiveAnimations.animations)
    {
        if (anim.isActive)
        {
            AnimationClip *additiveClip = getAnimationClip(anim.animationIndex);
            if (additiveClip)
            {
                applyAdditiveAnimation(*additiveClip, anim.currentTime, joints);
            }
        }
    }
}