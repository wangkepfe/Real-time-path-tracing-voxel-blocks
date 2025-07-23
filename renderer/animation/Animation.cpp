#include "Animation.h"
#include "util/DebugUtils.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <immintrin.h>  // For SIMD intrinsics
#include <cuda_runtime.h>

// Helper math functions for animation
namespace AnimationMath {
    // Build 4x4 matrix from translation, rotation (quaternion), and scale
    void matrixFromTRS(const float translation[3], const float rotation[4], const float scale[3], float matrix[16])
    {
        // Extract quaternion components
        float x = rotation[0], y = rotation[1], z = rotation[2], w = rotation[3];

        // Calculate rotation matrix elements
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, xz = x * z, yz = y * z;
        float wx = w * x, wy = w * y, wz = w * z;

        // Build matrix with rotation and scale
        matrix[0]  = scale[0] * (1.0f - 2.0f * (yy + zz));
        matrix[1]  = scale[0] * (2.0f * (xy + wz));
        matrix[2]  = scale[0] * (2.0f * (xz - wy));
        matrix[3]  = 0.0f;

        matrix[4]  = scale[1] * (2.0f * (xy - wz));
        matrix[5]  = scale[1] * (1.0f - 2.0f * (xx + zz));
        matrix[6]  = scale[1] * (2.0f * (yz + wx));
        matrix[7]  = 0.0f;

        matrix[8]  = scale[2] * (2.0f * (xz + wy));
        matrix[9]  = scale[2] * (2.0f * (yz - wx));
        matrix[10] = scale[2] * (1.0f - 2.0f * (xx + yy));
        matrix[11] = 0.0f;

        // Translation
        matrix[12] = translation[0];
        matrix[13] = translation[1];
        matrix[14] = translation[2];
        matrix[15] = 1.0f;
    }

    // Multiply two 4x4 matrices: result = a * b
    void multiplyMatrix4x4(const float a[16], const float b[16], float result[16])
    {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result[i * 4 + j] = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    result[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
                }
            }
        }
    }
}

// Wrapper function for backward compatibility
void matrixFromTRS(const float translation[3], const float rotation[4], const float scale[3], float matrix[16])
{
    AnimationMath::matrixFromTRS(translation, rotation, scale, matrix);
}

// ===== Skeleton Implementation =====

void Skeleton::updateJointTransforms()
{
    // Update transforms in hierarchical order
    for (size_t i = 0; i < joints.size(); ++i)
    {
        Joint& joint = joints[i];

        // Build local matrix from TRS
        matrixFromTRS(&joint.position.x, &joint.rotation.x, &joint.scale.x, joint.localMatrix);

        // Calculate global matrix
        if (joint.parentIndex == -1)
        {
            // Root joint - global = local
            memcpy(joint.globalMatrix, joint.localMatrix, sizeof(float) * 16);
        }
        else
        {
            // Child joint - global = parent_global * local
            const Joint& parent = joints[joint.parentIndex];
            AnimationMath::multiplyMatrix4x4(parent.globalMatrix, joint.localMatrix, joint.globalMatrix);
        }
    }
}

void Skeleton::uploadToGPU()
{
    if (!d_jointMatrices)
    {
        CUDA_CHECK(cudaMalloc(&d_jointMatrices, MAX_JOINTS * 16 * sizeof(float)));
    }

    if (!d_inverseBindMatrices)
    {
        CUDA_CHECK(cudaMalloc(&d_inverseBindMatrices, MAX_JOINTS * 16 * sizeof(float)));
    }

    // Prepare matrix data for upload
    float jointMatrices[MAX_JOINTS * 16];
    float inverseBindMatrices[MAX_JOINTS * 16];

    // Zero out arrays
    memset(jointMatrices, 0, sizeof(jointMatrices));
    memset(inverseBindMatrices, 0, sizeof(inverseBindMatrices));

    // Copy joint data
    for (size_t i = 0; i < joints.size() && i < MAX_JOINTS; ++i)
    {
        // Calculate final skinning matrix: joint_global * inverse_bind
        float skinningMatrix[16];
        AnimationMath::multiplyMatrix4x4(joints[i].globalMatrix, joints[i].inverseBindMatrix, skinningMatrix);

        memcpy(&jointMatrices[i * 16], skinningMatrix, sizeof(float) * 16);
        memcpy(&inverseBindMatrices[i * 16], joints[i].inverseBindMatrix, sizeof(float) * 16);
    }

            // Upload to GPU
    if (!d_jointMatrices || !d_inverseBindMatrices) {
        return;
    }

    // Upload joint matrices to GPU
    CUDA_CHECK(cudaMemcpy(d_jointMatrices, jointMatrices, MAX_JOINTS * 16 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inverseBindMatrices, inverseBindMatrices, MAX_JOINTS * 16 * sizeof(float), cudaMemcpyHostToDevice));

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

    if (d_inverseBindMatrices)
    {
        CUDA_CHECK(cudaFree(d_inverseBindMatrices));
        d_inverseBindMatrices = nullptr;
    }
}

// ===== AnimationManager Implementation =====

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

int AnimationManager::addAnimationClip(const AnimationClip& clip)
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

AnimationClip* AnimationManager::getAnimationClip(int clipIndex)
{
    if (clipIndex >= 0 && clipIndex < static_cast<int>(m_animationClips.size()))
    {
        return &m_animationClips[clipIndex];
    }
    return nullptr;
}

void AnimationManager::setSkeleton(const Skeleton& skeleton)
{
    m_skeleton.cleanup();
    m_skeleton = skeleton;
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

        AnimationClip* clip = getAnimationClip(clipIndex);
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
    AnimationClip* clip = getAnimationClip(m_primaryAnimation.clipIndex);
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
    AnimationClip* targetClip = getAnimationClip(clipIndex);
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

        // Animation update system

    // Update primary animation
    if (m_primaryAnimation.isPlaying && m_primaryAnimation.clipIndex != -1)
    {
                AnimationClip* clip = getAnimationClip(m_primaryAnimation.clipIndex);
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
    for (auto& additive : m_additiveAnimations)
    {
        AnimationClip* clip = getAnimationClip(additive.clipIndex);
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

void AnimationManager::updateAnimation(AnimationState& state, const AnimationClip& clip, float deltaTime)
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

void AnimationManager::evaluateAnimation(const AnimationClip& clip, float time, std::vector<Joint>& joints)
{
    // Reset all joints to bind pose first
    for (auto& joint : joints)
    {
        // Extract bind pose position from inverse bind matrix (negate the translation)
        joint.position = Float3(-joint.inverseBindMatrix[12], -joint.inverseBindMatrix[13], -joint.inverseBindMatrix[14]);
        joint.rotation = Float4(0.0f, 0.0f, 0.0f, 1.0f);
        joint.scale = Float3(1.0f);
    }

    // Apply all animation channels
    for (const auto& channel : clip.channels)
    {
        if (channel.targetJoint >= 0 && channel.targetJoint < static_cast<int>(joints.size()) &&
            channel.samplerIndex >= 0 && channel.samplerIndex < static_cast<int>(clip.samplers.size()))
        {
            const AnimationSampler& sampler = clip.samplers[channel.samplerIndex];
            Joint& joint = joints[channel.targetJoint];

            switch (channel.targetPath)
            {
                case AnimationChannel::TRANSLATION:
                    joint.position = interpolateTranslation(sampler, time);
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
            AnimationClip* fromClip = getAnimationClip(m_blendState.fromClipIndex);
            if (fromClip)
                evaluateAnimation(*fromClip, m_primaryAnimation.currentTime, fromJoints);
        }

        if (m_blendState.toClipIndex >= 0)
        {
            AnimationClip* toClip = getAnimationClip(m_blendState.toClipIndex);
            if (toClip)
                evaluateAnimation(*toClip, m_blendState.blendTime, toJoints);
        }

        // Blend between animations
        float blendFactor = m_blendState.blendTime / m_blendState.blendDuration;
        blendFactor = std::clamp(blendFactor, 0.0f, 1.0f);

        for (size_t i = 0; i < m_skeleton.joints.size(); ++i)
        {
            Joint& joint = m_skeleton.joints[i];
            const Joint& from = fromJoints[i];
            const Joint& to = toJoints[i];

            // Linear interpolation for position and scale
            joint.position = from.position * (1.0f - blendFactor) + to.position * blendFactor;
            joint.scale = from.scale * (1.0f - blendFactor) + to.scale * blendFactor;

            // Spherical interpolation for rotation
            joint.rotation = AnimationMath::quaternionSlerp(from.rotation, to.rotation, blendFactor);
        }
    }
    else if (m_primaryAnimation.isPlaying && m_primaryAnimation.clipIndex != -1)
    {
        // Just evaluate primary animation
        AnimationClip* clip = getAnimationClip(m_primaryAnimation.clipIndex);
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
    for (const auto& additive : m_additiveAnimations)
    {
        AnimationClip* clip = getAnimationClip(additive.clipIndex);
        if (!clip) continue;

        std::vector<Joint> additiveJoints = m_skeleton.joints;
        evaluateAnimation(*clip, additive.currentTime, additiveJoints);

        // Apply additive transformation
        for (size_t i = 0; i < m_skeleton.joints.size(); ++i)
        {
            Joint& joint = m_skeleton.joints[i];
            const Joint& additiveJoint = additiveJoints[i];

            // Additive blending
            joint.position = joint.position + additiveJoint.position * additive.weight;
            joint.scale = joint.scale + (additiveJoint.scale - Float3(1.0f)) * additive.weight;

            // For rotation, use quaternion multiplication for additive
            Float4 additiveRot = AnimationMath::quaternionSlerp(
                Float4(0.0f, 0.0f, 0.0f, 1.0f), additiveJoint.rotation, additive.weight);
            joint.rotation = AnimationMath::quaternionMultiply(joint.rotation, additiveRot);
        }
    }
}

Float3 AnimationManager::interpolateTranslation(const AnimationSampler& sampler, float time)
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

    const Float3& v0 = sampler.outputTranslations[keyIndex];
    const Float3& v1 = sampler.outputTranslations[keyIndex + 1];

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

Float4 AnimationManager::interpolateRotation(const AnimationSampler& sampler, float time)
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

    const Float4& q0 = sampler.outputRotations[keyIndex];
    const Float4& q1 = sampler.outputRotations[keyIndex + 1];

    switch (sampler.interpolation)
    {
        case AnimationSampler::STEP:
            return q0;
        case AnimationSampler::LINEAR:
        case AnimationSampler::CUBICSPLINE:  // Fallback to linear for now
            return AnimationMath::quaternionSlerp(q0, q1, factor);
        default:
            return q0;
    }
}

Float3 AnimationManager::interpolateScale(const AnimationSampler& sampler, float time)
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

    const Float3& v0 = sampler.outputScales[keyIndex];
    const Float3& v1 = sampler.outputScales[keyIndex + 1];

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

int AnimationManager::findKeyframeIndex(const std::vector<float>& times, float time)
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

void AnimationManager::matrixFromTRS(const Float3& translation, const Float4& rotation, const Float3& scale, float* matrix)
{
    // Build transformation matrix from Translation, Rotation (quaternion), Scale
    AnimationMath::matrixFromQuaternion(rotation, matrix);

    // Apply scale
    matrix[0] *= scale.x; matrix[1] *= scale.x; matrix[2] *= scale.x;
    matrix[4] *= scale.y; matrix[5] *= scale.y; matrix[6] *= scale.y;
    matrix[8] *= scale.z; matrix[9] *= scale.z; matrix[10] *= scale.z;

    // Apply translation
    matrix[3] = translation.x;
    matrix[7] = translation.y;
    matrix[11] = translation.z;

    // Set bottom row
    matrix[12] = 0.0f; matrix[13] = 0.0f; matrix[14] = 0.0f; matrix[15] = 1.0f;
}

// ===== AnimationMath Namespace Implementation =====

namespace AnimationMath {



void transposeMatrix4x4(const float* input, float* output)
{
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            output[j*4 + i] = input[i*4 + j];
        }
    }
}

void invertMatrix4x4(const float* input, float* output)
{
    // 4x4 matrix inversion using cofactor expansion
    float inv[16];

    inv[0] = input[5] * input[10] * input[15] -
             input[5] * input[11] * input[14] -
             input[9] * input[6] * input[15] +
             input[9] * input[7] * input[14] +
             input[13] * input[6] * input[11] -
             input[13] * input[7] * input[10];

    inv[4] = -input[4] * input[10] * input[15] +
              input[4] * input[11] * input[14] +
              input[8] * input[6] * input[15] -
              input[8] * input[7] * input[14] -
              input[12] * input[6] * input[11] +
              input[12] * input[7] * input[10];

    inv[8] = input[4] * input[9] * input[15] -
             input[4] * input[11] * input[13] -
             input[8] * input[5] * input[15] +
             input[8] * input[7] * input[13] +
             input[12] * input[5] * input[11] -
             input[12] * input[7] * input[9];

    inv[12] = -input[4] * input[9] * input[14] +
               input[4] * input[10] * input[13] +
               input[8] * input[5] * input[14] -
               input[8] * input[6] * input[13] -
               input[12] * input[5] * input[10] +
               input[12] * input[6] * input[9];

    inv[1] = -input[1] * input[10] * input[15] +
              input[1] * input[11] * input[14] +
              input[9] * input[2] * input[15] -
              input[9] * input[3] * input[14] -
              input[13] * input[2] * input[11] +
              input[13] * input[3] * input[10];

    inv[5] = input[0] * input[10] * input[15] -
             input[0] * input[11] * input[14] -
             input[8] * input[2] * input[15] +
             input[8] * input[3] * input[14] +
             input[12] * input[2] * input[11] -
             input[12] * input[3] * input[10];

    inv[9] = -input[0] * input[9] * input[15] +
              input[0] * input[11] * input[13] +
              input[8] * input[1] * input[15] -
              input[8] * input[3] * input[13] -
              input[12] * input[1] * input[11] +
              input[12] * input[3] * input[9];

    inv[13] = input[0] * input[9] * input[14] -
              input[0] * input[10] * input[13] -
              input[8] * input[1] * input[14] +
              input[8] * input[2] * input[13] +
              input[12] * input[1] * input[10] -
              input[12] * input[2] * input[9];

    inv[2] = input[1] * input[6] * input[15] -
             input[1] * input[7] * input[14] -
             input[5] * input[2] * input[15] +
             input[5] * input[3] * input[14] +
             input[13] * input[2] * input[7] -
             input[13] * input[3] * input[6];

    inv[6] = -input[0] * input[6] * input[15] +
              input[0] * input[7] * input[14] +
              input[4] * input[2] * input[15] -
              input[4] * input[3] * input[14] -
              input[12] * input[2] * input[7] +
              input[12] * input[3] * input[6];

    inv[10] = input[0] * input[5] * input[15] -
              input[0] * input[7] * input[13] -
              input[4] * input[1] * input[15] +
              input[4] * input[3] * input[13] +
              input[12] * input[1] * input[7] -
              input[12] * input[3] * input[5];

    inv[14] = -input[0] * input[5] * input[14] +
               input[0] * input[6] * input[13] +
               input[4] * input[1] * input[14] -
               input[4] * input[2] * input[13] -
               input[12] * input[1] * input[6] +
               input[12] * input[2] * input[5];

    inv[3] = -input[1] * input[6] * input[11] +
              input[1] * input[7] * input[10] +
              input[5] * input[2] * input[11] -
              input[5] * input[3] * input[10] -
              input[9] * input[2] * input[7] +
              input[9] * input[3] * input[6];

    inv[7] = input[0] * input[6] * input[11] -
             input[0] * input[7] * input[10] -
             input[4] * input[2] * input[11] +
             input[4] * input[3] * input[10] +
             input[8] * input[2] * input[7] -
             input[8] * input[3] * input[6];

    inv[11] = -input[0] * input[5] * input[11] +
               input[0] * input[7] * input[9] +
               input[4] * input[1] * input[11] -
               input[4] * input[3] * input[9] -
               input[8] * input[1] * input[7] +
               input[8] * input[3] * input[5];

    inv[15] = input[0] * input[5] * input[10] -
              input[0] * input[6] * input[9] -
              input[4] * input[1] * input[10] +
              input[4] * input[2] * input[9] +
              input[8] * input[1] * input[6] -
              input[8] * input[2] * input[5];

    float det = input[0] * inv[0] + input[1] * inv[4] + input[2] * inv[8] + input[3] * inv[12];

    if (det == 0)
    {
        // Matrix is not invertible, return identity
        memset(output, 0, sizeof(float) * 16);
        output[0] = output[5] = output[10] = output[15] = 1.0f;
        return;
    }

    det = 1.0f / det;

    for (int i = 0; i < 16; i++)
    {
        output[i] = inv[i] * det;
    }
}

Float4 quaternionSlerp(const Float4& q1, const Float4& q2, float t)
{
    Float4 qa = q1;
    Float4 qb = q2;

    // Compute dot product
    float dot = qa.x * qb.x + qa.y * qb.y + qa.z * qb.z + qa.w * qb.w;

    // If dot product is negative, slerp won't take the shorter path
    if (dot < 0.0f)
    {
        qb.x = -qb.x;
        qb.y = -qb.y;
        qb.z = -qb.z;
        qb.w = -qb.w;
        dot = -dot;
    }

    if (dot > 0.9995f)
    {
        // Linear interpolation for very close quaternions
        Float4 result;
        result.x = qa.x + t * (qb.x - qa.x);
        result.y = qa.y + t * (qb.y - qa.y);
        result.z = qa.z + t * (qb.z - qa.z);
        result.w = qa.w + t * (qb.w - qa.w);

        // Normalize
        float length = sqrtf(result.x * result.x + result.y * result.y + result.z * result.z + result.w * result.w);
        if (length > 0.0f)
        {
            result.x /= length;
            result.y /= length;
            result.z /= length;
            result.w /= length;
        }

        return result;
    }
    else
    {
        // Spherical linear interpolation
        float angle = acosf(dot);
        float sinAngle = sinf(angle);
        float factor1 = sinf((1.0f - t) * angle) / sinAngle;
        float factor2 = sinf(t * angle) / sinAngle;

        Float4 result;
        result.x = qa.x * factor1 + qb.x * factor2;
        result.y = qa.y * factor1 + qb.y * factor2;
        result.z = qa.z * factor1 + qb.z * factor2;
        result.w = qa.w * factor1 + qb.w * factor2;

        return result;
    }
}

Float4 quaternionMultiply(const Float4& q1, const Float4& q2)
{
    Float4 result;
    result.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
    result.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
    result.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
    result.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
    return result;
}

void matrixFromQuaternion(const Float4& q, float* matrix)
{
    float x2 = q.x * 2.0f;
    float y2 = q.y * 2.0f;
    float z2 = q.z * 2.0f;
    float xx = q.x * x2;
    float xy = q.x * y2;
    float xz = q.x * z2;
    float yy = q.y * y2;
    float yz = q.y * z2;
    float zz = q.z * z2;
    float wx = q.w * x2;
    float wy = q.w * y2;
    float wz = q.w * z2;

    matrix[0] = 1.0f - (yy + zz);
    matrix[1] = xy - wz;
    matrix[2] = xz + wy;
    matrix[3] = 0.0f;

    matrix[4] = xy + wz;
    matrix[5] = 1.0f - (xx + zz);
    matrix[6] = yz - wx;
    matrix[7] = 0.0f;

    matrix[8] = xz - wy;
    matrix[9] = yz + wx;
    matrix[10] = 1.0f - (xx + yy);
    matrix[11] = 0.0f;

    matrix[12] = 0.0f;
    matrix[13] = 0.0f;
    matrix[14] = 0.0f;
    matrix[15] = 1.0f;
}

Float4 matrixToQuaternion(const float* matrix)
{
    Float4 q;

    float trace = matrix[0] + matrix[5] + matrix[10];

    if (trace > 0.0f)
    {
        float s = sqrtf(trace + 1.0f) * 2.0f; // s = 4 * qw
        q.w = 0.25f * s;
        q.x = (matrix[6] - matrix[9]) / s;
        q.y = (matrix[8] - matrix[2]) / s;
        q.z = (matrix[1] - matrix[4]) / s;
    }
    else if (matrix[0] > matrix[5] && matrix[0] > matrix[10])
    {
        float s = sqrtf(1.0f + matrix[0] - matrix[5] - matrix[10]) * 2.0f; // s = 4 * qx
        q.w = (matrix[6] - matrix[9]) / s;
        q.x = 0.25f * s;
        q.y = (matrix[1] + matrix[4]) / s;
        q.z = (matrix[8] + matrix[2]) / s;
    }
    else if (matrix[5] > matrix[10])
    {
        float s = sqrtf(1.0f + matrix[5] - matrix[0] - matrix[10]) * 2.0f; // s = 4 * qy
        q.w = (matrix[8] - matrix[2]) / s;
        q.x = (matrix[1] + matrix[4]) / s;
        q.y = 0.25f * s;
        q.z = (matrix[6] + matrix[9]) / s;
    }
    else
    {
        float s = sqrtf(1.0f + matrix[10] - matrix[0] - matrix[5]) * 2.0f; // s = 4 * qz
        q.w = (matrix[1] - matrix[4]) / s;
        q.x = (matrix[8] + matrix[2]) / s;
        q.y = (matrix[6] + matrix[9]) / s;
        q.z = 0.25f * s;
    }

    return q;
}

} // namespace AnimationMath