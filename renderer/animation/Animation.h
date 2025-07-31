#pragma once

#include "../shaders/LinearMath.h"
#include <vector>
#include <string>

// Joint representation for skeletal animation
struct Joint
{
    int parentIndex = -1;                             // Index of parent joint (-1 for root)
    Float3 position = Float3(0.0f);                   // Local position relative to parent
    Float4 rotation = Float4(0.0f, 0.0f, 0.0f, 1.0f); // Quaternion rotation
    Float3 scale = Float3(1.0f);                      // Local scale

    // Bind pose (original transform from GLTF)
    Float3 bindPosition = Float3(0.0f);                   // Bind pose position
    Float4 bindRotation = Float4(0.0f, 0.0f, 0.0f, 1.0f); // Bind pose rotation
    Float3 bindScale = Float3(1.0f);                      // Bind pose scale

    // Computed transform matrices (updated during animation)
    Mat4 localMatrix;       // Local transform matrix
    Mat4 globalMatrix;      // World space transform matrix
    Mat4 inverseBindMatrix; // Inverse bind pose matrix

    std::string name; // Joint name for debugging
};

// Animation keyframe for different interpolation types
template <typename T>
struct AnimationKeyframe
{
    float time; // Time in seconds
    T value;    // Keyframe value
};

// Animation sampler for different property types
struct AnimationSampler
{
    enum InterpolationType
    {
        STEP,
        LINEAR,
        CUBICSPLINE
    };

    InterpolationType interpolation = LINEAR;
    std::vector<float> inputTimes;          // Time values
    std::vector<Float3> outputTranslations; // Translation keyframes
    std::vector<Float4> outputRotations;    // Rotation keyframes (quaternions)
    std::vector<Float3> outputScales;       // Scale keyframes
};

// Animation channel that targets a specific joint property
struct AnimationChannel
{
    enum TargetPath
    {
        TRANSLATION,
        ROTATION,
        SCALE
    };

    int targetJoint;       // Index of target joint
    TargetPath targetPath; // Which property to animate
    int samplerIndex;      // Index into animation's samplers array
};

// Complete animation clip
struct AnimationClip
{
    std::string name;                       // Animation name
    float duration;                         // Animation duration in seconds
    std::vector<AnimationSampler> samplers; // Animation samplers
    std::vector<AnimationChannel> channels; // Animation channels

    bool isLooping = true;      // Whether animation loops
    float playbackSpeed = 1.0f; // Playback speed multiplier
};

// Animation state for a playing animation
struct AnimationState
{
    int clipIndex = -1;       // Index of current animation clip
    float currentTime = 0.0f; // Current time in animation
    float weight = 1.0f;      // Blend weight (for animation blending)
    bool isPlaying = false;   // Whether animation is currently playing
    bool hasFinished = false; // Whether animation has finished (for non-looping)
};