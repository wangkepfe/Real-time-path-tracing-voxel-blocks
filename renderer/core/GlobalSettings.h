#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "shaders/LinearMath.h"

// Post Processing Pipeline Parameters Structure
struct PostProcessingPipelineParams
{
    // Bloom parameters
    bool enableBloom = true;
    float bloomThreshold = 1.0f;
    float bloomIntensity = 0.15f;
    float bloomRadius = 2.0f;

    // Auto-exposure parameters
    bool enableAutoExposure = true;
    float exposureSpeed = 1.0f;
    float exposureMin = -8.0f;
    float exposureMax = 8.0f;
    float exposureCompensation = 0.0f;
    float histogramMinPercent = 40.0f;
    float histogramMaxPercent = 80.0f;
    float targetLuminance = 0.18f;

    // Vignette parameters
    bool enableVignette = false;
    float vignetteStrength = 0.5f;
    float vignetteRadius = 0.8f;
    float vignetteSmoothness = 0.5f;

    // Lens flare parameters
    bool enableLensFlare = false;
    float lensFlareIntensity = 0.0100f;
    float lensFlareThreshold = 4.0f;
    float lensFlareGhostSpacing = 0.0800f;
    int lensFlareGhostCount = 4;
    float lensFlareHaloRadius = 0.1000f;
    float lensFlareSunSize = 0.0060f;
    float lensFlareDistortion = 0.0015f;
    bool lensFlareHalfRes = true;
    bool lensFlareNeighborFilter = true;
    int lensFlareMaxSpots = 16;

    std::vector<std::tuple<float *, std::string, float, float, bool>> GetValueList()
    {
        return {
            {&bloomThreshold, "Bloom Threshold", 0.0f, 5.0f, false},
            {&bloomIntensity, "Bloom Intensity", 0.0f, 2.0f, false},
            {&bloomRadius, "Bloom Radius", 0.5f, 5.0f, false},
            {&exposureSpeed, "Auto-Exposure Speed", 0.1f, 5.0f, false},
            {&exposureMin, "Exposure Min (EV)", -12.0f, 0.0f, false},
            {&exposureMax, "Exposure Max (EV)", 0.0f, 12.0f, false},
            {&exposureCompensation, "Exposure Compensation", -5.0f, 5.0f, false},
            {&histogramMinPercent, "Histogram Min %", 0.0f, 50.0f, false},
            {&histogramMaxPercent, "Histogram Max %", 50.0f, 100.0f, false},
            {&targetLuminance, "Target Luminance", 0.05f, 0.5f, false},
            {&vignetteStrength, "Vignette Strength", 0.0f, 2.0f, false},
            {&vignetteRadius, "Vignette Radius", 0.1f, 1.5f, false},
            {&vignetteSmoothness, "Vignette Smoothness", 0.1f, 1.0f, false},
            {&lensFlareIntensity, "Lens Flare Intensity", 0.0f, 1.0f, false},
            {&lensFlareThreshold, "Lens Flare Threshold", 1.0f, 10.0f, false},
            {&lensFlareGhostSpacing, "Ghost Spacing", 0.1f, 1.0f, false},
            {&lensFlareHaloRadius, "Halo Radius", 0.1f, 0.8f, false},
            {&lensFlareSunSize, "Sun Size", 0.005f, 0.05f, false},
            {&lensFlareDistortion, "Chromatic Aberration", 0.0f, 0.2f, false}};
    }

    std::vector<std::pair<bool *, std::string>> GetBooleanValueList()
    {
        return {
            {&enableBloom, "Enable Bloom"},
            {&enableAutoExposure, "Enable Auto Exposure"},
            {&enableVignette, "Enable Vignette"},
            {&enableLensFlare, "Enable Lens Flare"},
            {&lensFlareHalfRes, "Lens Flare Half Resolution"},
            {&lensFlareNeighborFilter, "Lens Flare Neighbor Filter"}};
    }

    std::vector<std::pair<int *, std::string>> GetIntValueList()
    {
        return {
            {&lensFlareGhostCount, "Lens Flare Ghost Count"},
            {&lensFlareMaxSpots, "Max Lens Flare Spots"}
        };
    }
};

struct DenoisingParams
{
    std::vector<std::pair<bool *, std::string>> GetBooleanValueList()
    {
        return {
            {&enableHitDistanceReconstruction, "Enable Hit Distance Reconstruction"},
            {&enablePrePass, "Enable Pre-Pass"},
            {&enableTemporalAccumulation, "Enable Temporal Accumulation"},
            {&enableHistoryFix, "Enable History Fix"},
            {&enableHistoryClamping, "Enable History Clamping"},
            {&enableSpatialFiltering, "Enable Spatial Filtering (A-trous)"},
            {&enableFireflyFilter, "Enable Firefly Filter"},
        };
    }

    std::vector<std::pair<float *, std::string>> GetValueList()
    {
        return {
            {&maxAccumulatedFrameNum, "Max Accumulated Frame Num"},
            {&maxFastAccumulatedFrameNum, "Max Fast Accumulated Frame Num"},
            {&phiLuminance, "Phi Luminance"},
            {&lobeAngleFraction, "Lobe Angle Fraction"},
            {&roughnessFraction, "Roughness Fraction"},
            {&depthThreshold, "Depth Threshold"},
            {&disocclusionThreshold, "Disocclusion Threshold"},
            {&disocclusionThresholdAlternate, "Disocclusion Threshold Alternate"},
            {&denoisingRange, "Denoising Range"},
        };
    }

    std::vector<std::pair<int *, std::string>> GetIntValueList()
    {
        return {
            {&atrousIterationNum, "A-trous Iteration Number"},
        };
    }

    bool enableHitDistanceReconstruction = false;
    bool enablePrePass = false;
    bool enableTemporalAccumulation = true;
    bool enableHistoryFix = true;
    bool enableHistoryClamping = true;
    bool enableSpatialFiltering = true;
    bool enableFireflyFilter = true;

    float maxAccumulatedFrameNum = 30.0f;
    float maxFastAccumulatedFrameNum = 6.0f;

    float phiLuminance = 2.0f;
    float lobeAngleFraction = 0.5f;
    float roughnessFraction = 0.15f;
    float depthThreshold = 0.003f;

    int atrousIterationNum = 5;

    float disocclusionThreshold = 0.01f;
    float disocclusionThresholdAlternate = 0.05f;

    float denoisingRange = 500000.0f;
};

// Tone Mapping Parameters Structure
// Tone Mapping Parameters Structure
struct ToneMappingParams
{
    // Manual exposure (used when auto-exposure is disabled)
    float manualExposure = 10.0f; // Manual exposure multiplier

    // Tone mapping curve selection
    enum ToneMappingCurve
    {
        CURVE_NARKOWICZ_ACES = 0, // Fast ACES approximation
        CURVE_UNCHARTED2 = 1,     // Uncharted 2 filmic
        CURVE_REINHARD = 2        // Simple Reinhard
    };
    ToneMappingCurve curve = CURVE_NARKOWICZ_ACES;

    // Highlight handling
    float highlightDesaturation = 0.8f; // Amount of desaturation for bright areas
    float whitePoint = 10.0f;           // Scene white point luminance

    // Color grading
    float contrast = 1.0f;   // Contrast adjustment
    float saturation = 1.0f; // Saturation adjustment
    float lift = 0.0f;       // Shadows lift
    float gain = 1.0f;       // Highlights gain

    // Output is always sRGB (no HDR output modes)

    // Chromatic adaptation
    bool enableChromaticAdaptation = true;

    std::vector<std::tuple<float *, std::string, float, float, bool>>
    GetValueList()
    {
        return {
            {&manualExposure, "Manual Exposure", 0.1f, 20.0f, true},
            {&highlightDesaturation, "Highlight Desaturation", 0.0f, 1.0f, false},
            {&whitePoint, "White Point", 1.0f, 20.0f, true},
            {&contrast, "Contrast", 0.5f, 2.0f, false},
            {&saturation, "Saturation", 0.0f, 2.0f, false},
            {&gain, "Gain", 0.5f, 2.0f, false},
            {&lift, "Lift", -0.5f, 0.5f, false}};
    }
};

struct SkyParams
{
    std::vector<std::tuple<float *, std::string, float, float, bool>>
    GetValueList()
    {
        return {{&timeOfDay, "Time of Day", 0.01f, 0.99f, false},
                {&sunAxisAngle, "Sun Axis Angle", 5.0f, 85.0f, false},
                {&sunAxisRotate, "Sun Axis Rotate", 0.0f, 360.0f, false},
                {&skyBrightness, "Sky Brightness", 0.000001f, 1.0f, true}};
    }

    bool needRegenerate = true;
    float timeOfDay = 0.25f;
    float sunAxisAngle = 45.0f;
    float sunAxisRotate = 0.0f;
    float skyBrightness = 1.0f; // 0.00001f;
};

// Character Movement Parameters
struct CharacterMovementParams
{
    std::vector<std::tuple<float *, std::string, float, float, bool>>
    GetValueList()
    {
        return {
            {&walkMoveForce, "Walk Move Force", 0.1f, 10.0f, false},
            {&runMoveForce, "Run Move Force", 0.1f, 15.0f, false},
            {&walkMaxSpeed, "Walk Max Speed", 1.0f, 10.0f, false},
            {&runMaxSpeed, "Run Max Speed", 1.0f, 15.0f, false},
            {&jumpForce, "Jump Force", 1.0f, 15.0f, false},
            {&gravity, "Gravity", -20.0f, -1.0f, false},
            {&friction, "Friction", 0.1f, 2.0f, false},
            {&rotationSpeed, "Rotation Speed", 1.0f, 20.0f, false},
            {&radius, "Collision Radius", 0.1f, 1.0f, false},
            {&height, "Character Height", 0.5f, 3.0f, false}};
    }

    float walkMoveForce = 3.0f;
    float runMoveForce = 5.0f;
    float walkMaxSpeed = 2.0f;
    float runMaxSpeed = 3.0f;
    float jumpForce = 6.0f;
    float gravity = -9.81f;
    float friction = 0.8f;
    float rotationSpeed = 8.0f;
    float radius = 0.3f;
    float height = 1.8f;
};

// Character Animation Parameters
struct CharacterAnimationParams
{
    std::vector<std::tuple<float *, std::string, float, float, bool>>
    GetValueList()
    {
        return {
            {&walkSpeedThreshold, "Walk Speed Threshold", 0.01f, 1.0f, false},
            {&mediumSpeedThreshold, "Medium Speed Threshold", 1.0f, 5.0f, false},
            {&runSpeedThreshold, "Run Speed Threshold", 0.01f, 1.0f, false},
            {&runMediumSpeedThreshold, "Run Medium Speed Threshold", 1.0f, 8.0f, false},
            {&animationSpeed, "Animation Speed", 0.1f, 3.0f, false},
            {&blendRatio, "Blend Ratio", 0.0f, 1.0f, false},
            {&placeAnimationSpeed, "Place Animation Speed", 0.1f, 3.0f, false}};
    }

    float walkSpeedThreshold = 0.01f;
    float mediumSpeedThreshold = 2.5f;
    float runSpeedThreshold = 0.01f;
    float runMediumSpeedThreshold = 4.0f;
    float animationSpeed = 1.0f;
    float blendRatio = 0.0f;
    float placeAnimationSpeed = 1.0f;
};

// Camera Movement Parameters
struct CameraMovementParams
{
    std::vector<std::tuple<float *, std::string, float, float, bool>>
    GetValueList()
    {
        return {
            {&moveSpeed, "Free Camera Move Speed", 0.1f, 10.0f, false},
            {&cursorMoveSpeed, "Mouse Sensitivity", 0.0001f, 0.01f, false},
            {&fov, "Field of View", 30.0f, 160.0f, false},
            {&followSpeed, "Character Follow Speed", 1.0f, 15.0f, false},
            {&followDistance, "Character Follow Distance", 1.0f, 20.0f, false},
            {&followHeight, "Character Follow Height", 0.5f, 10.0f, false},
            {&gameplayHeight, "Gameplay Height", 0.5f, 3.0f, false}};
    }

    float moveSpeed = 3.0f;
    float cursorMoveSpeed = 0.001f;
    float fov = 90.0f;
    float followSpeed = 5.0f;
    float followDistance = 5.0f;
    float followHeight = 2.5f;
    float gameplayHeight = 1.5f;
};

// Rendering Parameters
struct RenderingParams
{
    std::vector<std::tuple<float *, std::string, float, float, bool>>
    GetValueList()
    {
        return {
            {&maxFpsAllowed, "Max FPS", 30.0f, 240.0f, false},
            {&targetFPS, "Target FPS", 30.0f, 144.0f, false}};
    }

    std::vector<std::pair<bool *, std::string>> GetBooleanValueList()
    {
        return {
            {&dynamicResolution, "Dynamic Resolution"},
            {&vsync, "VSync"}};
    }

    std::vector<std::pair<int *, std::string>> GetIntValueList()
    {
        return {
            {&windowWidth, "Window Width"},
            {&windowHeight, "Window Height"},
            {&maxRenderWidth, "Max Render Width"},
            {&maxRenderHeight, "Max Render Height"},
            {&minRenderWidth, "Min Render Width"},
            {&minRenderHeight, "Min Render Height"}};
    }

    float maxFpsAllowed = 144.0f;
    float targetFPS = 60.0f;
    bool dynamicResolution = false;
    bool vsync = false;
    int windowWidth = 2560;
    int windowHeight = 1440;
    int maxRenderWidth = 2560;
    int maxRenderHeight = 1440;
    int minRenderWidth = 480;
    int minRenderHeight = 270;
};

class GlobalSettings
{
public:
    static GlobalSettings &Get()
    {
        static GlobalSettings instance;
        return instance;
    }
    GlobalSettings(GlobalSettings const &) = delete;
    void operator=(GlobalSettings const &) = delete;

    static DenoisingParams &GetDenoisingParams() { return Get().denoisingParams; }
    static ToneMappingParams &GetToneMappingParams() { return Get().toneMappingParams; }
    static PostProcessingPipelineParams &GetPostProcessingPipelineParams() { return Get().postProcessingPipelineParams; }
    static SkyParams &GetSkyParams() { return Get().skyParams; }
    static CharacterMovementParams &GetCharacterMovementParams() { return Get().characterMovementParams; }
    static CharacterAnimationParams &GetCharacterAnimationParams() { return Get().characterAnimationParams; }
    static CameraMovementParams &GetCameraMovementParams() { return Get().cameraMovementParams; }
    static RenderingParams &GetRenderingParams() { return Get().renderingParams; }

    // Offline rendering flag
    static bool IsOfflineMode() { return Get().offlineMode; }
    static void SetOfflineMode(bool offline) { Get().offlineMode = offline; }

    // Time management is now handled by Backend's Timer.h

    // YAML serialization methods
    bool LoadFromYAML(const std::string &filepath);
    void SaveToYAML(const std::string &filepath) const;

    DenoisingParams denoisingParams{};
    ToneMappingParams toneMappingParams{};
    PostProcessingPipelineParams postProcessingPipelineParams{};
    SkyParams skyParams{};
    CharacterMovementParams characterMovementParams{};
    CharacterAnimationParams characterAnimationParams{};
    CameraMovementParams cameraMovementParams{};
    RenderingParams renderingParams{};

    std::string cameraSaveFileName = "mycamera.bin";
    int iterationIndex = 0;

private:
    GlobalSettings() {}

    bool offlineMode = false;

    // Time management is now handled by Backend's Timer.h

    // Private helper methods for parsing YAML sections
    void parseDenosingSettings(const std::string &key, const std::string &value);
    void parseToneMappingSettings(const std::string &key, const std::string &value);
    void parsePostProcessingPipelineSettings(const std::string &key, const std::string &value);
    void parseSkySettings(const std::string &key, const std::string &value);
    void parseCharacterMovementSettings(const std::string &key, const std::string &value);
    void parseCharacterAnimationSettings(const std::string &key, const std::string &value);
    void parseCameraMovementSettings(const std::string &key, const std::string &value);
    void parseRenderingSettings(const std::string &key, const std::string &value);
};




