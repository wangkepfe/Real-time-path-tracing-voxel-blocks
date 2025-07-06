#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>


// Based on NRD ReLaX denoiser settings with comprehensive parameter coverage
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
            {&enableAntiFirefly, "Enable Anti-Firefly"},
            {&enableRoughnessEdgeStopping, "Enable Roughness Edge Stopping"},
        };
    }

    std::vector<std::pair<float *, std::string>> GetValueList()
    {
        return {
            // Temporal accumulation parameters
            {&maxAccumulatedFrameNum, "Max Accumulated Frame Num"},
            {&maxFastAccumulatedFrameNum, "Max Fast Accumulated Frame Num"},
            {&historyFixFrameNum, "History Fix Frame Num"},

            // Pre-pass blur settings
            {&prepassBlurRadius, "Pre-pass Blur Radius"},
            {&minHitDistanceWeight, "Min Hit Distance Weight"},

            // A-trous spatial filtering
            {&phiLuminance, "Phi Luminance"},
            {&lobeAngleFraction, "Lobe Angle Fraction"},
            {&roughnessFraction, "Roughness Fraction"},
            {&depthThreshold, "Depth Threshold"},
            {&minLuminanceWeight, "Min Luminance Weight"},

            // Edge stopping relaxation
            {&luminanceEdgeStoppingRelaxation, "Luminance Edge Stopping Relaxation"},
            {&normalEdgeStoppingRelaxation, "Normal Edge Stopping Relaxation"},
            {&roughnessEdgeStoppingRelaxation, "Roughness Edge Stopping Relaxation"},

            // Antilag settings
            {&antilagAccelerationAmount, "Antilag Acceleration Amount"},
            {&antilagSpatialSigmaScale, "Antilag Spatial Sigma Scale"},
            {&antilagTemporalSigmaScale, "Antilag Temporal Sigma Scale"},
            {&antilagResetAmount, "Antilag Reset Amount"},

            // History clamping
            {&historyClampingColorBoxSigmaScale, "History Clamping Color Box Sigma Scale"},

            // Disocclusion
            {&disocclusionThreshold, "Disocclusion Threshold"},
            {&disocclusionThresholdAlternate, "Disocclusion Threshold Alternate"},

            // Confidence driven parameters
            {&confidenceDrivenRelaxationMultiplier, "Confidence Driven Relaxation Multiplier"},
            {&confidenceDrivenLuminanceEdgeStoppingRelaxation, "Confidence Driven Luminance Edge Stopping Relaxation"},
            {&confidenceDrivenNormalEdgeStoppingRelaxation, "Confidence Driven Normal Edge Stopping Relaxation"},
        };
    }

    std::vector<std::pair<int *, std::string>> GetIntValueList()
    {
        return {
            {&atrousIterationNum, "A-trous Iteration Number"},
            {&historyFixBasePixelStride, "History Fix Base Pixel Stride"},
            {&spatialVarianceEstimationHistoryThreshold, "Spatial Variance Estimation History Threshold"},
        };
    }

    // Pass control flags (replace hardcoded if statements)
    bool enableHitDistanceReconstruction = false;
    bool enablePrePass = false;
    bool enableTemporalAccumulation = true;
    bool enableHistoryFix = true;
    bool enableHistoryClamping = true;
    bool enableSpatialFiltering = true;
    bool enableAntiFirefly = false;
    bool enableRoughnessEdgeStopping = true;

    // Temporal accumulation parameters (based on NRD ReLaX defaults)
    float maxAccumulatedFrameNum = 30.0f;
    float maxFastAccumulatedFrameNum = 6.0f;
    float historyFixFrameNum = 3.0f;

    // Pre-pass settings
    float prepassBlurRadius = 30.0f;
    float minHitDistanceWeight = 0.1f;

    // A-trous spatial filtering parameters
    float phiLuminance = 2.0f;
    float lobeAngleFraction = 0.5f;
    float roughnessFraction = 0.15f;
    float depthThreshold = 0.003f;
    float minLuminanceWeight = 0.0f;
    int atrousIterationNum = 5;

    // Edge stopping relaxation parameters
    float luminanceEdgeStoppingRelaxation = 0.5f;
    float normalEdgeStoppingRelaxation = 0.3f;
    float roughnessEdgeStoppingRelaxation = 1.0f;

    // Antilag settings (based on NRD ReLaX defaults)
    float antilagAccelerationAmount = 0.3f;
    float antilagSpatialSigmaScale = 4.5f;
    float antilagTemporalSigmaScale = 0.5f;
    float antilagResetAmount = 0.5f;

    // History clamping
    float historyClampingColorBoxSigmaScale = 2.0f;

    // History fix parameters
    int historyFixBasePixelStride = 14;
    float historyFixEdgeStoppingNormalPower = 8.0f;

    // Spatial variance estimation
    int spatialVarianceEstimationHistoryThreshold = 3;

    // Disocclusion parameters
    float disocclusionThreshold = 0.01f;
    float disocclusionThresholdAlternate = 0.05f;

    // Confidence driven parameters
    float confidenceDrivenRelaxationMultiplier = 0.0f;
    float confidenceDrivenLuminanceEdgeStoppingRelaxation = 0.0f;
    float confidenceDrivenNormalEdgeStoppingRelaxation = 0.0f;

    // Denoising range
    float denoisingRange = 500000.0f;
};

struct PostProcessParams
{
    std::vector<std::tuple<float *, std::string, float, float, bool>>
    GetValueList()
    {
        return {
            {&postGain, "Post Gain", 1.0f, 100.0f, true},
            {&gain, "Gain", 1.0f, 100.0f, true},
            {&maxWhite, "Max White", 0.000001f, 100.0f, true},
        };
    }

    float postGain = 1.0f;
    float gain = 16.0f;
    float maxWhite = 1.0f;
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
    static PostProcessParams &GetPostProcessParams()
    {
        return Get().postProcessParams;
    }
    static SkyParams &GetSkyParams() { return Get().skyParams; }

    // Offline rendering flag
    static bool IsOfflineMode() { return Get().offlineMode; }
    static void SetOfflineMode(bool offline) { Get().offlineMode = offline; }

    DenoisingParams denoisingParams{};
    PostProcessParams postProcessParams{};
    SkyParams skyParams{};

    std::string cameraSaveFileName = "mycamera.bin";
    int iterationIndex = 0;

private:
    GlobalSettings() {}

    bool offlineMode = false;
};