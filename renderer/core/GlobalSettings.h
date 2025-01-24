#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#define ENABLE_DENOISING_NOISE_CALCULATION 0

namespace jazzfusion
{

    struct RenderPassSettings
    {
        std::vector<std::pair<bool *, std::string>> GetValueList()
        {
            return {
                {&enableTemporalDenoising, "Enable Temporal Denoising"},
                {&enableLocalSpatialFilter, "Enable Local SpatialFilter "},
                {&enableNoiseLevelVisualize, "Enable Noise Level Visualize"},
                {&enableWideSpatialFilter, "Enable Wide Spatial Filter"},
                {&enableTemporalDenoising2, "Enable Temporal Denoising 2"},
                {&enableBilateralFilter, "Enable Bilateral Filter"},

                {&enablePostProcess, "Enable Post Process"},
                {&enableDownScalePasses, "Enable Down Scale Passes"},
                {&enableHistogram, "Enable Histogram"},
                {&enableAutoExposure, "Enable Auto Exposure"},
                {&enableBloomEffect, "Enable Bloom Effect"},
                {&enableLensFlare, "Enable Lens Flare"},
                {&enableToneMapping, "Enable Tone Mapping"},
                {&enableSharpening, "Enable Sharpening"},

                {&enableEASU, "Enable EASU"},
            };
        }

        bool enableTemporalDenoising = false;
        bool enableLocalSpatialFilter = false;
        bool enableNoiseLevelVisualize = false;
        bool enableWideSpatialFilter = false;
        bool enableTemporalDenoising2 = false;
        bool enableBilateralFilter = false;

        bool enablePostProcess = false;
        bool enableDownScalePasses = false;
        bool enableHistogram = false;
        bool enableAutoExposure = false;
        bool enableBloomEffect = false;
        bool enableLensFlare = false;
        bool enableToneMapping = false;
        bool enableSharpening = false;

        bool enableEASU = false;
    };

    struct DenoisingParams
    {
        std::vector<std::pair<float *, std::string>> GetValueList()
        {
            return {
                {&local_denoise_sigma_normal, "local_denoise_sigma_normal"},
                {&local_denoise_sigma_depth, "local_denoise_sigma_depth"},
                {&local_denoise_sigma_material, "local_denoise_sigma_material"},

                {&large_denoise_sigma_normal, "large_denoise_sigma_normal"},
                {&large_denoise_sigma_depth, "large_denoise_sigma_depth"},
                {&large_denoise_sigma_material, "large_denoise_sigma_material"},

                {&temporal_denoise_sigma_normal, "temporal_denoise_sigma_normal"},
                {&temporal_denoise_sigma_depth, "temporal_denoise_sigma_depth"},
                {&temporal_denoise_sigma_material, "temporal_denoise_sigma_material"},
                {&temporal_denoise_depth_diff_threshold,
                 "temporal_denoise_depth_diff_threshold"},
                {&temporal_denoise_baseBlendingFactor,
                 "temporal_denoise_baseBlendingFactor"},
                {&temporal_denoise_antiFlickeringWeight,
                 "temporal_denoise_antiFlickeringWeight"},

                {&noise_threshold_local, "noise_threshold_local"},
                {&noise_threshold_large, "noise_threshold_large"},

                {&noiseBlend, "noiseBlend"},
            };
        }

        std::vector<std::pair<bool *, std::string>> GetBooleanValueList()
        {
            return {
                {&temporal_denoise_use_softmax, "temporal_denoise_use_softmax"},
            };
        }

        float local_denoise_sigma_normal = 100.0f;
        float local_denoise_sigma_depth = 0.1f;
        float local_denoise_sigma_material = 100.0f;

        float large_denoise_sigma_normal = 100.0f;
        float large_denoise_sigma_depth = 0.01f;
        float large_denoise_sigma_material = 100.0f;

        float temporal_denoise_sigma_normal = 100.0f;
        float temporal_denoise_sigma_depth = 0.1f;
        float temporal_denoise_sigma_material = 100.0f;
        float temporal_denoise_depth_diff_threshold = 0.1f;
        float temporal_denoise_baseBlendingFactor = 1.0f / 2.0f;
        float temporal_denoise_antiFlickeringWeight = 0.8f;

        float noise_threshold_local = 0.001f;
        float noise_threshold_large = 0.001f;

        float noiseBlend = 0.0f;

        bool temporal_denoise_use_softmax = true;
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
        float skyBrightness = 1.0f;
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

        static RenderPassSettings &GetRenderPassSettings()
        {
            return Get().renderPassSettings;
        }
        static DenoisingParams &GetDenoisingParams() { return Get().denoisingParams; }
        static PostProcessParams &GetPostProcessParams()
        {
            return Get().postProcessParams;
        }
        static SkyParams &GetSkyParams() { return Get().skyParams; }

        RenderPassSettings renderPassSettings{};
        DenoisingParams denoisingParams{};
        PostProcessParams postProcessParams{};
        SkyParams skyParams{};

        std::string cameraSaveFileName = "mycamera.bin";
        int iterationIndex = 0;

    private:
        GlobalSettings() {}
    };

} // namespace jazzfusion