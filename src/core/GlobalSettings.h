#pragma once

#include <vector>
#include <utility>
#include <string>
#include <tuple>

namespace jazzfusion
{

struct RenderPassSettings
{
    std::vector<std::pair<bool*, std::string>> GetValueList()
    {
        return {
            { &enableTemporalDenoising  , "Enable Temporal Denoising"   },
            { &enableLocalSpatialFilter , "Enable Local SpatialFilter " },
            { &enableNoiseLevelVisualize, "Enable Noise Level Visualize"},
            { &enableWideSpatialFilter  , "Enable Wide Spatial Filter"  },
            { &enableTemporalDenoising2 , "Enable Temporal Denoising 2" },
            { &enablePostProcess        , "Enable Post Process"         },
            { &enableDownScalePasses    , "Enable Down Scale Passes"    },
            { &enableHistogram          , "Enable Histogram"            },
            { &enableAutoExposure       , "Enable Auto Exposure"        },
            { &enableBloomEffect        , "Enable Bloom Effect"         },
            { &enableLensFlare          , "Enable Lens Flare"           },
            { &enableToneMapping        , "Enable Tone Mapping"         },
            { &enableSharpening         , "Enable Sharpening"           }
        };
    }

    bool enableTemporalDenoising = true;
    bool enableLocalSpatialFilter = true;
    bool enableNoiseLevelVisualize = false;
    bool enableWideSpatialFilter = true;
    bool enableTemporalDenoising2 = true;

    bool enablePostProcess = true;
    bool enableDownScalePasses = true;
    bool enableHistogram = true;
    bool enableAutoExposure = true;
    bool enableBloomEffect = false;
    bool enableLensFlare = false;
    bool enableToneMapping = true;
    bool enableSharpening = true;
};

struct DenoisingParams
{
    std::vector<std::pair<float*, std::string>> GetValueList()
    {
        return {
            { &local_denoise_sigma_normal  , "local_denoise_sigma_normal" },
            { &local_denoise_sigma_depth   , "local_denoise_sigma_depth" },
            { &local_denoise_sigma_material, "local_denoise_sigma_material" },

            { &large_denoise_sigma_normal  , "large_denoise_sigma_normal" },
            { &large_denoise_sigma_depth   , "large_denoise_sigma_depth" },
            { &large_denoise_sigma_material, "large_denoise_sigma_material" },

            { &temporal_denoise_sigma_normal  , "temporal_denoise_sigma_normal" },
            { &temporal_denoise_sigma_depth   , "temporal_denoise_sigma_depth" },
            { &temporal_denoise_sigma_material, "temporal_denoise_sigma_material" },
            { &temporal_denoise_depth_diff_threshold, "temporal_denoise_depth_diff_threshold" },

            { &noise_threshold_local, "noise_threshold_local"},
            { &noise_threshold_large, "noise_threshold_large"},
        };
    }

    std::vector<std::pair<bool*, std::string>> GetBooleanValueList()
    {
        return {
            { &temporal_denoise_use_softmax  , "temporal_denoise_use_softmax"   },
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

    float noise_threshold_local = 0.001f;
    float noise_threshold_large = 0.001f;

    bool temporal_denoise_use_softmax = true;
};

class GlobalSettings
{
public:
    static GlobalSettings& Get()
    {
        static GlobalSettings instance;
        return instance;
    }
    GlobalSettings(GlobalSettings const&) = delete;
    void operator=(GlobalSettings const&) = delete;

    static RenderPassSettings& GetRenderPassSettings()
    {
        return Get().renderPassSettings;
    }
    static DenoisingParams& GetDenoisingParams()
    {
        return Get().denoisingParams;
    }

    static const std::string& GetCameraSaveFileName() { return Get().cameraSaveFileName; }

    RenderPassSettings renderPassSettings{};
    DenoisingParams denoisingParams{};

    std::string cameraSaveFileName = "mycamera.bin";

private:
    GlobalSettings() {}
};

}