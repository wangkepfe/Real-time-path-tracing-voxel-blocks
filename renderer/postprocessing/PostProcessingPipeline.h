#pragma once

#include "shaders/LinearMath.h"
#include "core/GlobalSettings.h"

class PostProcessingPipeline
{
public:
    PostProcessingPipeline();
    ~PostProcessingPipeline();
    
    void Initialize(int width, int height);
    
    // Execute the pipeline and return computed exposure
    float Execute(
        SurfObj colorBuffer,
        Int2 size,
        const PostProcessingPipelineParams& pipelineParams,
        const ToneMappingParams& toneMappingParams,
        float deltaTime);
        
private:
    // Luminance histogram for auto-exposure
    float* d_luminanceHistogram;
    int histogramBins;
    
    // Average luminance tracking
    float m_currentAvgLuminance;
    float m_targetAvgLuminance;
    
    // Frame timing
    float m_lastFrameTime;
    
    // Lens flare detection
    float* d_brightSpots;  // Device memory for bright spot data
    int* d_spotCount;      // Device memory for spot count
    
    // Helper functions
    void ComputeLuminanceHistogram(SurfObj colorBuffer, Int2 size);
    float ComputeAutoExposure(const PostProcessingPipelineParams& params, float deltaTime);
    void ApplyBloom(SurfObj colorBuffer, Int2 size, const PostProcessingPipelineParams& params);
    void ApplyLensFlare(SurfObj colorBuffer, Int2 size, const PostProcessingPipelineParams& params);
};