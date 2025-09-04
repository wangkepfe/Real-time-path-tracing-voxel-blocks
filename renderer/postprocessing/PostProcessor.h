#pragma once

#include "shaders/LinearMath.h"

#ifndef SurfObj
#define SurfObj cudaSurfaceObject_t
#endif
#ifndef TexObj
#define TexObj cudaTextureObject_t
#endif

// Forward declarations
class PostProcessingPipeline;

class PostProcessor
{
public:
    static PostProcessor &Get()
    {
        static PostProcessor instance;
        return instance;
    }
    PostProcessor(PostProcessor const &) = delete;
    void operator=(PostProcessor const &) = delete;

    void run(Float4 *interopBuffer, int inputWidth, int inputHeight, int outputWidth, int outputHeight, float deltaTime);
    
    // Get computed exposure for tone mapping
    float GetComputedExposure() const { return m_computedExposure; }

private:
    PostProcessor() : m_pipeline(nullptr), m_computedExposure(1.0f) {}
    ~PostProcessor();

    int inputWidth;
    int inputHeight;
    int outputWidth;
    int outputHeight;
    
    PostProcessingPipeline* m_pipeline;
    float m_computedExposure;
};