#pragma once

#include "shaders/LinearMath.h"

#ifndef SurfObj
#define SurfObj cudaSurfaceObject_t
#endif
#ifndef TexObj
#define TexObj cudaTextureObject_t
#endif

namespace jazzfusion
{

class PostProcessor
{
public:
    static PostProcessor& Get()
    {
        static PostProcessor instance;
        return instance;
    }
    PostProcessor(PostProcessor const&) = delete;
    void operator=(PostProcessor const&) = delete;

    void init(int inputWidth, int inputHeight, int outputWidth, int outputHeight);
    void render(Float4* interopBuffer, SurfObj colorBuffer, TexObj colorTex);

private:
    PostProcessor() {}

    int inputWidth;
    int inputHeight;
    int outputWidth;
    int outputHeight;

    TexObj m_scaledTexture;
    SurfObj m_scaledBuffer;
    cudaTextureDesc m_scaledTexDesc;
    cudaArray_t m_scaledBufferArray;
};

}