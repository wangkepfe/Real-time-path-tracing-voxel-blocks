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

    void run(Float4* interopBuffer, int inputWidth, int inputHeight, int outputWidth, int outputHeight);

private:
    PostProcessor() {}

    int inputWidth;
    int inputHeight;
    int outputWidth;
    int outputHeight;
};

}