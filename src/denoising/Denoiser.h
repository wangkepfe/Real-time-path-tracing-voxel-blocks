#pragma once

#include "shaders/LinearMath.h"

namespace jazzfusion
{

class Denoiser
{
public:
    static Denoiser& Get()
    {
        static Denoiser instance;
        return instance;
    }
    Denoiser(Denoiser const&) = delete;
    void operator=(Denoiser const&) = delete;

    void init(int width, int height, int historyWidth, int historyHeight);
    void run(SurfObj inColorBuffer, TexObj outColorBuffer);

private:
    Denoiser() {}

    Int2 bufferDim;
    Int2 historyDim;
};

}