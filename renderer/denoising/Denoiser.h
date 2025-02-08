#pragma once

#include "shaders/LinearMath.h"

class Denoiser
{
public:
    static Denoiser &Get()
    {
        static Denoiser instance;
        return instance;
    }
    Denoiser(Denoiser const &) = delete;
    void operator=(Denoiser const &) = delete;

    void run(int width, int height, int historyWidth, int historyHeight);

private:
    Denoiser() {}

    Int2 bufferDim{};
    Int2 historyDim{};
};