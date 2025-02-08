#pragma once

#include "shaders/Camera.h"

class RenderCamera
{
public:
    static RenderCamera &Get()
    {
        static RenderCamera instance;
        return instance;
    }
    RenderCamera(RenderCamera const &) = delete;
    void operator=(RenderCamera const &) = delete;

    Camera camera{};
    Camera historyCamera{};

private:
    RenderCamera() {}
};