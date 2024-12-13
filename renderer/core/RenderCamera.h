#pragma once

#include "shaders/Camera.h"

namespace jazzfusion
{

class RenderCamera
{
public:
    static RenderCamera& Get()
    {
        static RenderCamera instance;
        return instance;
    }
    RenderCamera(RenderCamera const&) = delete;
    void operator=(RenderCamera const&) = delete;

    Camera camera{};

private:
    RenderCamera() {}
};

}