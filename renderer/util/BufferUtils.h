#pragma once

#include "shaders/LinearMath.h"

namespace jazzfusion
{
    void BufferSetFloat1(Int2 bufferDim, SurfObj outBuffer, float val);
    void BufferSetFloat4(Int2 bufferDim, SurfObj outBuffer, Float4 val);
}