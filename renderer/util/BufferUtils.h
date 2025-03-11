#pragma once

#include "shaders/LinearMath.h"

void BufferSetFloat1(Int2 bufferDim, SurfObj outBuffer, float val);
void BufferSetFloat4(Int2 bufferDim, SurfObj outBuffer, Float4 val);

void BufferCopyFloat1(Int2 bufferDim, SurfObj inBuffer, SurfObj outBuffer);
void BufferCopyFloat4(Int2 bufferDim, SurfObj inBuffer, SurfObj outBuffer);