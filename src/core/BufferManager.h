#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "shaders/LinearMath.h"

namespace jazzfusion
{

enum Buffer2DName
{
    RenderColorBuffer,             // main render buffer
    MaterialBuffer,
    MaterialHistoryBuffer,

    AccumulationColorBuffer,       // accumulation render buffer
    HistoryColorBuffer,

    ColorBuffer4,                  // 1/4 res color buffer
    ColorBuffer16,                 // 1/6 res color buffer
    ColorBuffer64,                 // 1/64 res color buffer
    BloomBuffer4,                  // 1/4 res bloom buffer
    BloomBuffer16,                 // 1/16 bloom buffer

    NormalBuffer,                  // normalBu buffer
    DepthBuffer,                   // depth buffer
    HistoryDepthBuffer,            // depth buffer

    MotionVectorBuffer,            // motion vector buffer
    NoiseLevelBuffer,              // noise level
    NoiseLevelBuffer16x16,

    SkyBuffer,                     // sky
    SunBuffer,

    AlbedoBuffer,

    Buffer2DCount,
};

struct Buffer2D
{
    Buffer2D() = default;
    ~Buffer2D() { clear(); }

    void init(const cudaChannelFormatDesc* pFormat, UInt2 dim, uint usageFlag);
    void clear();

    SurfObj buffer;
    cudaArray_t bufferArray;
};

class BufferManager
{
public:
    static BufferManager& Get()
    {
        static BufferManager instance;
        return instance;
    }
    BufferManager(BufferManager const&) = delete;
    void operator=(BufferManager const&) = delete;

    void init();

    SurfObj GetBuffer2D(Buffer2DName name) { return m_buffers[(uint)name].buffer; }

private:
    std::vector<Buffer2D> m_buffers{};

    BufferManager() {}
};

}