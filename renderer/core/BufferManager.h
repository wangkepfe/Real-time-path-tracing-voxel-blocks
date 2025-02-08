#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "shaders/LinearMath.h"

enum Buffer2DName
{
    IlluminationBuffer,
    IlluminationOutputBuffer,
    IlluminationPingBuffer,
    IlluminationPongBuffer,
    NormalRoughnessBuffer,
    DepthBuffer,
    MaterialBuffer,
    AlbedoBuffer,
    MotionVectorBuffer,
    OutputColorBuffer,
    HistoryLengthBuffer,
    PrevDepthBuffer,
    PrevMaterialBuffer,
    PrevIlluminationBuffer,
    PrevFastIlluminationBuffer,
    PrevHistoryLengthBuffer,
    PrevNormalRoughnessBuffer,
    SkyBuffer,
    SunBuffer,
    DebugBuffer,
    UiBuffer,
    Buffer2DCount,
};

enum Buffer2DTextureMode
{
    NoTexture,
    PointFilteredTexture,
    LinearFilteredTexture,
};

struct Buffer2D
{
    Buffer2D() = default;
    ~Buffer2D();

    void init(int textureMode, const cudaChannelFormatDesc *pFormat, Int2 dim, unsigned int usageFlag);

    SurfObj buffer;
    cudaArray_t bufferArray;
    TexObj tex;
    cudaTextureDesc texDesc{};
    Int2 bufferDim;
};

class BufferManager
{
public:
    static BufferManager &Get()
    {
        static BufferManager instance;
        return instance;
    }
    BufferManager(BufferManager const &) = delete;
    void operator=(BufferManager const &) = delete;

    void init();

    Int2 GetBufferDim(Buffer2DName name) const { return m_buffers[(unsigned int)name].bufferDim; }
    SurfObj GetBuffer2D(Buffer2DName name) const { return m_buffers[(unsigned int)name].buffer; }
    TexObj GetTexture2D(Buffer2DName name) const { return m_buffers[(unsigned int)name].tex; }

private:
    std::vector<Buffer2D> m_buffers{};

    BufferManager() {}
};