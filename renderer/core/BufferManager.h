#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "shaders/LinearMath.h"

namespace jazzfusion
{

    enum Buffer2DName
    {
        IlluminationBuffer,
        IlluminationPingBuffer,
        NormalRoughnessBuffer,
        DepthBuffer,
        MaterialBuffer,
        AlbedoBuffer,
        MotionVectorBuffer,
        OutputColorBuffer,
        SkyBuffer,
        SunBuffer,
        Buffer2DCount,
    };

    struct Buffer2D
    {
        Buffer2D() = default;
        ~Buffer2D();

        void init(const cudaChannelFormatDesc *pFormat, UInt2 dim, unsigned int usageFlag);
        void clear();

        SurfObj buffer;
        cudaArray_t bufferArray;
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

        SurfObj GetBuffer2D(Buffer2DName name) const { return m_buffers[(unsigned int)name].buffer; }
        // TexObj GetRenderBufferTexture() const { return m_renderBufferTexture; }

    private:
        std::vector<Buffer2D> m_buffers{};

        // cudaTextureDesc m_renderBufferTexDesc{};
        // TexObj m_renderBufferTexture{};

        BufferManager() {}
    };

}