#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "shaders/LinearMath.h"
#include "shaders/RestirCommon.h"

enum Buffer2DName
{
    IlluminationBuffer,
    IlluminationOutputBuffer,
    IlluminationPingBuffer,
    IlluminationPongBuffer,
    DiffuseIlluminationBuffer,
    SpecularIlluminationBuffer,
    
    // Separate diffuse/specular ping-pong buffers for temporal accumulation
    DiffuseIlluminationPingBuffer,
    DiffuseIlluminationPongBuffer,
    SpecularIlluminationPingBuffer,
    SpecularIlluminationPongBuffer,
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
    
    // Separate previous frame buffers for diffuse/specular
    PrevDiffuseIlluminationBuffer,
    PrevDiffuseFastIlluminationBuffer,
    PrevSpecularIlluminationBuffer,
    PrevSpecularFastIlluminationBuffer,
    PrevHistoryLengthBuffer,
    PrevNormalRoughnessBuffer,
    
    // Specular-specific buffers
    SpecularHitDistBuffer,
    PrevSpecularHitDistBuffer,
    SpecularReprojectionConfidenceBuffer,
    
    // History confidence buffers
    DiffuseHistoryConfidenceBuffer,
    PrevDiffuseHistoryConfidenceBuffer,
    SpecularHistoryConfidenceBuffer,
    PrevSpecularHistoryConfidenceBuffer,
    DisocclusionThresholdMixBuffer,
    GeoNormalThinfilmBuffer,
    MaterialParameterBuffer,
    PrevMaterialParameterBuffer,
    PrevGeoNormalThinfilmBuffer,
    PrevAlbedoBuffer,
    SkyBuffer,
    SunBuffer,
    DebugBuffer,
    UIBuffer,
    
    // Post-processing pipeline buffers
    BloomExtractBuffer,
    BloomTempBuffer,
    LuminanceMip0Buffer,
    LuminanceMip1Buffer,
    LuminanceMip2Buffer,
    LuminanceMip3Buffer,
    LuminanceMip4Buffer,
    LuminanceMip5Buffer,
    
    // Confidence computation buffers  
    DiffuseGradientBuffer,
    FilteredDiffuseGradientBuffer,
    DiffuseConfidenceBuffer,
    PrevDiffuseConfidenceBuffer,
    SpecularGradientBuffer,
    FilteredSpecularGradientBuffer,
    SpecularConfidenceBuffer,
    PrevSpecularConfidenceBuffer,
    
    // ReSTIR luminance buffers
    RestirLuminanceBuffer,
    PrevRestirLuminanceBuffer,
    
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
    ~BufferManager();

    void init();

    Int2 GetBufferDim(Buffer2DName name) const { return m_buffers[(unsigned int)name].bufferDim; }
    SurfObj GetBuffer2D(Buffer2DName name) const { return m_buffers[(unsigned int)name].buffer; }
    TexObj GetTexture2D(Buffer2DName name) const { return m_buffers[(unsigned int)name].tex; }
    cudaArray_t GetBufferArray(Buffer2DName name) const { return m_buffers[(unsigned int)name].bufferArray; }

    uint32_t reservoirBlockRowPitch;
    uint32_t reservoirArrayPitch;
    DIReservoir *reservoirBuffer;
    uint8_t *neighborOffsetBuffer;

private:
    std::vector<Buffer2D> m_buffers{};

    BufferManager() {}
};