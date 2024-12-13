#include "core/BufferManager.h"
#include "core/Backend.h"
#include "util/KernelHelper.h"
#include "util/DebugUtils.h"
#include "sky/Sky.h"

#include <unordered_map>

#define SKY_WIDTH 512
#define SKY_HEIGHT 256
#define SKY_SIZE (SKY_WIDTH * SKY_HEIGHT)
#define SKY_SCAN_BLOCK_SIZE 256
#define SKY_SCAN_BLOCK_COUNT (SKY_SIZE / SKY_SCAN_BLOCK_SIZE)

#define SUN_WIDTH 32
#define SUN_HEIGHT 32
#define SUN_SIZE (SUN_WIDTH * SUN_HEIGHT)
#define SUN_SCAN_BLOCK_SIZE 32
#define SUN_SCAN_BLOCK_COUNT (SUN_SIZE / SUN_SCAN_BLOCK_SIZE)

namespace jazzfusion
{

void Buffer2D::init(const cudaChannelFormatDesc* pFormat,
    UInt2                  dim,
    uint                   usageFlag)
{
    CUDA_CHECK(cudaMallocArray(&bufferArray, pFormat, dim.x, dim.y, usageFlag));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = bufferArray;
    CUDA_CHECK(cudaCreateSurfaceObject(&buffer, &resDesc));
}

Buffer2D::~Buffer2D()
{
    CUDA_CHECK(cudaDestroySurfaceObject(buffer));
    if (bufferArray != nullptr)
    {
        CUDA_CHECK(cudaFreeArray(bufferArray));
        bufferArray = nullptr;
    }
}

void BufferManager::init()
{
    auto& backend = Backend::Get();
    uint renderWidth = backend.getMaxRenderWidth();
    uint renderHeight = backend.getMaxRenderHeight();

    UInt2 bufferSize = UInt2(renderWidth, renderHeight);
    UInt2 bufferSize4 = UInt2(DivRoundUp(renderWidth, 4u), DivRoundUp(renderHeight, 4u));
    UInt2 bufferSize16 = UInt2(DivRoundUp(bufferSize4.x, 4u), DivRoundUp(bufferSize4.y, 4u));
    UInt2 bufferSize64 = UInt2(DivRoundUp(bufferSize16.x, 4u), DivRoundUp(bufferSize16.y, 4u));
    UInt2 bufferSize8x8 = UInt2(DivRoundUp(renderWidth, 8u), DivRoundUp(renderHeight, 8u));
    UInt2 bufferSize16x16 = UInt2(DivRoundUp(renderWidth, 16u), DivRoundUp(renderHeight, 16u));

    UInt2 outputSize = UInt2(backend.getWidth(), backend.getHeight());

    const auto& skyModel = SkyModel::Get();
    UInt2 skySize = (UInt2)skyModel.getSkyRes();
    UInt2 sunSize = (UInt2)skyModel.getSunRes();

    struct Buffer2DDesc
    {
        cudaChannelFormatDesc format;
        UInt2 dim;
        uint usageFlag;
    };

    std::unordered_map<Buffer2DName, Buffer2DDesc> map =
    {
        { RenderColorBuffer       , { cudaCreateChannelDescHalf4()     , bufferSize, cudaArraySurfaceLoadStore } } ,

        { MaterialBuffer          , { cudaCreateChannelDesc<ushort1>() , bufferSize, cudaArraySurfaceLoadStore } } ,
        { MaterialHistoryBuffer   , { cudaCreateChannelDesc<ushort1>() , bufferSize, cudaArraySurfaceLoadStore } } ,

        { AccumulationColorBuffer , { cudaCreateChannelDescHalf4()     , bufferSize, cudaArraySurfaceLoadStore } } ,
        { HistoryColorBuffer      , { cudaCreateChannelDescHalf4()     , bufferSize, cudaArraySurfaceLoadStore } } ,

        { NormalBuffer            , { cudaCreateChannelDescHalf4()     , bufferSize, cudaArraySurfaceLoadStore } } ,
        { DepthBuffer             , { cudaCreateChannelDescHalf1()     , bufferSize, cudaArraySurfaceLoadStore } } ,
        { HistoryDepthBuffer      , { cudaCreateChannelDescHalf1()     , bufferSize, cudaArraySurfaceLoadStore } } ,

        { MotionVectorBuffer      , { cudaCreateChannelDescHalf2()     , bufferSize, cudaArraySurfaceLoadStore } } ,

        { AlbedoBuffer            , { cudaCreateChannelDescHalf4()     , bufferSize, cudaArraySurfaceLoadStore } } ,
        { HistoryAlbedoBuffer     , { cudaCreateChannelDescHalf4()     , bufferSize, cudaArraySurfaceLoadStore } } ,

        { OutputColorBuffer       , { cudaCreateChannelDescHalf4()     , outputSize, cudaArraySurfaceLoadStore } } ,

        { OutputColorBuffer       , { cudaCreateChannelDescHalf4()     , outputSize, cudaArraySurfaceLoadStore } } ,

        { SkyBuffer               , { cudaCreateChannelDescHalf4()     , skySize   , cudaArraySurfaceLoadStore } } ,
        { SunBuffer               , { cudaCreateChannelDescHalf4()     , sunSize   , cudaArraySurfaceLoadStore } } ,
    };

    assert(map.size() == Buffer2DCount);
    m_buffers.resize(Buffer2DCount);

    for (int i = 0; i < Buffer2DCount; ++i)
    {
        const Buffer2DDesc& desc = map[static_cast<Buffer2DName>(i)];
        m_buffers[i].init(&desc.format, desc.dim, desc.usageFlag);
    }

    // Code for creating a gather enabled texture
    // {
    //     cudaResourceDesc resDesc = {};
    //     resDesc.resType = cudaResourceTypeArray;
    //     resDesc.res.array.array = m_buffers[static_cast<uint>(RenderColorBuffer)].bufferArray;

    //     memset(&m_renderBufferTexDesc, 0, sizeof(cudaTextureDesc));

    //     m_renderBufferTexDesc.addressMode[0] = cudaAddressModeClamp;
    //     m_renderBufferTexDesc.addressMode[1] = cudaAddressModeClamp;
    //     m_renderBufferTexDesc.addressMode[2] = cudaAddressModeClamp;
    //     m_renderBufferTexDesc.filterMode = cudaFilterModePoint;
    //     m_renderBufferTexDesc.readMode = cudaReadModeElementType;
    //     m_renderBufferTexDesc.sRGB = 0;
    //     m_renderBufferTexDesc.borderColor[0] = 0.0f;
    //     m_renderBufferTexDesc.borderColor[1] = 0.0f;
    //     m_renderBufferTexDesc.borderColor[2] = 0.0f;
    //     m_renderBufferTexDesc.borderColor[3] = 0.0f;
    //     m_renderBufferTexDesc.normalizedCoords = 1;
    //     m_renderBufferTexDesc.maxAnisotropy = 1;
    //     m_renderBufferTexDesc.mipmapFilterMode = cudaFilterModePoint;
    //     m_renderBufferTexDesc.mipmapLevelBias = 0.0f;
    //     m_renderBufferTexDesc.minMipmapLevelClamp = 0.0f;
    //     m_renderBufferTexDesc.maxMipmapLevelClamp = 0.0f;

    //     CUDA_CHECK(cudaCreateTextureObject(&m_renderBufferTexture, &resDesc, &m_renderBufferTexDesc, nullptr));
    // }
}

}