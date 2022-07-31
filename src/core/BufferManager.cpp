#include "core/BufferManager.h"
#include "core/Backend.h"
#include "util/KernelHelper.h"
#include "util/DebugUtils.h"

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

void Buffer2D::clear()
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

    struct Buffer2DDesc
    {
        cudaChannelFormatDesc format;
        UInt2 dim;
    };

    std::unordered_map<Buffer2DName, Buffer2DDesc> map =
    {
        { RenderColorBuffer       , { cudaCreateChannelDescHalf4()     , bufferSize                          } } ,

        { MaterialBuffer          , { cudaCreateChannelDesc<ushort1>() , bufferSize                          } } ,
        { MaterialHistoryBuffer   , { cudaCreateChannelDesc<ushort1>() , bufferSize                          } } ,

        { AccumulationColorBuffer , { cudaCreateChannelDescHalf4()     , bufferSize                          } } ,
        { HistoryColorBuffer      , { cudaCreateChannelDescHalf4()     , bufferSize                          } } ,

        { NormalBuffer            , { cudaCreateChannelDescHalf4()     , bufferSize                          } } ,
        { DepthBuffer             , { cudaCreateChannelDescHalf1()     , bufferSize                          } } ,
        { HistoryDepthBuffer      , { cudaCreateChannelDescHalf1()     , bufferSize                          } } ,

        { MotionVectorBuffer      , { cudaCreateChannelDescHalf2()     , bufferSize                          } } ,

        { AlbedoBuffer            , { cudaCreateChannelDescHalf4()     , bufferSize                          } } ,
        { HistoryAlbedoBuffer     , { cudaCreateChannelDescHalf4()     , bufferSize                          } } ,

        { OutputColorBuffer       , { cudaCreateChannelDescHalf4()     , outputSize                          } } ,
    };

    assert(map.size() == Buffer2DCount);
    m_buffers.resize(Buffer2DCount);

    for (int i = 0; i < Buffer2DCount; ++i)
    {
        const Buffer2DDesc& desc = map[static_cast<Buffer2DName>(i)];
        m_buffers[i].init(&desc.format, desc.dim, cudaArraySurfaceLoadStore);
    }
}

}