#include "core/BufferManager.h"
#include "core/Backend.h"
#include "util/KernelHelper.h"
#include "util/DebugUtils.h"
#include "sky/Sky.h"

#include <unordered_map>
#include "BufferManager.h"

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

void Buffer2D::init(int textureMode,
                    const cudaChannelFormatDesc *pFormat,
                    Int2 dim,
                    unsigned int usageFlag)
{
    bufferDim = dim;

    CUDA_CHECK(cudaMallocArray(&bufferArray, pFormat, dim.x, dim.y, usageFlag));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = bufferArray;

    // Create buffer obj
    {
        CUDA_CHECK(cudaCreateSurfaceObject(&buffer, &resDesc));
    }

    if (textureMode == PointFilteredTexture) // with point filter
    {
        memset(&texDesc, 0, sizeof(cudaTextureDesc));

        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.addressMode[2] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.sRGB = 0;
        texDesc.borderColor[0] = 0.0f;
        texDesc.borderColor[1] = 0.0f;
        texDesc.borderColor[2] = 0.0f;
        texDesc.borderColor[3] = 0.0f;
        texDesc.normalizedCoords = 1;
        texDesc.maxAnisotropy = 1;
        texDesc.mipmapFilterMode = cudaFilterModePoint;
        texDesc.mipmapLevelBias = 0.0f;
        texDesc.minMipmapLevelClamp = 0.0f;
        texDesc.maxMipmapLevelClamp = 0.0f;

        CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
    }
    else if (textureMode == LinearFilteredTexture) // Create linear filter texture
    {
        memset(&texDesc, 0, sizeof(cudaTextureDesc));

        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.addressMode[2] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.sRGB = 0;
        texDesc.borderColor[0] = 0.0f;
        texDesc.borderColor[1] = 0.0f;
        texDesc.borderColor[2] = 0.0f;
        texDesc.borderColor[3] = 0.0f;
        texDesc.normalizedCoords = 1;
        texDesc.maxAnisotropy = 1;
        texDesc.mipmapFilterMode = cudaFilterModePoint;
        texDesc.mipmapLevelBias = 0.0f;
        texDesc.minMipmapLevelClamp = 0.0f;
        texDesc.maxMipmapLevelClamp = 0.0f;

        CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
    }
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
    auto &backend = Backend::Get();
    unsigned int renderWidth = backend.getMaxRenderWidth();
    unsigned int renderHeight = backend.getMaxRenderHeight();

    Int2 bufferSize = Int2(renderWidth, renderHeight);
    Int2 bufferSize4 = Int2(DivRoundUp(renderWidth, 4u), DivRoundUp(renderHeight, 4u));
    Int2 bufferSize16 = Int2(DivRoundUp(bufferSize4.x, 4u), DivRoundUp(bufferSize4.y, 4u));
    Int2 bufferSize64 = Int2(DivRoundUp(bufferSize16.x, 4u), DivRoundUp(bufferSize16.y, 4u));
    Int2 bufferSize8x8 = Int2(DivRoundUp(renderWidth, 8u), DivRoundUp(renderHeight, 8u));
    Int2 bufferSize16x16 = Int2(DivRoundUp(renderWidth, 16u), DivRoundUp(renderHeight, 16u));

    Int2 outputSize = Int2(backend.getWidth(), backend.getHeight());

    const auto &skyModel = SkyModel::Get();
    Int2 skySize = (Int2)skyModel.getSkyRes();
    Int2 sunSize = (Int2)skyModel.getSunRes();

    struct Buffer2DDesc
    {
        cudaChannelFormatDesc format;
        Int2 dim;
        unsigned int usageFlag;
        int textureMode;
    };

    std::unordered_map<Buffer2DName, Buffer2DDesc> map =
        {
            {IlluminationBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {IlluminationOutputBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {IlluminationPingBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {IlluminationPongBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {NormalRoughnessBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {DepthBuffer, {cudaCreateChannelDesc<float>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {MaterialBuffer, {cudaCreateChannelDesc<float>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {AlbedoBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {MotionVectorBuffer, {cudaCreateChannelDesc<float2>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {OutputColorBuffer, {cudaCreateChannelDesc<float4>(), outputSize, cudaArraySurfaceLoadStore, NoTexture}},
            {HistoryLengthBuffer, {cudaCreateChannelDesc<float>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {PrevDepthBuffer, {cudaCreateChannelDesc<float>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {PrevMaterialBuffer, {cudaCreateChannelDesc<float>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {PrevIlluminationBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, LinearFilteredTexture}},
            {PrevFastIlluminationBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, LinearFilteredTexture}},
            {PrevHistoryLengthBuffer, {cudaCreateChannelDesc<float>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {PrevNormalRoughnessBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, LinearFilteredTexture}},
            {SkyBuffer, {cudaCreateChannelDesc<float4>(), skySize, cudaArraySurfaceLoadStore, NoTexture}},
            {SunBuffer, {cudaCreateChannelDesc<float4>(), sunSize, cudaArraySurfaceLoadStore, NoTexture}},
            {DebugBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {UiBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
        };

    assert(map.size() == Buffer2DCount);
    m_buffers.resize(Buffer2DCount);

    for (int i = 0; i < Buffer2DCount; ++i)
    {
        const Buffer2DDesc &desc = map[static_cast<Buffer2DName>(i)];
        m_buffers[i].init(static_cast<Buffer2DName>(i), &desc.format, desc.dim, desc.usageFlag);
    }
}