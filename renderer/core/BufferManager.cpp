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

#define RTXDI_RESERVOIR_BLOCK_SIZE 16

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

BufferManager::~BufferManager()
{
    CUDA_CHECK(cudaFree(reservoirBuffer));
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
            {PrevNormalRoughnessBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, LinearFilteredTexture}},
            {GeoNormalThinfilmBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {MaterialParameterBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {PrevMaterialParameterBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {PrevGeoNormalThinfilmBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {PrevAlbedoBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {SkyBuffer, {cudaCreateChannelDesc<float4>(), skySize, cudaArraySurfaceLoadStore, NoTexture}},
            {SunBuffer, {cudaCreateChannelDesc<float4>(), sunSize, cudaArraySurfaceLoadStore, NoTexture}},
            {DebugBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
            {UIBuffer, {cudaCreateChannelDesc<float4>(), bufferSize, cudaArraySurfaceLoadStore, NoTexture}},
        };

    assert(map.size() == Buffer2DCount);
    m_buffers.resize(Buffer2DCount);

    for (int i = 0; i < Buffer2DCount; ++i)
    {
        const Buffer2DDesc &desc = map[static_cast<Buffer2DName>(i)];
        m_buffers[i].init(static_cast<Buffer2DName>(i), &desc.format, desc.dim, desc.usageFlag);
    }

    {
        uint32_t renderWidthBlocks = (renderWidth + RTXDI_RESERVOIR_BLOCK_SIZE - 1) / RTXDI_RESERVOIR_BLOCK_SIZE;
        uint32_t renderHeightBlocks = (renderHeight + RTXDI_RESERVOIR_BLOCK_SIZE - 1) / RTXDI_RESERVOIR_BLOCK_SIZE;
        reservoirBlockRowPitch = renderWidthBlocks * (RTXDI_RESERVOIR_BLOCK_SIZE * RTXDI_RESERVOIR_BLOCK_SIZE);
        reservoirArrayPitch = reservoirBlockRowPitch * renderHeightBlocks;

        // CUDA_CHECK(cudaMalloc(&reservoirBuffer, 2 * reservoirArrayPitch * sizeof(DIReservoir)));
        // CUDA_CHECK(cudaMemset(reservoirBuffer, 0, 2 * reservoirArrayPitch * sizeof(DIReservoir)));

        CUDA_CHECK(cudaMalloc(&reservoirBuffer, 2 * bufferSize.x * bufferSize.y * sizeof(DIReservoir)));
        CUDA_CHECK(cudaMemset(reservoirBuffer, 0, 2 * bufferSize.x * bufferSize.y * sizeof(DIReservoir)));
    }

    const uint32_t neighborOffsetCount = 8192;
    std::vector<uint8_t> offsets;
    offsets.resize(neighborOffsetCount * 2);
    {
        uint8_t *buffer = offsets.data();
        // Create a sequence of low-discrepancy samples within a unit radius around the origin
        // for "randomly" sampling neighbors during spatial resampling
        int R = 250;
        const float phi2 = 1.0f / 1.3247179572447f;
        uint32_t num = 0;
        float u = 0.5f;
        float v = 0.5f;
        while (num < neighborOffsetCount * 2)
        {
            u += phi2;
            v += phi2 * phi2;
            if (u >= 1.0f)
                u -= 1.0f;
            if (v >= 1.0f)
                v -= 1.0f;

            float rSq = (u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f);
            if (rSq > 0.25f)
                continue;

            buffer[num++] = int8_t((u - 0.5f) * R);
            buffer[num++] = int8_t((v - 0.5f) * R);
        }
    }
    CUDA_CHECK(cudaMalloc(&neighborOffsetBuffer, 2 * neighborOffsetCount * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(neighborOffsetBuffer, offsets.data(), 2 * neighborOffsetCount * sizeof(uint8_t), cudaMemcpyHostToDevice));
}
