#include "util/BufferUtils.h"

#include "util/KernelHelper.h"
#include "shaders/LinearMath.h"
#include "shaders/Sampler.h"

__global__ void BufferSetFloat1_impl(Int2 bufferDim, SurfObj outBuffer, float val)
{
    Int2 threadPos;
    threadPos.x = threadIdx.x;
    threadPos.y = threadIdx.y;

    Int2 pixelPos;
    pixelPos.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelPos.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelPos.x >= bufferDim.x || pixelPos.y >= bufferDim.y)
    {
        return;
    }

    Store2DFloat1(val, outBuffer, pixelPos);
}

void BufferSetFloat1(Int2 bufferDim, SurfObj outBuffer, float val)
{
    BufferSetFloat1_impl KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
        bufferDim,
        outBuffer,
        val);
}

__global__ void BufferSetFloat4_impl(Int2 bufferDim, SurfObj outBuffer, Float4 val)
{
    Int2 threadPos;
    threadPos.x = threadIdx.x;
    threadPos.y = threadIdx.y;

    Int2 pixelPos;
    pixelPos.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelPos.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelPos.x >= bufferDim.x || pixelPos.y >= bufferDim.y)
    {
        return;
    }

    Store2DFloat4(val, outBuffer, pixelPos);
}

void BufferSetFloat4(Int2 bufferDim, SurfObj outBuffer, Float4 val)
{
    BufferSetFloat4_impl KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
        bufferDim,
        outBuffer,
        val);
}

__global__ void BufferCopyFloat1_impl(
    Int2 bufferDim,
    SurfObj inBuffer,
    SurfObj outBuffer)
{
    Int2 threadPos;
    threadPos.x = threadIdx.x;
    threadPos.y = threadIdx.y;

    Int2 pixelPos;
    pixelPos.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelPos.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelPos.x >= bufferDim.x || pixelPos.y >= bufferDim.y)
    {
        return;
    }

    float val = Load2DFloat1(inBuffer, pixelPos);
    Store2DFloat1(val, outBuffer, pixelPos);
}

void BufferCopyFloat1(
    Int2 bufferDim,
    SurfObj inBuffer,
    SurfObj outBuffer)
{
    BufferCopyFloat1_impl KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
        bufferDim,
        inBuffer,
        outBuffer);
}

__global__ void BufferCopyFloat4_impl(
    Int2 bufferDim,
    SurfObj inBuffer,
    SurfObj outBuffer)
{
    Int2 threadPos;
    threadPos.x = threadIdx.x;
    threadPos.y = threadIdx.y;

    Int2 pixelPos;
    pixelPos.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelPos.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelPos.x >= bufferDim.x || pixelPos.y >= bufferDim.y)
    {
        return;
    }

    Float4 val = Load2DFloat4(inBuffer, pixelPos);
    Store2DFloat4(val, outBuffer, pixelPos);
}

void BufferCopyFloat4(
    Int2 bufferDim,
    SurfObj inBuffer,
    SurfObj outBuffer)
{
    BufferCopyFloat4_impl KERNEL_ARGS2(GetGridDim(bufferDim.x, bufferDim.y, BLOCK_DIM_8x8x1), GetBlockDim(BLOCK_DIM_8x8x1))(
        bufferDim,
        inBuffer,
        outBuffer);
}