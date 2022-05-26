#pragma once

#include <cuda_runtime.h>
#include "shaders/Common.h"

namespace jazzfusion
{

enum BlockDimType
{
    BLOCK_DIM_8x8x1,
};

INL_HOST_DEVICE unsigned int DivRoundUp(unsigned int dividend, unsigned int divisor) { return (dividend + divisor - 1) / divisor; }

INL_HOST_DEVICE dim3 GetBlockDim(BlockDimType blockDimType = BLOCK_DIM_8x8x1)
{
    switch (blockDimType)
    {
    case BLOCK_DIM_8x8x1: return dim3(8, 8, 1);
    default: return dim3(8, 8, 1);
    }
}

INL_HOST_DEVICE dim3 GetGridDim(int width, int height, BlockDimType blockDimType = BLOCK_DIM_8x8x1)
{
    dim3 blockDim = GetBlockDim(blockDimType);
    return { DivRoundUp(width, blockDim.x), DivRoundUp(height, blockDim.y), 1 };
}

}