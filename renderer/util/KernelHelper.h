#pragma once

#include <cuda_runtime.h>
#include "shaders/Common.h"

namespace jazzfusion
{

    enum BlockDimType
    {
        BLOCK_DIM_8x8x1,
        BLOCK_DIM_8x4x1,
        BLOCK_DIM_16x16x1,
        BLOCK_DIM_4x4x4,
    };

    INL_HOST_DEVICE unsigned int DivRoundUp(unsigned int dividend, unsigned int divisor) { return (dividend + divisor - 1) / divisor; }

    INL_HOST_DEVICE dim3 GetBlockDim(BlockDimType blockDimType = BLOCK_DIM_8x8x1)
    {
        switch (blockDimType)
        {
        case BLOCK_DIM_8x8x1:
            return dim3(8, 8, 1);
        case BLOCK_DIM_8x4x1:
            return dim3(8, 4, 1);
        case BLOCK_DIM_16x16x1:
            return dim3(16, 16, 1);
        case BLOCK_DIM_4x4x4:
            return dim3(4, 4, 4);
        default:
            return dim3(8, 8, 1);
        }
    }

    INL_HOST_DEVICE dim3 GetGridDim(int width, int height, BlockDimType blockDimType = BLOCK_DIM_8x8x1)
    {
        dim3 blockDim = GetBlockDim(blockDimType);
        return {DivRoundUp(width, blockDim.x), DivRoundUp(height, blockDim.y), 1};
    }

    INL_HOST_DEVICE dim3 GetGridDim(int width, int height, int depth, BlockDimType blockDimType = BLOCK_DIM_4x4x4)
    {
        dim3 blockDim = GetBlockDim(blockDimType);
        return {DivRoundUp(width, blockDim.x), DivRoundUp(height, blockDim.y), DivRoundUp(depth, blockDim.z)};
    }

}