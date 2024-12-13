#pragma once

#include "Voxel.h"

#include "VoxelMath.h"

#include <optional>
#include <iostream>

namespace vox
{

    /*
     *              -------------
     *              |\           \
     *              | \           \
     *              |  \           \
     *   height(y)  |   -------------
     *              |   |           |
     *              |   |           |
     *              |   |           |
     *              |   |           |
     *              |   |           |
     *               \  |           |
     *     width(z)   \ |           |
     *                 \|           |
     *                  -------------
     *                     width(x)
     */
    struct VoxelChunk
    {
        static const unsigned int width = 256;
        static const unsigned int height = 64;

        VoxelChunk()
        {
            cudaMallocManaged(&data, width * width * height * sizeof(Voxel));
        }

        ~VoxelChunk()
        {
            cudaFree(data);
        }

        void clear()
        {
            cudaMemset(data, 0, width * width * height * sizeof(Voxel));
        }

        Voxel *data;
    };

}