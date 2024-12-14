#pragma once

#include "Voxel.h"

#include "VoxelMath.h"

#include <optional>
#include <iostream>

namespace vox
{
    struct VoxelChunk
    {
        static const unsigned int width = 128;

        VoxelChunk()
        {
            cudaMallocManaged(&data, width * width * width * sizeof(Voxel));
        }

        ~VoxelChunk()
        {
            cudaFree(data);
        }

        void clear()
        {
            cudaMemset(data, 0, width * width * width * sizeof(Voxel));
        }

        Voxel get(unsigned int x, unsigned int y, unsigned int z)
        {
            return data[GetLinearId(x, y, z, width)];
        }

        void set(unsigned int x, unsigned int y, unsigned int z, unsigned int id)
        {
            data[GetLinearId(x, y, z, width)].id = id;
        }

        Voxel *data;
    };

}