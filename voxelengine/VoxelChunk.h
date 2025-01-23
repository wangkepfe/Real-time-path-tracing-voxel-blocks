#pragma once

#include "Voxel.h"

#include "VoxelMath.h"

#include "Block.h"

#include <optional>
#include <iostream>

namespace vox
{
    struct VoxelChunk
    {
        static const unsigned int width = 32;

        VoxelChunk()
        {
            // cudaMallocManaged(&data, width * width * width * sizeof(Voxel));
            data = new Voxel[width * width * width];
        }

        ~VoxelChunk()
        {
            // cudaFree(data);
            delete[] data;
        }

        void clear()
        {
            // cudaMemset(data, 0, width * width * width * sizeof(Voxel));
            memset(data, 0, width * width * width * sizeof(Voxel));
        }

        unsigned int size()
        {
            return width * width * width * sizeof(Voxel);
        }

        Voxel get(unsigned int x, unsigned int y, unsigned int z)
        {
            return data[GetLinearId(x, y, z, width)];
        }

        void set(unsigned int x, unsigned int y, unsigned int z, unsigned int id)
        {
            data[GetLinearId(x, y, z, width)].id = id;
        }

        Voxel get(jazzfusion::Float3 pos)
        {
            if (pos.y <= 8)
            {
                return Voxel{BlockTypeWater};
            }
            return data[GetLinearId((unsigned int)pos.x, (unsigned int)pos.y, (unsigned int)pos.z, width)];
        }

        Voxel *data;
    };

}