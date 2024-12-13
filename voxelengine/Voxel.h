#pragma once

#include <cstdint>

namespace vox
{

using VoxelDataType = uint8_t;

struct Voxel
{
    Voxel() {}
    ~Voxel() {}

    Voxel(VoxelDataType id) : id{ id } {}

    union
    {
        struct
        {
            VoxelDataType id : 1;
            VoxelDataType unused : 7;
        };

        VoxelDataType allBits;
    };
};

}