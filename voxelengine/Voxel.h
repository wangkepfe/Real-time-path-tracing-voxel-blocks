#pragma once

#include <cstdint>
#include "shaders/LinearMath.h"

using VoxelDataType = uint8_t;

struct Voxel
{
    INL_HOST_DEVICE Voxel() {}
    INL_HOST_DEVICE ~Voxel() {}

    INL_HOST_DEVICE Voxel(VoxelDataType id) : id{id} {}

    union
    {
        struct
        {
            VoxelDataType id : 8;
        };

        VoxelDataType allBits;
    };
};