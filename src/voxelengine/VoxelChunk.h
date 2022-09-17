#pragma once

#include "Voxel.h"

#include "shaders/LinearMath.h"

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
    static const uint width = 2;
    static const uint height = 2;

    uint getLinearId(uint x, uint y, uint z)
    {
        if (x < width && y < height && z < width)
        {
            return x + width * (z + height * y);
        }
        else
        {
            return UINT_MAX;
        }
    }

    void clear()
    {
        memset(data, 0, width * width * height * sizeof(Voxel));
    }

    bool inBound(uint x, uint y, uint z)
    {
        return getLinearId(x, y, z) != UINT_MAX;
    }

    void set(const Voxel& v, uint x, uint y, uint z)
    {
        uint linearId = getLinearId(x, y, z);
        if (linearId != UINT_MAX)
        {
            data[linearId] = v;
        }
        else
        {
            std::cout << "warning: set invalid voxel\n";
        }
    }

    std::optional<Voxel> get(uint x, uint y, uint z)
    {
        uint linearId = getLinearId(x, y, z);
        if (linearId != UINT_MAX)
        {
            return data[linearId];
        }
        else
        {
            return std::nullopt;
        }
    }

    template<typename Lambda>
    void foreach(Lambda&& func)
    {
        for (int y = 0; y < VoxelChunk::height; ++y)
        {
            for (int z = 0; z < VoxelChunk::width; ++z)
            {
                for (int x = 0; x < VoxelChunk::width; ++x)
                {
                    uint linearId = getLinearId(x, y, z);
                    func(data[linearId], x, y, z);
                }
            }
        }
    }

    Voxel data[width * width * height] = {};
};

}