#pragma once

#include "shaders/LinearMath.h"
#include "shaders/SystemParameter.h"
#include <cassert>
#include <cstdint>
#include <cstring>

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
        assert(x < width);
        assert(y < height);
        assert(z < width);

        return x + width * (z + height * y);
    }

    void clear()
    {
        memset(data, 0, width * width * height * sizeof(Voxel));
    }

    void set(const Voxel& v, uint x, uint y, uint z)
    {
        uint linearId = getLinearId(x, y, z);
        data[linearId] = v;
    }

    const Voxel& get(uint x, uint y, uint z)
    {
        uint linearId = getLinearId(x, y, z);
        return data[linearId];
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

class VoxelEngine
{
public:
    static VoxelEngine& Get()
    {
        static VoxelEngine instance;
        return instance;
    }
    VoxelEngine(VoxelEngine const&) = delete;
    void operator=(VoxelEngine const&) = delete;
    ~VoxelEngine();

    void init();
    void generateVoxels();

private:
    VoxelEngine() {}

    VoxelChunk data[1] = {};
};

}