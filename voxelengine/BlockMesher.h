#pragma once

#include "VoxelChunk.h"

#include "shaders/LinearMath.h"
#include "shaders/SystemParameter.h"

#include <cassert>
#include <unordered_set>
#include <cstdint>

namespace vox
{

static constexpr uint8_t AxisX = 0;
static constexpr uint8_t AxisY = 1;
static constexpr uint8_t AxisZ = 2;

struct QuadFace
{
    union
    {
        struct
        {
            uint8_t x;
            uint8_t y;
            uint8_t z;
            uint8_t axis;
        };

        uint32_t allbits;
    };
};

struct QuadFaceHasher
{
    unsigned long long operator() (const QuadFace& v) const
    {
        return v.allbits;
    }
};

struct QuadFaceEqualOperator
{
    bool operator() (const QuadFace& a, const QuadFace& b) const
    {
        return a.allbits == b.allbits;
    }
};

class BlockMesher
{
public:
    BlockMesher(
        VoxelChunk& voxelChunk,
        std::vector<jazzfusion::VertexAttributes>& attributes,
        std::vector<uint>& indices)
        : voxelChunk{ voxelChunk },
        attributes{ attributes },
        indices{ indices }
    {}
    ~BlockMesher() {}

    void process();
    void update(const Voxel& voxel, int x, int y, int z);

private:

    void addVoxel(const Voxel& voxel, uint8_t x, uint8_t y, uint8_t z);
    void facesToMesh();

    VoxelChunk& voxelChunk;

    // Mesh
    std::vector<jazzfusion::VertexAttributes>& attributes;
    std::vector<uint>& indices;

    std::unordered_set<QuadFace, QuadFaceHasher, QuadFaceEqualOperator> faces;
};

}