#pragma once

#include "BlockMesher.h"
#include "VoxelChunk.h"

#include "shaders/SystemParameter.h"
#include <cassert>
#include <cstring>
#include <functional>

namespace vox
{

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
    void update();
    void generateVoxels();

    VoxelChunk data[1] = {};
    bool leftMouseButtonClicked = false;

    // static std::function<void()> UpdateFunc;

private:
    VoxelEngine() {}

    std::vector<BlockMesher> blockMeshers;
};

static inline void UpdateFunc()
{
    VoxelEngine::Get().update();
}

}