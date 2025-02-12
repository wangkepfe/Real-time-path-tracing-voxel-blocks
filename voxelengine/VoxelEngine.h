#pragma once

#include "VoxelSceneGen.h"
#include "VoxelChunk.h"

#include "shaders/SystemParameter.h"
#include <cassert>
#include <cstring>
#include <functional>

class VoxelEngine
{
public:
    static VoxelEngine &Get()
    {
        static VoxelEngine instance;
        return instance;
    }
    VoxelEngine(VoxelEngine const &) = delete;
    void operator=(VoxelEngine const &) = delete;
    ~VoxelEngine();

    void init();
    void update();
    void generateVoxels();
    void reload();

    int totalNumBlockTypes;
    int totalNumUninstancedGeometries;
    int totalNumInstancedGeometries;
    int totalNumGeometries;

    VoxelChunk voxelChunk;
    bool leftMouseButtonClicked = false;

    std::vector<unsigned int> currentFaceCount;
    std::vector<unsigned int> maxFaceCount;
    std::vector<std::vector<unsigned int>> freeFaces;
    std::vector<std::vector<unsigned int>> faceLocation;

private:
    void initInstanceGeometry();
    void updateInstances();
    void updateUninstancedMeshes(Voxel *d_data);

    VoxelEngine()
    {
    }
};