#pragma once

#include "VoxelSceneGen.h"
#include "VoxelChunk.h"
#include "ChunkGeometryBuffer.h"

#include "shaders/SystemParameter.h"
#include <cassert>
#include <cstring>
#include <functional>
#include <vector>

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
    void initEntities();

    int totalNumUninstancedGeometries;
    int totalNumInstancedGeometries;
    int totalNumGeometries;

    // Multi-chunk support (ChunkConfiguration is defined in VoxelSceneGen.h)
    ChunkConfiguration chunkConfig;
    std::vector<VoxelChunk> voxelChunks;

    bool leftMouseButtonClicked = false;

    // Professional geometry buffer management
    ChunkGeometryManager geometryManager;

    // Legacy face tracking (still needed for updateSingleVoxelGlobal compatibility)
    std::vector<std::vector<std::vector<unsigned int>>> faceLocation;
    std::vector<std::vector<std::vector<unsigned int>>> freeFaces;

    // Helper functions for coordinate conversion
    unsigned int getChunkIndex(unsigned int chunkX, unsigned int chunkY, unsigned int chunkZ) const;
    void globalToChunkCoords(unsigned int globalX, unsigned int globalY, unsigned int globalZ,
                           unsigned int &chunkX, unsigned int &chunkY, unsigned int &chunkZ,
                           unsigned int &localX, unsigned int &localY, unsigned int &localZ) const;
    void chunkToGlobalCoords(unsigned int chunkX, unsigned int chunkY, unsigned int chunkZ,
                           unsigned int localX, unsigned int localY, unsigned int localZ,
                           unsigned int &globalX, unsigned int &globalY, unsigned int &globalZ) const;

    // Helper to get voxel at global coordinates
    Voxel getVoxelAtGlobal(unsigned int globalX, unsigned int globalY, unsigned int globalZ) const;
    void setVoxelAtGlobal(unsigned int globalX, unsigned int globalY, unsigned int globalZ, unsigned int blockId);

private:
    void initInstanceGeometry();
    void updateInstances();
    void updateUninstancedMeshes(const std::vector<Voxel*> &d_dataChunks);
    
    // UNIFIED BLOCK PLACEMENT SYSTEM
    void placeUninstancedBlock(const Int3& pos, unsigned int blockId, int objectId, Scene& scene);
    void placeInstancedBlock(const Int3& pos, unsigned int blockId, int objectId, unsigned int originalBlockId, unsigned int deleteBlockId, Scene& scene);
    void removeUninstancedBlock(const Int3& pos, unsigned int blockId, int objectId, Scene& scene);
    void removeInstancedBlock(const Int3& pos, unsigned int blockId, int objectId, Scene& scene);

    VoxelEngine()
    {
    }
};