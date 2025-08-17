#pragma once

#include "VoxelSceneGen.h"
#include "VoxelChunk.h"

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

    // Center block information for GUI display
    struct CenterBlockInfo {
        int blockId = 0;
        std::string blockName = "Empty";
        Int3 position = Int3(-1, -1, -1);
        bool hasValidBlock = false;
    } centerBlockInfo;

    // Chunk-specific face tracking buffers
    // Structure: [chunkIndex][objectId]
    std::vector<std::vector<unsigned int>> currentFaceCount;
    std::vector<std::vector<unsigned int>> maxFaceCount;
    std::vector<std::vector<std::vector<unsigned int>>> freeFaces;
    std::vector<std::vector<std::vector<unsigned int>>> faceLocation;

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

    // Public block manipulation functions
    void deleteBlock(const Int3& pos, int blockId);
    void addBlock(const Int3& pos, int blockId);

private:
    // Refactored update functions
    struct RayTraversalResult {
        bool hasSpaceToCreate = false;
        bool hitSurface = false;
        Int3 createPos = Int3(-1, -1, -1);
        Int3 deletePos = Int3(-1, -1, -1);
        int deleteBlockId = -1;
        int hitX = -1, hitY = -1, hitZ = -1;
    };
    
    RayTraversalResult performRayTraversal();
    
    void deleteInstancedBlock(const Int3& pos, int blockId);
    void deleteUninstancedBlock(const Int3& pos, int blockId);
    void addInstancedBlock(const Int3& pos, int blockId);
    void addUninstancedBlock(const Int3& pos, int blockId);
    
    void updateSceneForBlock(int objectId, unsigned int instanceId, unsigned int chunkIndex = 0);
    void initInstanceGeometry();
    void updateInstances();
    void updateUninstancedMeshes(const std::vector<Voxel*> &d_dataChunks);

    VoxelEngine()
    {
    }
};