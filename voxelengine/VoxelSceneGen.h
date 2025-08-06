#pragma once

#include "VoxelChunk.h"
#include "core/Scene.h"
#include "shaders/SystemParameter.h"

#include <vector>

// Chunk configuration structure to avoid circular dependencies
struct ChunkConfiguration
{
    unsigned int chunksX = 2;
    unsigned int chunksY = 1;
    unsigned int chunksZ = 2;

    unsigned int getTotalChunks() const { return chunksX * chunksY * chunksZ; }
    unsigned int getGlobalWidth() const { return chunksX * VoxelChunk::width; }
    unsigned int getGlobalHeight() const { return chunksY * VoxelChunk::width; }
    unsigned int getGlobalDepth() const { return chunksZ * VoxelChunk::width; }
};

// Forward declaration for VoxelEngine
class VoxelEngine;

// Multi-chunk voxel initialization
void initVoxelsMultiChunk(VoxelChunk &voxelChunk, Voxel **d_data, unsigned int chunkIndex,
                          const ChunkConfiguration &chunkConfig);

void generateMesh(VertexAttributes **attr,
                  unsigned int **indices,
                  std::vector<unsigned int> &faceLocation,
                  unsigned int &attrSize,
                  unsigned int &indicesSize,
                  unsigned int &currentFaceCount,
                  unsigned int &maxFaceCount,
                  VoxelChunk &voxelChunk,
                  Voxel *d_data,
                  int id);

void freeDeviceVoxelData(Voxel *d_data);

void updateSingleVoxel(
    unsigned int x,
    unsigned int y,
    unsigned int z,
    unsigned int newVal,
    VoxelChunk &voxelChunk,
    VertexAttributes *attr,
    unsigned int *indices,
    std::vector<unsigned int> &faceLocation,
    unsigned int &attrSize,
    unsigned int &indicesSize,
    unsigned int &currentFaceCount,
    unsigned int &maxFaceCount,
    std::vector<unsigned int> &freeFaces);

// Multi-chunk version for global coordinates
void updateSingleVoxelGlobal(
    unsigned int globalX,
    unsigned int globalY,
    unsigned int globalZ,
    unsigned int newVal,
    std::vector<VoxelChunk> &voxelChunks,
    const ChunkConfiguration &chunkConfig,
    VertexAttributes *attr,
    unsigned int *indices,
    std::vector<unsigned int> &faceLocation,
    unsigned int &attrSize,
    unsigned int &indicesSize,
    unsigned int &currentFaceCount,
    unsigned int &maxFaceCount,
    std::vector<unsigned int> &freeFaces);

void generateSea(VertexAttributes **attr,
                 unsigned int **indices,
                 unsigned int &attrSize,
                 unsigned int &indicesSize,
                 int width);
