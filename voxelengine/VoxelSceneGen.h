#pragma once

#include "VoxelChunk.h"
#include "core/Scene.h"
#include "shaders/SystemParameter.h"

#include <vector>

namespace vox
{
    void generateMesh(jazzfusion::VertexAttributes **attr,
                      unsigned int **indices,
                      std::vector<unsigned int> &faceLocation,
                      unsigned int &attrSize,
                      unsigned int &indicesSize,
                      unsigned int &currentFaceCount,
                      unsigned int &maxFaceCount,
                      VoxelChunk &voxelChunk);

    void updateSingleVoxel(
        unsigned int x,
        unsigned int y,
        unsigned int z,
        unsigned int newVal,
        VoxelChunk &voxelChunk,
        jazzfusion::VertexAttributes *attr,
        unsigned int *indices,
        std::vector<unsigned int> &faceLocation,
        unsigned int &attrSize,
        unsigned int &indicesSize,
        unsigned int &currentFaceCount,
        unsigned int &maxFaceCount,
        std::vector<unsigned int> &freeFaces);
}