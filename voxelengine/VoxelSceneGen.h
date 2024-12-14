#pragma once

#include "VoxelChunk.h"
#include "core/Scene.h"
#include "shaders/SystemParameter.h"

#include <vector>

namespace vox
{
    void initVoxelChunk(VoxelChunk &voxelChunk);
    void generateMesh(jazzfusion::VertexAttributes **attr,
                      unsigned int **indices,
                      unsigned int &attrSize,
                      unsigned int &indicesSize,
                      VoxelChunk &voxelChunk);
}