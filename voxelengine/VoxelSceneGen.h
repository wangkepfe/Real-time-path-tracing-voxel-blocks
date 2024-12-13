#pragma once

#include "VoxelChunk.h"
#include "core/Scene.h"
#include "shaders/SystemParameter.h"

#include <vector>

namespace vox
{
    void generateMesh(std::vector<jazzfusion::VertexAttributes> &attr, std::vector<unsigned int> &idx, VoxelChunk &voxelChunk);
}