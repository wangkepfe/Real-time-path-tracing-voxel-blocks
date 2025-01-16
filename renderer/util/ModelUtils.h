#pragma once

#include "shaders/SystemParameter.h"
#include <string>

namespace jazzfusion
{
    void loadModel(VertexAttributes **d_attr,
                   unsigned int **d_indices,
                   unsigned int &attrSize,
                   unsigned int &indicesSize,
                   const std::string &filename);
}