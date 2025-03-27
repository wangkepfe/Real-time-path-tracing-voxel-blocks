#pragma once

#include "MaterialState.h"

struct Surface
{
    Float3 pos;
    float depth;
    bool isThinfilm;
    int materialId;
    MaterialState state;
};