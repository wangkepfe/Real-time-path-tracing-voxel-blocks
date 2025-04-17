#pragma once

#include <optix.h>
#include <cuda_runtime.h>

struct DIReservoir
{
    unsigned int lightData;
    unsigned int uvData;
    float weightSum;
    float targetPdf;
    float M;
};