#pragma once

#include <optix.h>
#include <cuda_runtime.h>

struct DIReservoir
{
    unsigned int lightData;
    unsigned int uvData;
    float weightSum; // Used during streaming (RIS weight sum) and later holds reservoir weight.
    float targetPdf; // Target PDF of the selected sample.
    float M;         // Number of samples considered (to be quantized into 32 bits).
};