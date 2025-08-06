#pragma once

#include <cuda_runtime.h>

#include "ShaderDebugUtils.h"

//-----------------------------------------------------------------------------
// Data structure for a bin (alias table entry)
struct AliasTableBin
{
    float q;   // scaled probability (after alias processing)
    float p;   // original normalized probability
    int alias; // alias index (or -1 if none)
};

//-----------------------------------------------------------------------------
// The AliasTable class. In the constructor, we assume that 'd_weights' is a pointer
// to an array of float weights in GPU memory and that there are 'n' of them.
class AliasTable
{
public:
    // Default constructor.
    AliasTable() : bins(nullptr), len(0) {}

    // builds the alias table on the GPU.
    void update(const float *d_weights, unsigned int n, float &sumWeight);

    // Destructor.
    ~AliasTable();

    __host__ __device__ bool initialized() const { return bins != nullptr; }

    // Device function to perform sampling.
    __device__ unsigned int sample(float u, float &pmf) const
    {
        // Handle empty alias table case
        if (len == 0 || bins == nullptr) {
            pmf = 0.0f;
            return 0;
        }
        
        int offset = min(int(u * len), int(len - 1));
        float up = min(u * len - offset, 0.999999f);

        if (up < bins[offset].q)
        {
            pmf = bins[offset].p;
            return offset;
        }
        else
        {
            int alias = bins[offset].alias;
            pmf = bins[alias].p;
            return alias;
        }
    }

    // Host/device inline functions.
    __host__ __device__ unsigned int size() const { return len; }
    __device__ float PMF(unsigned int index) const { 
        if (len == 0 || bins == nullptr || index >= len) {
            return 0.0f;
        }
        return bins[index].p; 
    }

private:
    AliasTableBin *bins = nullptr;
    unsigned int len;
};

// int offset = min(int(u * len), int(len - 1));
// pmf = 1.0f / float(len);
// return offset;