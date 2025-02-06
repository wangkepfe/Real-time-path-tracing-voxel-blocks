#include "AliasTable.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <algorithm>

namespace jazzfusion
{

    //-----------------------------------------------------------------------------
    // CUDA kernel to initialize the bins: compute p = weight/sum and scaled q = p*n.
    __global__ void initBinsKernel(const float *weights, AliasTableBin *bins, float *q, int n, float sum)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            float p = weights[i] / sum;
            bins[i].p = p;
            // Scale probability by n so that the ideal value is 1.
            q[i] = p * n;
            // (q for a deficit entry will be later written into bins[i].q.)
            bins[i].q = 0.f;
            bins[i].alias = -1;
        }
    }

    // Kernel that processes a round of pairing.
    // Each thread handles one pairing: it takes one small index and one large index,
    // writes the alias for the small index, and subtracts the deficit (1 - q_small) from the large index.
    __global__ void pairingKernel(AliasTableBin *bins, float *d_q,
                                  const int *d_small, const int *d_large, int pairingCount)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < pairingCount)
        {
            int iSmall = d_small[tid];
            int iLarge = d_large[tid];
            // For the small index, record its q and assign the donor alias
            bins[iSmall].q = d_q[iSmall];
            bins[iSmall].alias = iLarge;
            // Determine how much is needed to “fill” this small bucket to 1.
            float deficit = 1.0f - d_q[iSmall];
            // Atomically subtract the deficit from the donor (large) index.
            atomicAdd(&d_q[iLarge], -deficit);
        }
    }

    // Kernel to finalize any remaining bins: set their q to 1 if they have no alias.
    __global__ void finalizeKernel(AliasTableBin *bins, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            if (bins[i].alias == -1)
            {
                bins[i].q = 1.0f;
            }
        }
    }

    void AliasTable::update(const float *d_weights, unsigned int n, float &sumWeight)
    {
        len = n;
        // --- Step 1: Compute the sum of weights using Thrust (all on device) ---
        thrust::device_ptr<const float> dev_weights(d_weights);
        sumWeight = thrust::reduce(dev_weights, dev_weights + n, 0.0f, thrust::plus<float>());

        // --- Step 2: Allocate (or reuse) device memory for bins and a temporary array d_q ---
        if (bins == nullptr)
        {
            cudaMalloc(&bins, n * sizeof(AliasTableBin));
        }
        float *d_q;
        cudaMalloc(&d_q, n * sizeof(float));

        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        initBinsKernel<<<gridSize, blockSize>>>(d_weights, bins, d_q, n, sumWeight);
        cudaDeviceSynchronize();

        // --- Step 3: Partition indices into "small" (q < 1) and "large" (q >= 1) sets ---
        thrust::device_vector<int> indices(n);
        thrust::sequence(indices.begin(), indices.end());
        thrust::device_vector<int> d_small(n);
        thrust::device_vector<int> d_large(n);

        auto small_end = thrust::copy_if(indices.begin(), indices.end(), d_small.begin(),
                                         [d_q] __device__(int i)
                                         { return d_q[i] < 1.0f; });
        int smallCount = small_end - d_small.begin();

        auto large_end = thrust::copy_if(indices.begin(), indices.end(), d_large.begin(),
                                         [d_q] __device__(int i)
                                         { return d_q[i] >= 1.0f; });
        int largeCount = large_end - d_large.begin();

        // --- Step 4: Iteratively perform pairing rounds in parallel ---
        // In each round, process as many pairings as possible (one pairing per thread).
        while (smallCount > 0 && largeCount > 0)
        {
            int pairingCount = std::min(smallCount, largeCount);
            int pairingBlockSize = 256;
            int pairingGridSize = (pairingCount + pairingBlockSize - 1) / pairingBlockSize;
            pairingKernel<<<pairingGridSize, pairingBlockSize>>>(bins, d_q,
                                                                 thrust::raw_pointer_cast(d_small.data()),
                                                                 thrust::raw_pointer_cast(d_large.data()),
                                                                 pairingCount);
            cudaDeviceSynchronize();

            // --- Step 5: Re-partition all indices based on the updated d_q ---
            thrust::device_vector<int> new_small(n);
            thrust::device_vector<int> new_large(n);
            auto new_small_end = thrust::copy_if(indices.begin(), indices.end(), new_small.begin(),
                                                 [d_q] __device__(int i)
                                                 { return d_q[i] < 1.0f; });
            int newSmallCount = new_small_end - new_small.begin();
            auto new_large_end = thrust::copy_if(indices.begin(), indices.end(), new_large.begin(),
                                                 [d_q] __device__(int i)
                                                 { return d_q[i] >= 1.0f; });
            int newLargeCount = new_large_end - new_large.begin();
            d_small.swap(new_small);
            d_large.swap(new_large);
            smallCount = newSmallCount;
            largeCount = newLargeCount;
        }

        // --- Step 6: Finalize remaining bins ---
        gridSize = (n + blockSize - 1) / blockSize;
        finalizeKernel<<<gridSize, blockSize>>>(bins, n);
        cudaDeviceSynchronize();

        // Free the temporary d_q array.
        cudaFree(d_q);
    }

    AliasTable::~AliasTable()
    {
        if (bins != nullptr)
            cudaFree(bins);
    }
}