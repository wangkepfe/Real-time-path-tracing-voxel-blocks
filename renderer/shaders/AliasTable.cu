#include "AliasTable.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>

#include <algorithm>
#include <vector>
#include <queue>
#include <numeric>

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
// Each thread handles one pairing: it takes one smallList index and one largeList index,
// writes the alias for the smallList index, and subtracts the deficit (1 - q_small) from the largeList index.
__global__ void pairingKernel(AliasTableBin *bins, float *d_q,
                              const int *d_small, const int *d_large, int pairingCount)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < pairingCount)
    {
        int iSmall = d_small[tid];
        int iLarge = d_large[tid];
        // For the smallList index, record its q and assign the donor alias
        bins[iSmall].q = d_q[iSmall];
        bins[iSmall].alias = iLarge;
        // Determine how much is needed to “fill” this smallList bucket to 1.
        float deficit = 1.0f - d_q[iSmall];
        // Atomically subtract the deficit from the donor (largeList) index.
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

void AliasTable::update(const float *weights, unsigned int n, float &sumWeight)
{
    len = n;
    constexpr bool g_use_cpu_build = true;
    if (g_use_cpu_build)
    {
        // Compute the sum of weights on the device using Thrust.

        thrust::device_ptr<const float> dev_weights(weights);
        sumWeight = thrust::reduce(dev_weights, dev_weights + n, 0.0f, thrust::plus<float>());

        // Copy the weights from device to host.
        std::vector<float> h_weights(n);
        cudaMemcpy(h_weights.data(), weights, n * sizeof(float), cudaMemcpyDeviceToHost);

        // Compute normalized probabilities and scaled probabilities.
        std::vector<float> prob(n);
        std::vector<float> scaledProb(n);
        std::vector<int> alias(n, -1);
        for (unsigned int i = 0; i < n; ++i)
        {
            float p = h_weights[i] / sumWeight;
            prob[i] = p;
            scaledProb[i] = p * n;
        }

        // Partition indices into "smallList" and "large" buckets.
        std::queue<int> smallList;
        std::queue<int> large;
        for (unsigned int i = 0; i < n; ++i)
        {
            if (scaledProb[i] < 1.0f)
                smallList.push(i);
            else
                large.push(i);
        }

        // Process the two queues.
        while (!smallList.empty() && !large.empty())
        {
            int smallIdx = smallList.front();
            smallList.pop();
            int largeIdx = large.front();
            large.pop();

            alias[smallIdx] = largeIdx;
            // Decrease the large bucket's probability.
            scaledProb[largeIdx] -= (1.0f - scaledProb[smallIdx]);

            if (scaledProb[largeIdx] < 1.0f)
                smallList.push(largeIdx);
            else
                large.push(largeIdx);
        }

        // Finalize remaining entries.
        while (!smallList.empty())
        {
            int idx = smallList.front();
            smallList.pop();
            scaledProb[idx] = 1.0f;
        }
        while (!large.empty())
        {
            int idx = large.front();
            large.pop();
            scaledProb[idx] = 1.0f;
        }

        // Build a temporary host alias table.
        std::vector<AliasTableBin> h_bins(n);
        for (unsigned int i = 0; i < n; ++i)
        {
            h_bins[i].p = prob[i];
            h_bins[i].q = scaledProb[i];
            h_bins[i].alias = alias[i];
        }

        // Allocate (or reallocate) device memory for bins.
        if (bins != nullptr)
        {
            cudaFree(bins);
            bins = nullptr;
        }
        cudaMalloc(&bins, n * sizeof(AliasTableBin));
        // Upload the host alias table into device memory.
        cudaMemcpy(bins, h_bins.data(), n * sizeof(AliasTableBin), cudaMemcpyHostToDevice);
    }
    else
    {
        // ----------------------------
        // GPU BUILD: build the alias table entirely on the device.
        // Assumes that 'weights' is a device pointer.
        // ----------------------------
        // Compute sum of weights using Thrust.

        thrust::device_ptr<const float> dev_weights(weights);
        sumWeight = thrust::reduce(dev_weights, dev_weights + n, 0.0f, thrust::plus<float>());

        // Allocate device memory for bins.
        if (bins != nullptr)
        {
            cudaFree(bins);
            bins = nullptr;
        }
        cudaMalloc(&bins, n * sizeof(AliasTableBin));

        float *d_q;
        cudaMalloc(&d_q, n * sizeof(float));

        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        initBinsKernel<<<gridSize, blockSize>>>(weights, bins, d_q, n, sumWeight);
        cudaDeviceSynchronize();

        // Partition indices into "smallList" (q < 1) and "largeList" (q >= 1) sets.
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

        // Iteratively process pairing rounds.
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

            // Re-partition indices based on updated probabilities.
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

        gridSize = (n + blockSize - 1) / blockSize;
        finalizeKernel<<<gridSize, blockSize>>>(bins, n);
        cudaDeviceSynchronize();

        cudaFree(d_q);
    }
}

AliasTable::~AliasTable()
{
    if (bins != nullptr)
        cudaFree(bins);
}