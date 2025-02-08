// large_scale_ascii_test.cu
#include "../../shaders/AliasTable.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>

// Kernel that calls the device-side sample() function for each sample.
__global__ void sampleKernel(const AliasTable table, const float *d_randoms,
                             unsigned int *d_indices, float *d_pmf, int numSamples)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numSamples)
    {
        float pmf;
        unsigned int index = table.sample(d_randoms[tid], pmf);
        d_indices[tid] = index;
        d_pmf[tid] = pmf;
    }
}

// Helper: Print an aggregated ASCII histogram from a frequency vector.
void printHistogram(const std::vector<unsigned int> &counts, int numBuckets)
{
    int n = counts.size();
    int groupSize = n / numBuckets;
    if (groupSize == 0)
        groupSize = 1;
    std::vector<unsigned int> bucketSums(numBuckets, 0);
    for (int i = 0; i < n; i++)
    {
        int bucket = i / groupSize;
        if (bucket >= numBuckets)
            bucket = numBuckets - 1;
        bucketSums[bucket] += counts[i];
    }
    // Find maximum bucket sum for scaling bar lengths.
    unsigned int maxSum = *std::max_element(bucketSums.begin(), bucketSums.end());
    std::cout << "Histogram (each bar scaled to max 50 chars):\n";
    for (int b = 0; b < numBuckets; b++)
    {
        std::cout << "[" << b << "]: ";
        int barLen = (maxSum > 0) ? (bucketSums[b] * 50) / maxSum : 0;
        for (int j = 0; j < barLen; j++)
        {
            std::cout << "#";
        }
        std::cout << " (" << bucketSums[b] << ")\n";
    }
    std::cout << "\n";
}

// Runs one test: given a host vector of weights and a test name, it builds the alias table,
// draws numSamples samples, and prints an ASCII histogram of the frequency counts.
void runTest(const std::vector<float> &h_weights, const std::string &testName)
{
    unsigned int n = h_weights.size();
    const unsigned int numSamples = 10000; // number of samples to draw

    std::cout << "Running test: " << testName << "\n";

    // Allocate and copy weights to device.
    float *d_weights;
    cudaMalloc(&d_weights, n * sizeof(float));
    cudaMemcpy(d_weights, h_weights.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // Build the alias table.
    AliasTable table;
    float sumWeights = 0.0f;
    table.update(d_weights, n, sumWeights);

    // (Optional) You could verify that sumWeights equals the expected sum,
    // but here we simply print it.
    std::cout << "AliasTable update succeeded. Sum weights: " << sumWeights << "\n";

    // Allocate device memory for sampling.
    float *d_randoms;
    unsigned int *d_indices;
    float *d_pmf;
    cudaMalloc(&d_randoms, numSamples * sizeof(float));
    cudaMalloc(&d_indices, numSamples * sizeof(unsigned int));
    cudaMalloc(&d_pmf, numSamples * sizeof(float));

    // Generate random numbers uniformly in [0, 1) on host.
    std::vector<float> h_randoms(numSamples);
    for (unsigned int i = 0; i < numSamples; ++i)
        h_randoms[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) + 1.0f);
    cudaMemcpy(d_randoms, h_randoms.data(), numSamples * sizeof(float), cudaMemcpyHostToDevice);

    // Launch sampling kernel.
    int blockSize = 256;
    int gridSize = (numSamples + blockSize - 1) / blockSize;
    sampleKernel<<<gridSize, blockSize>>>(table, d_randoms, d_indices, d_pmf, numSamples);
    cudaDeviceSynchronize();

    // Copy sample results back to host.
    std::vector<unsigned int> h_indices(numSamples);
    cudaMemcpy(h_indices.data(), d_indices, numSamples * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Build histogram: count frequency for each bin (there are n bins).
    std::vector<unsigned int> frequency(n, 0);
    for (unsigned int i = 0; i < numSamples; i++)
    {
        if (h_indices[i] < n)
            frequency[h_indices[i]]++;
    }

    // For display purposes, aggregate the 10000 bins into 80 buckets.
    int numBuckets = 30;
    std::cout << "Histogram for " << testName << ":\n";
    printHistogram(frequency, numBuckets);

    // Cleanup device memory.
    cudaFree(d_weights);
    cudaFree(d_randoms);
    cudaFree(d_indices);
    cudaFree(d_pmf);
}

int main()
{
    srand(static_cast<unsigned int>(time(0)));

    // Test 1: 10,000 uniform weights (each weight = 1.0)
    int numBins = 10000;
    std::vector<float> uniformWeights(numBins, 1.0f);
    // runTest(uniformWeights, "Uniform Weights");

    // Test 2: 10,000 linear weights (for bin i, weight = i+1)
    std::vector<float> linearWeights(numBins);
    for (int i = 0; i < numBins; i++)
    {
        linearWeights[i] = static_cast<float>(i + 1);
    }
    runTest(linearWeights, "Linear Weights");

    return 0;
}
