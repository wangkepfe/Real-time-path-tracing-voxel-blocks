#include "VertexSkinning.h"
#include "util/DebugUtils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Global skinning data on GPU
static float* d_globalJointMatrices = nullptr;
static bool skinningInitialized = false;

// Device function to transform a point by a 4x4 matrix
__device__ Float3 transformPoint(const Float3& point, const float* matrix)
{
    Float3 result;
    result.x = matrix[0] * point.x + matrix[1] * point.y + matrix[2] * point.z + matrix[3];
    result.y = matrix[4] * point.x + matrix[5] * point.y + matrix[6] * point.z + matrix[7];
    result.z = matrix[8] * point.x + matrix[9] * point.y + matrix[10] * point.z + matrix[11];
    return result;
}

// Device function for 4x4 matrix multiplication
__device__ void multiplyFloat4x4(const float* a, const float* b, float* result)
{
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            result[i * 4 + j] = 0.0f;
            for (int k = 0; k < 4; ++k)
            {
                result[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
            }
        }
    }
}

// Device function to blend vertex transforms based on joint weights
__device__ Float3 blendTransforms(const Float3& vertex, const Int4& joints, const Float4& weights,
                                  const float* jointMatrices, int numJoints)
{
    Float3 result = Float3(0.0f);

    // Apply skinning for each joint with non-zero weight
    if (weights.x > 0.0f && joints.x < numJoints)
    {
        Float3 transformed = transformPoint(vertex, &jointMatrices[joints.x * 16]);
        result = result + transformed * weights.x;
    }

    if (weights.y > 0.0f && joints.y < numJoints)
    {
        Float3 transformed = transformPoint(vertex, &jointMatrices[joints.y * 16]);
        result = result + transformed * weights.y;
    }

    if (weights.z > 0.0f && joints.z < numJoints)
    {
        Float3 transformed = transformPoint(vertex, &jointMatrices[joints.z * 16]);
        result = result + transformed * weights.z;
    }

    if (weights.w > 0.0f && joints.w < numJoints)
    {
        Float3 transformed = transformPoint(vertex, &jointMatrices[joints.w * 16]);
        result = result + transformed * weights.w;
    }

    return result;
}

// CUDA kernel for applying vertex skinning
__global__ void applyVertexSkinning(
    VertexAttributes* vertices,
    const VertexAttributes* originalVertices,
    const float* jointMatrices,
    int numVertices,
    int numJoints)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numVertices) return;

    const VertexAttributes& original = originalVertices[idx];
    VertexAttributes& skinned = vertices[idx];

    // Copy texture coordinates (unchanged)
    skinned.texcoord = original.texcoord;
    skinned.jointIndices = original.jointIndices;
    skinned.jointWeights = original.jointWeights;

    // Apply vertex skinning to position
    skinned.vertex = blendTransforms(
        original.vertex,
        original.jointIndices,
        original.jointWeights,
        jointMatrices,
        numJoints
    );
}

// Optimized CUDA kernel using shared memory for joint matrices
__global__ void applyVertexSkinningOptimized(
    VertexAttributes* vertices,
    const VertexAttributes* originalVertices,
    const float* jointMatrices,
    int numVertices,
    int numJoints)
{
    extern __shared__ float sharedJointMatrices[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Cooperatively load joint matrices into shared memory
    int matricesToLoad = min(numJoints, MAX_SKINNING_JOINTS);
    int matrixElementsPerThread = (matricesToLoad * 16 + blockSize - 1) / blockSize;

    for (int i = 0; i < matrixElementsPerThread; ++i)
    {
        int elementIdx = tid + i * blockSize;
        if (elementIdx < matricesToLoad * 16)
        {
            sharedJointMatrices[elementIdx] = jointMatrices[elementIdx];
        }
    }

    __syncthreads();

    if (idx >= numVertices) return;

    const VertexAttributes& original = originalVertices[idx];
    VertexAttributes& skinned = vertices[idx];

    // Copy unchanged attributes
    skinned.texcoord = original.texcoord;
    skinned.jointIndices = original.jointIndices;
    skinned.jointWeights = original.jointWeights;

    // Apply vertex skinning using shared memory
    skinned.vertex = blendTransforms(
        original.vertex,
        original.jointIndices,
        original.jointWeights,
        sharedJointMatrices,
        matricesToLoad
    );
}

// Host functions implementation

void initVertexSkinning()
{
    if (skinningInitialized) return;

    // Allocate global joint matrices buffer
    CUDA_CHECK(cudaMalloc(&d_globalJointMatrices, MAX_SKINNING_JOINTS * 16 * sizeof(float)));

    // Initialize with identity matrices
    float identityMatrices[MAX_SKINNING_JOINTS * 16];
    for (int i = 0; i < MAX_SKINNING_JOINTS; ++i)
    {
        // Set identity matrix
        for (int j = 0; j < 16; ++j)
        {
            identityMatrices[i * 16 + j] = (j % 5 == 0) ? 1.0f : 0.0f; // Identity: 1 on diagonal, 0 elsewhere
        }
    }

    CUDA_CHECK(cudaMemcpy(d_globalJointMatrices, identityMatrices,
                          MAX_SKINNING_JOINTS * 16 * sizeof(float), cudaMemcpyHostToDevice));

    skinningInitialized = true;

    std::cout << "Vertex skinning system initialized" << std::endl;
}

void cleanupVertexSkinning()
{
    if (!skinningInitialized) return;

    if (d_globalJointMatrices)
    {
        CUDA_CHECK(cudaFree(d_globalJointMatrices));
        d_globalJointMatrices = nullptr;
    }

    skinningInitialized = false;

    std::cout << "Vertex skinning system cleaned up" << std::endl;
}

void updateSkinningMatrices(const float* hostMatrices, int numJoints)
{
    if (!skinningInitialized)
    {
        initVertexSkinning();
    }

    if (numJoints > MAX_SKINNING_JOINTS)
    {
        std::cerr << "Warning: Too many joints (" << numJoints << "), truncating to "
                  << MAX_SKINNING_JOINTS << std::endl;
        numJoints = MAX_SKINNING_JOINTS;
    }

    CUDA_CHECK(cudaMemcpy(d_globalJointMatrices, hostMatrices,
                          numJoints * 16 * sizeof(float), cudaMemcpyHostToDevice));
}

void applySkinningToVertices(VertexAttributes* d_vertices, const VertexAttributes* d_originalVertices,
                           int numVertices, const SkinningData& skinningData)
{
    if (!skinningData.enabled || numVertices == 0) {
        return;
    }

    if (!d_vertices || !d_originalVertices || !skinningData.jointMatrices) {
        return;
    }

    // Launch configuration
    const int blockSize = 256;
    const int numBlocks = (numVertices + blockSize - 1) / blockSize;

    // Calculate shared memory size for optimized kernel
    int sharedMemSize = min(skinningData.numJoints, MAX_SKINNING_JOINTS) * 16 * sizeof(float);

        // Choose kernel based on number of joints and available shared memory
    if (skinningData.numJoints <= 32 && sharedMemSize <= 48 * 1024) // Max shared memory per block
    {
        // Use optimized kernel with shared memory
        applyVertexSkinningOptimized<<<numBlocks, blockSize, sharedMemSize>>>(
            d_vertices,
            d_originalVertices,
            skinningData.jointMatrices ? skinningData.jointMatrices : d_globalJointMatrices,
            numVertices,
            skinningData.numJoints
        );
    }
    else
    {
        // Use standard kernel
        applyVertexSkinning<<<numBlocks, blockSize>>>(
            d_vertices,
            d_originalVertices,
            skinningData.jointMatrices ? skinningData.jointMatrices : d_globalJointMatrices,
            numVertices,
            skinningData.numJoints
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Kernel for batch vertex skinning (for multiple entities)
__global__ void applyBatchVertexSkinning(
    VertexAttributes** verticesArray,
    const VertexAttributes** originalVerticesArray,
    const float** jointMatricesArray,
    const int* numVerticesArray,
    const int* numJointsArray,
    int numBatches)
{
    int batchIdx = blockIdx.z;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batchIdx >= numBatches) return;
    if (idx >= numVerticesArray[batchIdx]) return;

    VertexAttributes* vertices = verticesArray[batchIdx];
    const VertexAttributes* originalVertices = originalVerticesArray[batchIdx];
    const float* jointMatrices = jointMatricesArray[batchIdx];
    int numJoints = numJointsArray[batchIdx];

    const VertexAttributes& original = originalVertices[idx];
    VertexAttributes& skinned = vertices[idx];

    // Copy unchanged attributes
    skinned.texcoord = original.texcoord;
    skinned.jointIndices = original.jointIndices;
    skinned.jointWeights = original.jointWeights;

    // Apply vertex skinning
    skinned.vertex = blendTransforms(
        original.vertex,
        original.jointIndices,
        original.jointWeights,
        jointMatrices,
        numJoints
    );
}

// Host function for batch processing multiple animated entities
void applyBatchSkinning(VertexAttributes** d_verticesArray,
                       const VertexAttributes** d_originalVerticesArray,
                       const float** d_jointMatricesArray,
                       const int* d_numVerticesArray,
                       const int* d_numJointsArray,
                       int numBatches,
                       int maxVertices)
{
    if (numBatches == 0) return;

    const int blockSize = 256;
    dim3 gridSize((maxVertices + blockSize - 1) / blockSize, 1, numBatches);
    dim3 blockDim(blockSize, 1, 1);

    applyBatchVertexSkinning<<<gridSize, blockDim>>>(
        d_verticesArray,
        d_originalVerticesArray,
        d_jointMatricesArray,
        d_numVerticesArray,
        d_numJointsArray,
        numBatches
    );

    CUDA_CHECK(cudaGetLastError());
}