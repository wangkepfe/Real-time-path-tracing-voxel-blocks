#include "VertexSkinning.h"
#include "util/DebugUtils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Global skinning data on GPU
static Mat4 *d_globalJointMatrices = nullptr;
static bool skinningInitialized = false;

// Device function to blend vertex transforms based on joint weights
__device__ Float3 blendTransforms(const Float3 &vertex, const VertexSkinningData &skinningData,
                                  const Mat4 *jointMatrices, int numJoints)
{
    Float3 result = Float3(0.0f);

    // Apply skinning for each joint with non-zero weight
    if (skinningData.jointWeights.x > 0.0f && skinningData.jointIndices.x < numJoints)
    {
        Float3 transformed = jointMatrices[skinningData.jointIndices.x] * vertex;
        result = result + transformed * skinningData.jointWeights.x;
    }

    if (skinningData.jointWeights.y > 0.0f && skinningData.jointIndices.y < numJoints)
    {
        Float3 transformed = jointMatrices[skinningData.jointIndices.y] * vertex;
        result = result + transformed * skinningData.jointWeights.y;
    }

    if (skinningData.jointWeights.z > 0.0f && skinningData.jointIndices.z < numJoints)
    {
        Float3 transformed = jointMatrices[skinningData.jointIndices.z] * vertex;
        result = result + transformed * skinningData.jointWeights.z;
    }

    if (skinningData.jointWeights.w > 0.0f && skinningData.jointIndices.w < numJoints)
    {
        Float3 transformed = jointMatrices[skinningData.jointIndices.w] * vertex;
        result = result + transformed * skinningData.jointWeights.w;
    }

    return result;
}

// CUDA kernel for applying vertex skinning
__global__ void applyVertexSkinning(
    VertexAttributes *vertices,
    const VertexAttributes *originalVertices,
    const VertexSkinningData *skinningData,
    const Mat4 *jointMatrices,
    int numVertices,
    int numJoints)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numVertices)
        return;

    const VertexAttributes &original = originalVertices[idx];
    const VertexSkinningData &skinning = skinningData[idx];
    VertexAttributes &skinned = vertices[idx];

    // Copy texture coordinates (unchanged)
    skinned.texcoord = original.texcoord;

    // Apply vertex skinning to position
    skinned.vertex = blendTransforms(
        original.vertex,
        skinning,
        jointMatrices,
        numJoints);
}

// Optimized CUDA kernel using shared memory for joint matrices
__global__ void applyVertexSkinningOptimized(
    VertexAttributes *vertices,
    const VertexAttributes *originalVertices,
    const VertexSkinningData *skinningData,
    const Mat4 *jointMatrices,
    int numVertices,
    int numJoints)
{
    extern __shared__ Mat4 sharedJointMatrices[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Cooperatively load joint matrices into shared memory
    int matricesToLoad = min(numJoints, MAX_SKINNING_JOINTS);
    int matricesPerThread = (matricesToLoad + blockSize - 1) / blockSize;

    for (int i = 0; i < matricesPerThread; ++i)
    {
        int matrixIdx = tid + i * blockSize;
        if (matrixIdx < matricesToLoad)
        {
            sharedJointMatrices[matrixIdx] = jointMatrices[matrixIdx];
        }
    }

    __syncthreads();

    if (idx >= numVertices)
        return;

    const VertexAttributes &original = originalVertices[idx];
    const VertexSkinningData &skinning = skinningData[idx];
    VertexAttributes &skinned = vertices[idx];

    // Copy unchanged attributes
    skinned.texcoord = original.texcoord;

    // Apply vertex skinning using shared memory
    skinned.vertex = blendTransforms(
        original.vertex,
        skinning,
        sharedJointMatrices,
        matricesToLoad);
}

// Host functions implementation

void initVertexSkinning()
{
    if (skinningInitialized)
        return;

    // Allocate global joint matrices buffer
    CUDA_CHECK(cudaMalloc(&d_globalJointMatrices, MAX_SKINNING_JOINTS * sizeof(Mat4)));

    // Initialize with identity matrices
    Mat4 identityMatrices[MAX_SKINNING_JOINTS];
    for (int i = 0; i < MAX_SKINNING_JOINTS; ++i)
    {
        identityMatrices[i] = Mat4(); // Default constructor creates identity
    }

    CUDA_CHECK(cudaMemcpy(d_globalJointMatrices, identityMatrices,
                          MAX_SKINNING_JOINTS * sizeof(Mat4), cudaMemcpyHostToDevice));

    skinningInitialized = true;

    std::cout << "Vertex skinning system initialized" << std::endl;
}

void cleanupVertexSkinning()
{
    if (!skinningInitialized)
        return;

    if (d_globalJointMatrices)
    {
        CUDA_CHECK(cudaFree(d_globalJointMatrices));
        d_globalJointMatrices = nullptr;
    }

    skinningInitialized = false;

    std::cout << "Vertex skinning system cleaned up" << std::endl;
}

void updateSkinningMatrices(const float *hostMatrices, int numJoints)
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

    CUDA_CHECK(cudaMemcpy(d_globalJointMatrices, (Mat4 *)hostMatrices,
                          numJoints * sizeof(Mat4), cudaMemcpyHostToDevice));
}

void applySkinningToVertices(VertexAttributes *d_vertices, const VertexAttributes *d_originalVertices,
                             const VertexSkinningData *d_skinningData, int numVertices, const SkinningData &skinningData)
{
    if (!skinningData.enabled || numVertices == 0)
    {
        return;
    }

    if (!d_vertices || !d_originalVertices || !d_skinningData || !skinningData.jointMatrices)
    {
        return;
    }

    // Launch configuration
    const int blockSize = 256;
    const int numBlocks = (numVertices + blockSize - 1) / blockSize;

    // Calculate shared memory size for optimized kernel
    int sharedMemSize = min(skinningData.numJoints, MAX_SKINNING_JOINTS) * sizeof(Mat4);

    // Choose kernel based on number of joints and available shared memory
    if (skinningData.numJoints <= 32 && sharedMemSize <= 48 * 1024) // Max shared memory per block
    {
        // Use optimized kernel with shared memory
        applyVertexSkinningOptimized<<<numBlocks, blockSize, sharedMemSize>>>(
            d_vertices,
            d_originalVertices,
            d_skinningData,
            skinningData.jointMatrices ? skinningData.jointMatrices : d_globalJointMatrices,
            numVertices,
            skinningData.numJoints);
    }
    else
    {
        // Use standard kernel
        applyVertexSkinning<<<numBlocks, blockSize>>>(
            d_vertices,
            d_originalVertices,
            d_skinningData,
            skinningData.jointMatrices ? skinningData.jointMatrices : d_globalJointMatrices,
            numVertices,
            skinningData.numJoints);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}