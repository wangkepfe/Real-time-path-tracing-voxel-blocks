#include "VoxelSceneGen.h"
#include "util/KernelHelper.h"

#include <cub/cub.cuh> // For prefix sums

#include <fstream>

using namespace jazzfusion;
using namespace vox;

__device__ void ComputeFaceVertices(Float3 basePos, int faceId, Float3 outVerts[4])
{
    switch (faceId)
    {
    case 0: // Up
        outVerts[0] = basePos + Float3(-0.5f, 0.5f, -0.5f);
        outVerts[1] = basePos + Float3(0.5f, 0.5f, -0.5f);
        outVerts[2] = basePos + Float3(0.5f, 0.5f, 0.5f);
        outVerts[3] = basePos + Float3(-0.5f, 0.5f, 0.5f);
        break;
    case 1: // Down
        outVerts[0] = basePos + Float3(-0.5f, -0.5f, 0.5f);
        outVerts[1] = basePos + Float3(0.5f, -0.5f, 0.5f);
        outVerts[2] = basePos + Float3(0.5f, -0.5f, -0.5f);
        outVerts[3] = basePos + Float3(-0.5f, -0.5f, -0.5f);
        break;
    case 2: // Left
        outVerts[0] = basePos + Float3(-0.5f, -0.5f, 0.5f);
        outVerts[1] = basePos + Float3(-0.5f, 0.5f, 0.5f);
        outVerts[2] = basePos + Float3(-0.5f, 0.5f, -0.5f);
        outVerts[3] = basePos + Float3(-0.5f, -0.5f, -0.5f);
        break;
    case 3: // Right
        outVerts[0] = basePos + Float3(0.5f, -0.5f, -0.5f);
        outVerts[1] = basePos + Float3(0.5f, 0.5f, -0.5f);
        outVerts[2] = basePos + Float3(0.5f, 0.5f, 0.5f);
        outVerts[3] = basePos + Float3(0.5f, -0.5f, 0.5f);
        break;
    case 4: // Back
        outVerts[0] = basePos + Float3(0.5f, -0.5f, 0.5f);
        outVerts[1] = basePos + Float3(0.5f, 0.5f, 0.5f);
        outVerts[2] = basePos + Float3(-0.5f, 0.5f, 0.5f);
        outVerts[3] = basePos + Float3(-0.5f, -0.5f, 0.5f);
        break;
    case 5: // Front
        outVerts[0] = basePos + Float3(-0.5f, -0.5f, -0.5f);
        outVerts[1] = basePos + Float3(-0.5f, 0.5f, -0.5f);
        outVerts[2] = basePos + Float3(0.5f, 0.5f, -0.5f);
        outVerts[3] = basePos + Float3(0.5f, -0.5f, -0.5f);
        break;
    }
}

__global__ void GenerateVoxelChunk(Voxel *voxels, unsigned int width)
{
    Int3 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    idx.z = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx.x >= width || idx.y >= width || idx.z >= width)
        return;

    Voxel val;
    val.id = 0;
    if (idx.y < 64)
    {
        val.id = 1;
    }

    unsigned int linearId = GetLinearId(idx.x, idx.y, idx.z, width);
    voxels[linearId] = val;
}

__global__ void MarkValidFaces(
    Voxel *voxels,
    unsigned int *d_faceValid,
    unsigned int width)
{
    Int3 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    idx.z = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx.x >= width || idx.y >= width || idx.z >= width)
        return;

    unsigned int linearId = GetLinearId(idx.x, idx.y, idx.z, width);
    Voxel center = voxels[linearId];

    if (center.id == 0)
    {
        // No faces
        for (int f = 0; f < 6; f++)
            d_faceValid[linearId * 6 + f] = 0;
        return;
    }

    Int3 faceDirections[6] = {
        {0, 1, 0},  // Up
        {0, -1, 0}, // Down
        {-1, 0, 0}, // Left
        {1, 0, 0},  // Right
        {0, 0, 1},  // Back
        {0, 0, -1}  // Front
    };

    for (int f = 0; f < 6; f++)
    {
        Int3 dir = faceDirections[f];
        int nx = idx.x + dir.x;
        int ny = idx.y + dir.y;
        int nz = idx.z + dir.z;

        bool neighborEmpty = true;
        if (nx >= 0 && nx < width && ny >= 0 && ny < width && nz >= 0 && nz < width)
        {
            Voxel neighbor = voxels[GetLinearId(nx, ny, nz, width)];
            if (neighbor.id != 0)
                neighborEmpty = false;
        }

        // If neighbor is empty, this face is valid
        d_faceValid[linearId * 6 + f] = (neighborEmpty ? 1u : 0u);
    }
}

__global__ void CompactMesh(
    VertexAttributes *d_attrOut,
    unsigned int *d_idxOut,
    Voxel *voxels,
    unsigned int *d_faceValidPrefixSum,
    unsigned int width)
{
    Int3 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    idx.z = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx.x >= width || idx.y >= width || idx.z >= width)
        return;

    unsigned int linearId = GetLinearId(idx.x, idx.y, idx.z, width);
    Voxel center = voxels[linearId];
    if (center.id == 0)
        return;

    Float3 centerPos((float)idx.x, (float)idx.y, (float)idx.z);

    for (int f = 0; f < 6; f++)
    {
        unsigned int oldFaceIndex = linearId * 6 + f;

        unsigned int currPrefix = d_faceValidPrefixSum[oldFaceIndex];
        unsigned int prevPrefix = (oldFaceIndex > 0) ? d_faceValidPrefixSum[oldFaceIndex - 1] : 0;
        unsigned int faceValid = currPrefix - prevPrefix;

        if (faceValid == 1)
        {
            unsigned int newFaceIndex = currPrefix - 1;
            unsigned int vOffset = newFaceIndex * 4;
            unsigned int iOffset = newFaceIndex * 6;

            Float3 verts[4];
            ComputeFaceVertices(centerPos, f, verts);

            for (int i = 0; i < 4; i++)
            {
                d_attrOut[vOffset + i].vertex = verts[i];
            }

            // Two triangles: (0,1,2) and (0,2,3)
            d_idxOut[iOffset + 0] = vOffset + 0;
            d_idxOut[iOffset + 1] = vOffset + 1;
            d_idxOut[iOffset + 2] = vOffset + 2;
            d_idxOut[iOffset + 3] = vOffset + 0;
            d_idxOut[iOffset + 4] = vOffset + 2;
            d_idxOut[iOffset + 5] = vOffset + 3;
        }
    }
}

namespace
{
    bool dumpMeshToOBJ(
        const jazzfusion::VertexAttributes *attr,
        const unsigned int *indices,
        unsigned int attrSize,
        unsigned int indicesSize,
        const std::string &filename)
    {
        std::ofstream out(filename);
        if (!out.is_open())
        {
            return false;
        }

        out << "# Exported mesh\n";
        out << "# Vertices: " << attrSize << "\n";
        out << "# Faces: " << indicesSize / 3 << "\n\n";

        // Write vertices
        // OBJ format vertex line: v x y z
        for (size_t i = 0; i < attrSize; i++)
        {
            const Float3 &v = attr[i].vertex;
            // Y is up, which aligns with typical OBJ interpretation, no change needed
            out << "v " << v.x << " " << v.y << " " << v.z << "\n";
        }
        out << "\n";

        // Write faces
        // OBJ faces are 1-based indices. Each face line: f i j k
        // indices vector is assumed to be composed of triangles
        for (size_t i = 0; i < indicesSize; i += 3)
        {
            // Add 1 to each index because OBJ indexing starts at 1
            out << "f " << (indices[i] + 1) << " " << (indices[i + 1] + 1) << " " << (indices[i + 2] + 1) << "\n";
        }

        out.close();
        return true;
    }
}

namespace vox
{
    void initVoxelChunk(VoxelChunk &voxelChunk)
    {
        dim3 blockDim = GetBlockDim(BLOCK_DIM_4x4x4);
        dim3 gridDim = GetGridDim(voxelChunk.width, voxelChunk.width, voxelChunk.width, BLOCK_DIM_4x4x4);

        // 1. Generate voxel data
        GenerateVoxelChunk KERNEL_ARGS2(gridDim, blockDim)(voxelChunk.data, voxelChunk.width);
        cudaDeviceSynchronize();
    }

    void generateMesh(jazzfusion::VertexAttributes **attr,
                      unsigned int **indices,
                      unsigned int &attrSize,
                      unsigned int &indicesSize,
                      VoxelChunk &voxelChunk)
    {
        dim3 blockDim = GetBlockDim(BLOCK_DIM_4x4x4);
        dim3 gridDim = GetGridDim(voxelChunk.width, voxelChunk.width, voxelChunk.width, BLOCK_DIM_4x4x4);

        // 2. Mark valid faces
        unsigned int *d_faceValid;
        size_t totalVoxels = voxelChunk.width * voxelChunk.width * voxelChunk.width;
        size_t totalFaces = totalVoxels * 6;
        cudaMallocManaged(&d_faceValid, totalFaces * sizeof(unsigned int));

        MarkValidFaces KERNEL_ARGS2(gridDim, blockDim)(voxelChunk.data, d_faceValid, voxelChunk.width);
        cudaDeviceSynchronize();

        // 3. Prefix sum on face validity
        unsigned int *d_faceValidPrefixSum;
        cudaMallocManaged(&d_faceValidPrefixSum, totalFaces * sizeof(unsigned int));

        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        // Determine temp storage size
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_faceValid, d_faceValidPrefixSum, totalFaces);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_faceValid, d_faceValidPrefixSum, totalFaces);
        cudaFree(d_temp_storage);

        cudaDeviceSynchronize();

        // 4. Find how many faces are valid total
        unsigned int totalValidFaces = 0;
        cudaMemcpy(&totalValidFaces, &d_faceValidPrefixSum[totalFaces - 1], sizeof(unsigned int), cudaMemcpyDeviceToHost);

        // Now we know how large our final arrays should be:
        // Each valid face: 4 vertices, 6 indices
        attrSize = totalValidFaces * 4;
        indicesSize = totalValidFaces * 6;

        if (*attr != nullptr)
        {
            cudaFree(*attr);
        }
        if (*indices != nullptr)
        {
            cudaFree(*indices);
        }

        cudaMallocManaged(attr, attrSize * sizeof(jazzfusion::VertexAttributes));
        cudaMallocManaged(indices, indicesSize * sizeof(unsigned int));

        // 5. Compact the mesh
        CompactMesh KERNEL_ARGS2(gridDim, blockDim)(*attr, *indices, voxelChunk.data, d_faceValidPrefixSum, voxelChunk.width);
        CUDA_CHECK(cudaPeekAtLastError());
        cudaDeviceSynchronize();

        dumpMeshToOBJ(*attr, *indices, attrSize, indicesSize, "debug.obj");

        // Cleanup
        cudaFree(d_faceValid);
        cudaFree(d_faceValidPrefixSum);
    }
}
