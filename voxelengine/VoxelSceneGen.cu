#include "VoxelSceneGen.h"
#include "util/KernelHelper.h"

#include <cub/cub.cuh> // For prefix sums

#include <fstream>

using namespace jazzfusion;
using namespace vox;

__device__ __host__ void ComputeFaceVertices(Float3 basePos, int faceId, Float3 outVerts[4])
{
    switch (faceId)
    {
    case 0: // Up (top face at y+1)
        outVerts[0] = basePos + Float3(0.0f, 1.0f, 0.0f);
        outVerts[1] = basePos + Float3(1.0f, 1.0f, 0.0f);
        outVerts[2] = basePos + Float3(1.0f, 1.0f, 1.0f);
        outVerts[3] = basePos + Float3(0.0f, 1.0f, 1.0f);
        break;

    case 1: // Down (bottom face at y)
        outVerts[0] = basePos + Float3(0.0f, 0.0f, 1.0f);
        outVerts[1] = basePos + Float3(1.0f, 0.0f, 1.0f);
        outVerts[2] = basePos + Float3(1.0f, 0.0f, 0.0f);
        outVerts[3] = basePos + Float3(0.0f, 0.0f, 0.0f);
        break;

    case 2: // Left (face at x)
        outVerts[0] = basePos + Float3(0.0f, 0.0f, 1.0f);
        outVerts[1] = basePos + Float3(0.0f, 1.0f, 1.0f);
        outVerts[2] = basePos + Float3(0.0f, 1.0f, 0.0f);
        outVerts[3] = basePos + Float3(0.0f, 0.0f, 0.0f);
        break;

    case 3: // Right (face at x+1)
        outVerts[0] = basePos + Float3(1.0f, 0.0f, 0.0f);
        outVerts[1] = basePos + Float3(1.0f, 1.0f, 0.0f);
        outVerts[2] = basePos + Float3(1.0f, 1.0f, 1.0f);
        outVerts[3] = basePos + Float3(1.0f, 0.0f, 1.0f);
        break;

    case 4: // Back (face at z+1)
        outVerts[0] = basePos + Float3(1.0f, 0.0f, 1.0f);
        outVerts[1] = basePos + Float3(1.0f, 1.0f, 1.0f);
        outVerts[2] = basePos + Float3(0.0f, 1.0f, 1.0f);
        outVerts[3] = basePos + Float3(0.0f, 0.0f, 1.0f);
        break;

    case 5: // Front (face at z)
        outVerts[0] = basePos + Float3(0.0f, 0.0f, 0.0f);
        outVerts[1] = basePos + Float3(0.0f, 1.0f, 0.0f);
        outVerts[2] = basePos + Float3(1.0f, 1.0f, 0.0f);
        outVerts[3] = basePos + Float3(1.0f, 0.0f, 0.0f);
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
    if (idx.y < width / 2)
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
    unsigned int *d_faceLocation,
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
    {
        for (int f = 0; f < 6; f++)
        {
            unsigned int oldFaceIndex = linearId * 6 + f;
            d_faceLocation[oldFaceIndex] = UINT_MAX;
        }
        return;
    }

    Float3 centerPos((float)idx.x, (float)idx.y, (float)idx.z);

    for (int f = 0; f < 6; f++)
    {
        unsigned int oldFaceIndex = linearId * 6 + f;

        unsigned int currPrefix = d_faceValidPrefixSum[oldFaceIndex];
        unsigned int prevPrefix = (oldFaceIndex > 0) ? d_faceValidPrefixSum[oldFaceIndex - 1] : 0;
        unsigned int faceValid = currPrefix - prevPrefix;

        d_faceLocation[oldFaceIndex] = (faceValid == 1) ? (currPrefix - 1) : UINT_MAX;

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

    bool neighborEmpty(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int width, const Voxel *voxels)
    {
        if (nx >= width || ny >= width || nz >= width)
            return true; // Out of bounds => treat as empty
        Voxel n = voxels[GetLinearId(nx, ny, nz, width)];
        return (n.id == 0);
    }
}

namespace vox
{
    void generateMesh(jazzfusion::VertexAttributes **attr,
                      unsigned int **indices,
                      std::vector<unsigned int> &faceLocation,
                      unsigned int &attrSize,
                      unsigned int &indicesSize,
                      unsigned int &currentFaceCount,
                      unsigned int &maxFaceCount,
                      VoxelChunk &voxelChunk)
    {
        dim3 blockDim = GetBlockDim(BLOCK_DIM_4x4x4);
        dim3 gridDim = GetGridDim(voxelChunk.width, voxelChunk.width, voxelChunk.width, BLOCK_DIM_4x4x4);

        unsigned int *d_faceValid;
        unsigned int *d_faceLocation;
        size_t totalVoxels = voxelChunk.width * voxelChunk.width * voxelChunk.width;
        size_t totalFaces = totalVoxels * 6;
        cudaMalloc(&d_faceValid, totalFaces * sizeof(unsigned int));
        cudaMalloc(&d_faceLocation, totalFaces * sizeof(unsigned int));

        Voxel *d_data;
        cudaMalloc(&d_data, totalVoxels * sizeof(Voxel));

        // 1. generate
        GenerateVoxelChunk KERNEL_ARGS2(gridDim, blockDim)(d_data, voxelChunk.width);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaPeekAtLastError());

        // 2. Mark valid faces
        MarkValidFaces KERNEL_ARGS2(gridDim, blockDim)(d_data, d_faceValid, voxelChunk.width);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaPeekAtLastError());

        // 3. Prefix sum on face validity
        unsigned int *d_faceValidPrefixSum;
        cudaMalloc(&d_faceValidPrefixSum, totalFaces * sizeof(unsigned int));

        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        // Determine temp storage size
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_faceValid, d_faceValidPrefixSum, totalFaces);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_faceValid, d_faceValidPrefixSum, totalFaces);
        cudaFree(d_temp_storage);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaPeekAtLastError());

        // 4. Find how many faces are valid total
        cudaMemcpy(&currentFaceCount, &d_faceValidPrefixSum[totalFaces - 1], sizeof(unsigned int), cudaMemcpyDeviceToHost);
        maxFaceCount = currentFaceCount * 2;

        attrSize = currentFaceCount * 4;
        indicesSize = currentFaceCount * 6;

        if (*attr != nullptr)
        {
            cudaFree(*attr);
        }
        if (*indices != nullptr)
        {
            cudaFree(*indices);
        }

        cudaMallocManaged(attr, maxFaceCount * 4 * sizeof(jazzfusion::VertexAttributes));
        cudaMallocManaged(indices, maxFaceCount * 6 * sizeof(unsigned int));

        // 5. Compact the mesh
        CompactMesh KERNEL_ARGS2(gridDim, blockDim)(*attr, *indices, d_faceLocation, d_data, d_faceValidPrefixSum, voxelChunk.width);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaPeekAtLastError());

        faceLocation.resize(totalFaces);
        cudaMemcpy(faceLocation.data(), d_faceLocation, totalFaces * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        if (0)
        {
            dumpMeshToOBJ(*attr, *indices, attrSize, indicesSize, "debug0.obj");
        }

        cudaMemcpy(voxelChunk.data, d_data, totalVoxels * sizeof(Voxel), cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_faceValid);
        cudaFree(d_faceLocation);
        cudaFree(d_faceValidPrefixSum);

        cudaFree(d_data);
    }

    // CPU function to update the mesh for a single changed voxel (add or remove).
    // newVal = 1 if the voxel is now filled, 0 if empty.
    // This function:
    // 1. Updates the voxel data.
    // 2. Recomputes which faces should be present for this voxel.
    // 3. Adds or removes faces in the mesh buffers accordingly.
    void updateSingleVoxel(
        unsigned int x,
        unsigned int y,
        unsigned int z,
        unsigned int newVal,
        VoxelChunk &voxelChunk,
        jazzfusion::VertexAttributes *attr,
        unsigned int *indices,
        std::vector<unsigned int> &faceLocation,
        unsigned int &attrSize,
        unsigned int &indicesSize,
        unsigned int &currentFaceCount,
        unsigned int &maxFaceCount,
        std::vector<unsigned int> &freeFaces)
    {
        // Update the voxel
        unsigned int vid = GetLinearId(x, y, z, voxelChunk.width);
        Voxel oldVoxel = voxelChunk.data[vid];
        voxelChunk.data[vid].id = newVal;

        // If voxel state didn't change, no update needed
        if (oldVoxel.id == newVal)
            return;

        // For each of the 6 faces of this voxel, determine if the face should exist
        // Face is valid if voxel is filled and neighbor in that direction is empty.
        bool facesShouldExist[6];
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
            unsigned int nx = x + dir.x;
            unsigned int ny = y + dir.y;
            unsigned int nz = z + dir.z;
            bool neighEmpty = neighborEmpty(nx, ny, nz, voxelChunk.width, voxelChunk.data);
            facesShouldExist[f] = ((newVal != 0) && neighEmpty) || ((newVal == 0) && !neighEmpty);
        }

        Float3 centerPos((float)x, (float)y, (float)z);

        for (int f = 0; f < 6; f++)
        {
            int faceId = (int)(vid * 6 + f);
            unsigned int existingFaceIndex = faceLocation[faceId];
            bool currentlyExists = (existingFaceIndex != UINT_MAX);
            bool shouldExist = facesShouldExist[f];

            if (currentlyExists && !shouldExist)
            {
                // Remove the face
                // Zero out vertices and indices so it doesn't render
                unsigned int vOffset = existingFaceIndex * 4;
                unsigned int iOffset = existingFaceIndex * 6;

                for (int i = 0; i < 4; i++)
                {
                    attr[vOffset + i].vertex = Float3(0.0f, 0.0f, 0.0f);
                }
                for (int i = 0; i < 6; i++)
                {
                    indices[iOffset + i] = 0; // or invalid index, if the renderer checks this
                }

                // Mark this face slot as free for reuse
                freeFaces.push_back((unsigned int)existingFaceIndex);
                faceLocation[faceId] = UINT_MAX; // face no longer exists
            }
            else if (!currentlyExists && shouldExist)
            {
                // Add a new face
                unsigned int newFaceIndex;
                if (!freeFaces.empty())
                {
                    // Reuse a freed slot
                    newFaceIndex = freeFaces.back();
                    freeFaces.pop_back();
                }
                else
                {
                    // Append at the end if we have capacity
                    if (currentFaceCount >= maxFaceCount)
                    {
                        // Out of space - you may choose to reallocate or do a full rebuild
                        // For now, just return or handle error
                        return;
                    }
                    newFaceIndex = currentFaceCount;
                    currentFaceCount++;
                }

                Float3 verts[4];
                ComputeFaceVertices(centerPos, f, verts);

                unsigned int vOffset = newFaceIndex * 4;
                unsigned int iOffset = newFaceIndex * 6;

                // Write vertices
                for (int i = 0; i < 4; i++)
                {
                    attr[vOffset + i].vertex = verts[i];
                }

                // Write indices
                indices[iOffset + 0] = vOffset + 0;
                indices[iOffset + 1] = vOffset + 1;
                indices[iOffset + 2] = vOffset + 2;
                indices[iOffset + 3] = vOffset + 0;
                indices[iOffset + 4] = vOffset + 2;
                indices[iOffset + 5] = vOffset + 3;

                // Update mapping
                faceLocation[faceId] = newFaceIndex;
            }
            // If currentlyExists == shouldExist, no action needed
        }

        attrSize = currentFaceCount * 4;
        indicesSize = currentFaceCount * 6;

        if (0)
        {
            dumpMeshToOBJ(attr, indices, attrSize, indicesSize, "debug" + std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(z) + ".obj");
        }
    }
}
