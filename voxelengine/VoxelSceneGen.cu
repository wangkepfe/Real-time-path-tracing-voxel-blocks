#include "VoxelSceneGen.h"
#include "ChunkGeometryBuffer.h"

#include "Noise.h"

#include "util/KernelHelper.h"

#include "Block.h"

#include <cub/cub.cuh> // For prefix sums

#include <fstream>

__device__ __host__ void ComputeFaceVertices(Float3 basePos, int faceId, Float3 outVerts[4])
{
    switch (faceId)
    {
    case 0: // Up (top face at y+1)
        outVerts[0] = basePos + Float3(0.0f, 1.0f, 0.0f);
        outVerts[1] = basePos + Float3(0.0f, 1.0f, 1.0f);
        outVerts[2] = basePos + Float3(1.0f, 1.0f, 1.0f);
        outVerts[3] = basePos + Float3(1.0f, 1.0f, 0.0f);
        break;

    case 1: // Down (bottom face at y)
        outVerts[0] = basePos + Float3(1.0f, 0.0f, 0.0f);
        outVerts[1] = basePos + Float3(1.0f, 0.0f, 1.0f);
        outVerts[2] = basePos + Float3(0.0f, 0.0f, 1.0f);
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

__global__ void GenerateVoxelChunk(Voxel *voxels, float *noise, unsigned int width)
{
    Int3 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    idx.z = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx.x >= width || idx.y >= width || idx.z >= width)
        return;

    Voxel val;
    val.id = BlockTypeEmpty;

    float noiseVal = noise[idx.z * width + idx.x];

    float terrainHeight = max(0.1f, (noiseVal * 1.4f - 0.7f + 0.25f) * width);
    terrainHeight = min(terrainHeight, width * 0.9f);
    if (idx.y < terrainHeight)
    {
        float verticalDepth = terrainHeight - idx.y;

        if (terrainHeight < width * (0.25f + 0.05f))
        {
            if (verticalDepth < 3.5f)
            {
                val.id = BlockTypeSand;
            }
            else
            {
                val.id = BlockTypeRocks;
            }
        }
        else if (terrainHeight < width * (0.25f + 0.6f) && terrainHeight > width * (0.25f + 0.3f))
        {
            if (verticalDepth < 5.5f)
            {
                val.id = BlockTypeCliff;
            }
            else
            {
                val.id = BlockTypeRocks;
            }
        }
        else
        {
            if (verticalDepth < 1.5f)
            {
                val.id = BlockTypeSoil;
            }
            else if (verticalDepth < 5.5f)
            {
                val.id = BlockTypeCliff;
            }
            else
            {
                val.id = BlockTypeRocks;
            }
        }
    }

    // HACK DISABLED - Engine must be robust enough to handle missing block types
    // for (int i = 1; i < BlockTypeNum; ++i)
    // {
    //     if (idx.x == 1 && idx.y == 1 && idx.z == i)
    //     {
    //         val.id = i;
    //     }
    // }

    unsigned int linearId = GetLinearId(idx.x, idx.y, idx.z, width);
    voxels[linearId] = val;
}

__global__ void MarkValidFaces(
    Voxel *voxels,
    unsigned int *d_faceValid,
    unsigned int width,
    int id)
{
    Int3 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    idx.z = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx.x >= width || idx.y >= width || idx.z >= width)
        return;

    unsigned int linearId = GetLinearId(idx.x, idx.y, idx.z, width);
    Voxel center = voxels[linearId];

    if (center.id != id)
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
            if (neighbor.id == id)
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
    unsigned int width,
    int id)
{
    Int3 idx;
    idx.x = blockIdx.x * blockDim.x + threadIdx.x;
    idx.y = blockIdx.y * blockDim.y + threadIdx.y;
    idx.z = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx.x >= width || idx.y >= width || idx.z >= width)
        return;

    unsigned int linearId = GetLinearId(idx.x, idx.y, idx.z, width);
    Voxel center = voxels[linearId];
    if (center.id != id)
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

            // Two triangles
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
        const VertexAttributes *attr,
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

    bool isNeighborEmpty(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int width, const Voxel *voxels, int id, int oldId)
    {
        if (nx >= width || ny >= width || nz >= width)
            return true; // Out of bounds => treat as empty
        Voxel n = voxels[GetLinearId(nx, ny, nz, width)];
        
        // FACE VISIBILITY FIX: A neighbor is empty if it has no block (id == 0)
        // When placing a block (id != 0), face shows if neighbor is empty
        // When removing a block (id == 0), face shows if neighbor has a block
        return n.id == 0;
    }
}

// New multi-chunk initialization function
void initVoxelsMultiChunk(VoxelChunk &voxelChunk, Voxel **d_data, unsigned int chunkIndex,
                          const ChunkConfiguration &chunkConfig)
{
    dim3 blockDim = GetBlockDim(BLOCK_DIM_4x4x4);
    dim3 gridDim = GetGridDim(voxelChunk.width, voxelChunk.width, voxelChunk.width, BLOCK_DIM_4x4x4);

    size_t totalVoxels = voxelChunk.width * voxelChunk.width * voxelChunk.width;
    size_t noiseMapSize = voxelChunk.width * voxelChunk.width;

    cudaMalloc(d_data, totalVoxels * sizeof(Voxel));

    // Calculate chunk coordinates
    unsigned int chunkX = chunkIndex % chunkConfig.chunksX;
    unsigned int chunkZ = (chunkIndex / chunkConfig.chunksX) % chunkConfig.chunksZ;
    unsigned int chunkY = chunkIndex / (chunkConfig.chunksX * chunkConfig.chunksZ);

    // Calculate global offset for this chunk
    unsigned int globalOffsetX = chunkX * VoxelChunk::width;
    unsigned int globalOffsetZ = chunkZ * VoxelChunk::width;

    PerlinNoiseGenerator noiseGenerator(4, 124); // Use same seed for all chunks
    constexpr float freq = 1.0f / 64.0f;

    std::vector<float> noiseMap(noiseMapSize);
    for (int x = 0; x < voxelChunk.width; ++x)
    {
        for (int z = 0; z < voxelChunk.width; ++z)
        {
            // Use global coordinates for noise sampling to ensure continuity
            float globalX = globalOffsetX + x;
            float globalZ = globalOffsetZ + z;
            float noiseValue = noiseGenerator.getNoise(globalX * freq, globalZ * freq);
            noiseMap[z * voxelChunk.width + x] = noiseValue;
        }
    }

    float *d_noise;
    cudaMalloc(&d_noise, noiseMapSize * sizeof(float));
    cudaMemcpy(d_noise, noiseMap.data(), noiseMapSize * sizeof(float), cudaMemcpyHostToDevice);

    GenerateVoxelChunk KERNEL_ARGS2(gridDim, blockDim)(*d_data, d_noise, voxelChunk.width);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

    cudaMemcpy(voxelChunk.data, *d_data, totalVoxels * sizeof(Voxel), cudaMemcpyDeviceToHost);
    cudaFree(d_noise);
}

void freeDeviceVoxelData(Voxel *d_data)
{
    CUDA_CHECK(cudaFree(d_data));
}

void generateMesh(VertexAttributes **attr,
                  unsigned int **indices,
                  std::vector<unsigned int> &faceLocation,
                  unsigned int &attrSize,
                  unsigned int &indicesSize,
                  unsigned int &currentFaceCount,
                  unsigned int &maxFaceCount,
                  VoxelChunk &voxelChunk,
                  Voxel *d_data,
                  int id)
{
    dim3 blockDim = GetBlockDim(BLOCK_DIM_4x4x4);
    dim3 gridDim = GetGridDim(voxelChunk.width, voxelChunk.width, voxelChunk.width, BLOCK_DIM_4x4x4);

    unsigned int *d_faceValid;
    unsigned int *d_faceLocation;
    size_t totalVoxels = voxelChunk.width * voxelChunk.width * voxelChunk.width;
    size_t totalFaces = totalVoxels * 6;
    cudaMalloc(&d_faceValid, totalFaces * sizeof(unsigned int));
    cudaMalloc(&d_faceLocation, totalFaces * sizeof(unsigned int));

    // 2. Mark valid faces
    MarkValidFaces KERNEL_ARGS2(gridDim, blockDim)(d_data, d_faceValid, voxelChunk.width, id);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

    // std::vector<unsigned int> h_faceValid(totalFaces);
    // cudaMemcpy(h_faceValid.data(), d_faceValid, totalFaces * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // std::cout << "face valid id = " << id << std::endl;
    // for (auto val : h_faceValid)
    // {
    //     std::cout << val << " ";
    // }
    // std::cout << std::endl;

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

    // Handle zero instances case - early exit with empty geometry
    if (currentFaceCount == 0)
    {
        // DYNAMIC PLACEMENT FIX: Set minimum capacity for future block placement
        maxFaceCount = 1000; // Reserve space for ~166 blocks (6 faces each)
        attrSize = 0;
        indicesSize = 0;
        printf("INIT DEBUG: Chunk empty, setting maxFaceCount=%u for future block placement\n", maxFaceCount);

        // ALLOCATE BUFFERS: Even for empty chunks, allocate buffers for future use
        if (*attr != nullptr)
        {
            cudaFree(*attr);
        }
        if (*indices != nullptr)
        {
            cudaFree(*indices);
        }
        
        // Allocate buffers with capacity for maxFaceCount faces
        cudaMalloc(attr, maxFaceCount * 4 * sizeof(VertexAttributes));
        cudaMalloc(indices, maxFaceCount * 6 * sizeof(unsigned int));
        printf("INIT DEBUG: Allocated buffers for %u faces (%zu bytes)\n", 
               maxFaceCount, maxFaceCount * (4 * sizeof(VertexAttributes) + 6 * sizeof(unsigned int)));

        // Initialize empty face location vector
        faceLocation.assign(totalFaces, UINT_MAX);

        // Cleanup and return early
        cudaFree(d_faceValid);
        cudaFree(d_faceLocation);
        cudaFree(d_faceValidPrefixSum);
        return;
    }

    // maxFaceCount = currentFaceCount * 2;
    maxFaceCount = (totalFaces / 4 > 1u) ? (totalFaces / 4) : 1u; // Ensure minimum of 1 to avoid zero allocation

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

    cudaMallocManaged(attr, maxFaceCount * 4 * sizeof(VertexAttributes));
    cudaMallocManaged(indices, maxFaceCount * 6 * sizeof(unsigned int));

    // 5. Compact the mesh
    CompactMesh KERNEL_ARGS2(gridDim, blockDim)(*attr, *indices, d_faceLocation, d_data, d_faceValidPrefixSum, voxelChunk.width, id);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

    faceLocation.resize(totalFaces);
    cudaMemcpy(faceLocation.data(), d_faceLocation, totalFaces * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    if (0)
    {
        dumpMeshToOBJ(*attr, *indices, attrSize, indicesSize, "debug" + std::to_string(id) + ".obj");
    }

    // Cleanup
    cudaFree(d_faceValid);
    cudaFree(d_faceLocation);
    cudaFree(d_faceValidPrefixSum);
}

void updateSingleFace(
    int faceId,
    std::vector<unsigned int> &faceLocation,
    bool shouldExist,
    int f,
    VertexAttributes *attr,
    unsigned int *indices,
    std::vector<unsigned int> &freeFaces,
    unsigned int &currentFaceCount,
    unsigned int &maxFaceCount,
    Float3 centerPos)
{
    unsigned int existingFaceIndex = faceLocation[faceId];
    bool currentlyExists = (existingFaceIndex != UINT_MAX);

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
                // EMERGENCY FIX: If maxFaceCount is 0, the chunk wasn't properly initialized
                if (maxFaceCount == 0) {
                    printf("FACE DEBUG: EMERGENCY - maxFaceCount=0, cannot create faces! Chunk needs initialization.\n");
                    return;
                }
                
                // Out of space - you may choose to reallocate or do a full rebuild
                printf("FACE DEBUG: ERROR - No space for new face! currentFaceCount=%u >= maxFaceCount=%u\n", 
                       currentFaceCount, maxFaceCount);
                return;
            }
            newFaceIndex = currentFaceCount;
            currentFaceCount++;
        }

        printf("FACE DEBUG: SUCCESS - Creating face %d at index %u (currentFaceCount now %u)\n", 
               f, newFaceIndex, currentFaceCount);

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
}

bool getColocatedFace(int &colocatedVid, int &colocatedFaceF, Float3 &colocatedCenterPos, unsigned int x, unsigned int y, unsigned int z, int f, VoxelChunk &voxelChunk)
{
    // Remember:
    // f = 0 => Up    => neighbor in +Y direction => opposite face is 1 (Down)
    // f = 1 => Down  => neighbor in -Y direction => opposite face is 0 (Up)
    // f = 2 => Left  => neighbor in -X direction => opposite face is 3 (Right)
    // f = 3 => Right => neighbor in +X direction => opposite face is 2 (Left)
    // f = 4 => Back  => neighbor in +Z direction => opposite face is 5 (Front)
    // f = 5 => Front => neighbor in -Z direction => opposite face is 4 (Back)

    // Check each face direction, ensure we stay within [0, width-1] bounds
    if (f == 0 && y < voxelChunk.width - 1) // Up => neighbor is (x, y+1, z)
    {
        colocatedVid = GetLinearId(x, y + 1, z, voxelChunk.width);
        colocatedFaceF = 1; // Down
        colocatedCenterPos = Float3((float)x, (float)(y + 1), (float)z);
        return true;
    }
    else if (f == 1 && y > 0) // Down => neighbor is (x, y-1, z)
    {
        colocatedVid = GetLinearId(x, y - 1, z, voxelChunk.width);
        colocatedFaceF = 0; // Up
        colocatedCenterPos = Float3((float)x, (float)(y - 1), (float)z);
        return true;
    }
    else if (f == 2 && x > 0) // Left => neighbor is (x-1, y, z)
    {
        colocatedVid = GetLinearId(x - 1, y, z, voxelChunk.width);
        colocatedFaceF = 3; // Right
        colocatedCenterPos = Float3((float)(x - 1), (float)y, (float)z);
        return true;
    }
    else if (f == 3 && x < voxelChunk.width - 1) // Right => neighbor is (x+1, y, z)
    {
        colocatedVid = GetLinearId(x + 1, y, z, voxelChunk.width);
        colocatedFaceF = 2; // Left
        colocatedCenterPos = Float3((float)(x + 1), (float)y, (float)z);
        return true;
    }
    else if (f == 4 && z < voxelChunk.width - 1) // Back => neighbor is (x, y, z+1)
    {
        colocatedVid = GetLinearId(x, y, z + 1, voxelChunk.width);
        colocatedFaceF = 5; // Front
        colocatedCenterPos = Float3((float)x, (float)y, (float)(z + 1));
        return true;
    }
    else if (f == 5 && z > 0) // Front => neighbor is (x, y, z-1)
    {
        colocatedVid = GetLinearId(x, y, z - 1, voxelChunk.width);
        colocatedFaceF = 4; // Back
        colocatedCenterPos = Float3((float)x, (float)y, (float)(z - 1));
        return true;
    }

    // If no valid opposite/neighbor within bounds was found, return false
    return false;
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
    VertexAttributes *attr,
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

    // DEBUG: Print voxel state change
    printf("VOXEL DEBUG: Position (%u,%u,%u) oldVal=%u newVal=%u (changed=%s)\n",
           x, y, z, oldVoxel.id, newVal, (oldVoxel.id != newVal) ? "YES" : "NO");
    
    // DEBUG: Check surrounding voxels to understand the context
    printf("CONTEXT DEBUG: Checking neighbors of (%u,%u,%u):\n", x, y, z);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue; // Skip center voxel
                int nx = (int)x + dx;
                int ny = (int)y + dy; 
                int nz = (int)z + dz;
                if (nx >= 0 && nx < (int)voxelChunk.width && 
                    ny >= 0 && ny < (int)voxelChunk.width && 
                    nz >= 0 && nz < (int)voxelChunk.width) {
                    unsigned int nid = GetLinearId(nx, ny, nz, voxelChunk.width);
                    printf("  neighbor (%d,%d,%d): ID=%u\n", nx, ny, nz, voxelChunk.data[nid].id);
                }
            }
        }
    }

    // If voxel state didn't change, no update needed
    if (oldVoxel.id == newVal) {
        printf("VOXEL DEBUG: Skipping update - voxel already has value %u\n", newVal);
        return;
    }

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
        bool neighEmpty = isNeighborEmpty(nx, ny, nz, voxelChunk.width, voxelChunk.data, newVal, oldVoxel.id);
        facesShouldExist[f] = ((newVal != 0) && neighEmpty) || ((newVal == 0) && !neighEmpty);
    }

    Float3 centerPos((float)x, (float)y, (float)z);
    
    // DEBUG: Print the coordinates being used for face generation
    if (newVal != 0) {
        printf("FACE DEBUG: updateSingleVoxel called with local coords (%u, %u, %u) → centerPos(%.1f, %.1f, %.1f)\n",
               x, y, z, centerPos.x, centerPos.y, centerPos.z);
        
        // Count how many faces will be created for this voxel
        int facesToCreate = 0;
        for (int f = 0; f < 6; f++) {
            Int3 dir = faceDirections[f];
            unsigned int nx = x + dir.x;
            unsigned int ny = y + dir.y;
            unsigned int nz = z + dir.z;
            bool neighEmpty = isNeighborEmpty(nx, ny, nz, voxelChunk.width, voxelChunk.data, newVal, oldVoxel.id);
            if ((newVal != 0) && neighEmpty) {
                facesToCreate++;
            }
        }
        printf("FACE DEBUG: Will create %d faces for this block\n", facesToCreate);
    }

    for (int f = 0; f < 6; f++)
    {
        int faceId = (int)(vid * 6 + f);
        bool shouldExist = (newVal != 0) ? facesShouldExist[f] : false;

        printf("FACE DEBUG: Face %d shouldExist=%s\n", f, shouldExist ? "YES" : "NO");
        updateSingleFace(
            faceId,
            faceLocation,
            shouldExist,
            f,
            attr,
            indices,
            freeFaces,
            currentFaceCount,
            maxFaceCount,
            centerPos);
        printf("FACE DEBUG: After updateSingleFace, currentFaceCount=%u\n", currentFaceCount);

        int colocatedFaceF;
        int colocatedVid;
        Float3 colocatedCenterPos;
        bool hasColocatedFace = getColocatedFace(colocatedVid, colocatedFaceF, colocatedCenterPos, x, y, z, f, voxelChunk);
        if (hasColocatedFace)
        {
            int colocatedFaceId = colocatedVid * 6 + colocatedFaceF;

            shouldExist = (newVal != 0) ? false : facesShouldExist[f];

            updateSingleFace(
                colocatedFaceId,
                faceLocation,
                shouldExist,
                colocatedFaceF,
                attr,
                indices,
                freeFaces,
                currentFaceCount,
                maxFaceCount,
                colocatedCenterPos);
        }
    }

    attrSize = currentFaceCount * 4;
    indicesSize = currentFaceCount * 6;
    
    printf("FACE DEBUG: Final geometry - currentFaceCount=%u, attrSize=%u, indicesSize=%u\n",
           currentFaceCount, attrSize, indicesSize);

    if (0)
    {
        dumpMeshToOBJ(attr, indices, attrSize, indicesSize, "debug" + std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(z) + ".obj");
    }
}

void updateSingleVoxelGlobal(
    unsigned int globalX,
    unsigned int globalY,
    unsigned int globalZ,
    unsigned int newVal,
    std::vector<VoxelChunk> &voxelChunks,
    const ChunkConfiguration &chunkConfig,
    VertexAttributes *attr,
    unsigned int *indices,
    std::vector<unsigned int> &faceLocation,
    unsigned int &attrSize,
    unsigned int &indicesSize,
    unsigned int &currentFaceCount,
    unsigned int &maxFaceCount,
    std::vector<unsigned int> &freeFaces)
{
    // Convert global coordinates to chunk coordinates
    unsigned int chunkX = globalX / VoxelChunk::width;
    unsigned int chunkY = globalY / VoxelChunk::width;
    unsigned int chunkZ = globalZ / VoxelChunk::width;

    unsigned int localX = globalX % VoxelChunk::width;
    unsigned int localY = globalY % VoxelChunk::width;
    unsigned int localZ = globalZ % VoxelChunk::width;

    // Check bounds
    if (chunkX >= chunkConfig.chunksX || chunkY >= chunkConfig.chunksY || chunkZ >= chunkConfig.chunksZ)
        return;

    // Calculate chunk index
    unsigned int chunkIndex = chunkX + chunkConfig.chunksX * (chunkZ + chunkConfig.chunksZ * chunkY);

    // Update the voxel in the appropriate chunk
    VoxelChunk &targetChunk = voxelChunks[chunkIndex];
    
    // DEBUG: Print coordinate conversion
    printf("COORD DEBUG: Global(%u,%u,%u) → Chunk(%u,%u,%u) Local(%u,%u,%u) ChunkIndex=%u\n",
           globalX, globalY, globalZ, chunkX, chunkY, chunkZ, localX, localY, localZ, chunkIndex);
    
    printf("CALL DEBUG: About to call updateSingleVoxel with newVal=%u\n", newVal);
    updateSingleVoxel(
        localX, localY, localZ,
        newVal,
        targetChunk,
        attr,
        indices,
        faceLocation,
        attrSize,
        indicesSize,
        currentFaceCount,
        maxFaceCount,
        freeFaces);
    printf("CALL DEBUG: updateSingleVoxel completed\n");
}

void generateSea(VertexAttributes **attr,
                 unsigned int **indices,
                 unsigned int &attrSize,
                 unsigned int &indicesSize,
                 int width)
{
    float xMax = (float)width - 0.1f;
    float xMin = 0.1f;
    float yMax = (float)(width / 4) - 0.1f;
    float yMin = 0.1f;
    float zMax = (float)width - 0.1f;
    float zMin = 0.1f;

    float scaleX = xMax - xMin;
    float scaleY = yMax - yMin;
    float scaleZ = zMax - zMin;

    if (*attr != nullptr)
    {
        cudaFree(*attr);
    }
    if (*indices != nullptr)
    {
        cudaFree(*indices);
    }

    int faceCount = 6;

    cudaMalloc(attr, faceCount * 4 * sizeof(VertexAttributes));
    cudaMalloc(indices, faceCount * 6 * sizeof(unsigned int));

    attrSize = faceCount * 4;
    indicesSize = faceCount * 6;

    VertexAttributes *d_attrOut = *attr;
    unsigned int *d_idxOut = *indices;

    std::vector<VertexAttributes> attrOut(faceCount * 4);
    std::vector<unsigned int> idxOut(faceCount * 6);

    VertexAttributes *h_attrOut = attrOut.data();
    unsigned int *h_idxOut = idxOut.data();

    for (int f = 0; f < faceCount; f++)
    {
        unsigned int newFaceIndex = f; // We can treat f as the face index

        unsigned int vOffset = newFaceIndex * 4; // each face has 4 vertices
        unsigned int iOffset = newFaceIndex * 6; // each face has 6 indices

        Float3 verts[4];
        ComputeFaceVertices(Float3(0.0f, 0.0f, 0.0f), f, verts);

        for (int i = 0; i < 4; i++)
        {
            Float3 &v = verts[i];
            float scaledX = xMin + v.x * scaleX;
            float scaledY = yMin + v.y * scaleY;
            float scaledZ = zMin + v.z * scaleZ;

            h_attrOut[vOffset + i].vertex.x = scaledX;
            h_attrOut[vOffset + i].vertex.y = scaledY;
            h_attrOut[vOffset + i].vertex.z = scaledZ;
        }

        // Set up the two triangles for this face
        h_idxOut[iOffset + 0] = vOffset + 0;
        h_idxOut[iOffset + 1] = vOffset + 1;
        h_idxOut[iOffset + 2] = vOffset + 2;

        h_idxOut[iOffset + 3] = vOffset + 0;
        h_idxOut[iOffset + 4] = vOffset + 2;
        h_idxOut[iOffset + 5] = vOffset + 3;
    }

    cudaMemcpy(d_attrOut, h_attrOut, faceCount * 4 * sizeof(VertexAttributes), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idxOut, h_idxOut, faceCount * 6 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    if (0)
    {
        dumpMeshToOBJ(h_attrOut, h_idxOut, attrSize, indicesSize, "debug_sea.obj");
    }
}

// New generateMesh overload using ChunkGeometryBuffer
void generateMesh(VertexAttributes **attr,
                  unsigned int **indices,
                  unsigned int &attrSize,
                  unsigned int &indicesSize,
                  ChunkGeometryBuffer &geometryBuffer,
                  VoxelChunk &voxelChunk,
                  Voxel *d_data,
                  int id)
{
    dim3 blockDim = GetBlockDim(BLOCK_DIM_4x4x4);
    dim3 gridDim = GetGridDim(voxelChunk.width, voxelChunk.width, voxelChunk.width, BLOCK_DIM_4x4x4);

    unsigned int *d_faceValid;
    unsigned int *d_faceLocation;
    size_t totalVoxels = voxelChunk.width * voxelChunk.width * voxelChunk.width;
    size_t totalFaces = totalVoxels * 6;
    cudaMalloc(&d_faceValid, totalFaces * sizeof(unsigned int));
    cudaMalloc(&d_faceLocation, totalFaces * sizeof(unsigned int));

    // Mark valid faces
    MarkValidFaces KERNEL_ARGS2(gridDim, blockDim)(d_data, d_faceValid, voxelChunk.width, id);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

    // Prefix sum on face validity
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

    // Find how many faces are valid total
    unsigned int currentFaceCount = 0;
    cudaMemcpy(&currentFaceCount, &d_faceValidPrefixSum[totalFaces - 1], sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Handle zero instances case - use geometry buffer to allocate capacity
    if (currentFaceCount == 0)
    {
        attrSize = 0;
        indicesSize = 0;
        
        // GeometryBuffer is already initialized with capacity, just reset
        geometryBuffer.reset();
        
        // Set output pointers to geometry buffer (which has capacity allocated)
        *attr = geometryBuffer.getVertexBuffer();
        *indices = geometryBuffer.getIndexBuffer();
        
        printf("GEOMETRY BUFFER DEBUG: Empty chunk, using buffer capacity %u\n", geometryBuffer.getCapacity());

        // Cleanup and return early
        cudaFree(d_faceValid);
        cudaFree(d_faceLocation);
        cudaFree(d_faceValidPrefixSum);
        return;
    }

    // Ensure geometry buffer has enough capacity
    if (!geometryBuffer.ensureCapacity(currentFaceCount)) {
        printf("GEOMETRY BUFFER ERROR: Failed to ensure capacity for %u faces\n", currentFaceCount);
        cudaFree(d_faceValid);
        cudaFree(d_faceLocation);
        cudaFree(d_faceValidPrefixSum);
        return;
    }

    attrSize = currentFaceCount * 4;
    indicesSize = currentFaceCount * 6;

    // Get buffers from geometry manager
    *attr = geometryBuffer.getVertexBuffer();
    *indices = geometryBuffer.getIndexBuffer();

    // Reset the geometry buffer and set face count
    geometryBuffer.reset();
    // Manually set the face count (simulating allocation)
    for (unsigned int i = 0; i < currentFaceCount; i++) {
        geometryBuffer.allocateFace();
    }

    // 5. Compact the mesh directly
    CompactMesh KERNEL_ARGS2(gridDim, blockDim)(*attr, *indices, d_faceLocation, d_data, d_faceValidPrefixSum, voxelChunk.width, id);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaPeekAtLastError());

    // Cleanup
    cudaFree(d_faceValid);
    cudaFree(d_faceLocation);
    cudaFree(d_faceValidPrefixSum);

    printf("GEOMETRY BUFFER: Generated mesh with %u faces, using buffer capacity %u\n", 
           currentFaceCount, geometryBuffer.getCapacity());
}
