#include "VoxelEngine.h"
#include "VoxelMath.h"
#include "Block.h"

#include "core/Scene.h"
#include "core/InputHandler.h"
#include "core/RenderCamera.h"

#include "util/ModelUtils.h"
#include "shaders/ShaderDebugUtils.h"

// Helper function: applies a 3x4 transform (stored as 12 floats in row-major order)
// to a given 3D position.
__device__ inline Float3 applyTransform(const Float3 &v, const float *t)
{
    // t[0..3] is the first row, t[4..7] is the second row, t[8..11] is the third row.
    return Float3(
        t[0] * v.x + t[1] * v.y + t[2] * v.z + t[3],
        t[4] * v.x + t[5] * v.y + t[6] * v.z + t[7],
        t[8] * v.x + t[9] * v.y + t[10] * v.z + t[11]);
}

//-----------------------------------------------------------------------------
// CUDA kernel to generate LightInfo entries for each instance of a mesh.
// Inputs:
//   - vertices: pointer to VertexAttributes array (each must contain a member "position" of type Float3).
//   - indices: pointer to unsigned int index array (assumed to form triangles).
//   - numIndices: total number of indices (must be a multiple of 3).
//   - transforms: pointer to a flattened array of instance transforms (each instance has 12 floats).
//   - numInstances: number of instance transforms.
//   - radiance: emissive radiance value (a scalar); to get radiant intensity multiply by area.
// Output:
//   - output: pointer to device buffer of LightInfo, with length = (numIndices/3) * numInstances.
__global__ void generateLightInfosKernel(const VertexAttributes *vertices,
                                         const unsigned int *indices,
                                         unsigned int numIndices,
                                         const float *transforms, // flattened; each instance: 12 floats.
                                         unsigned int numInstances,
                                         Float3 radiance,
                                         LightInfo *globalLights,
                                         unsigned int globalOffset)
{
    // Compute total number of triangles in the mesh.
    unsigned int numTriangles = numIndices / 3;

    // Each thread handles one triangle for one instance.
    unsigned int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int totalWork = numTriangles * numInstances;
    if (globalId >= totalWork)
        return;

    // Map the global id to an instance index and a triangle index.
    unsigned int instanceIdx = globalId / numTriangles;
    unsigned int triangleIdx = globalId % numTriangles;

    // Pointer to the current instance's transform (12 floats).
    const float *t = &transforms[instanceIdx * 12];

    // Read the three vertex indices for this triangle.
    unsigned int idx0 = indices[triangleIdx * 3 + 0];
    unsigned int idx1 = indices[triangleIdx * 3 + 1];
    unsigned int idx2 = indices[triangleIdx * 3 + 2];

    // Load the triangle’s vertices (assuming "position" is a Float3 member of VertexAttributes).
    Float3 v0ModelSpace = vertices[idx0].vertex;
    Float3 v1ModelSpace = vertices[idx1].vertex;
    Float3 v2ModelSpace = vertices[idx2].vertex;

    // Transform each vertex by the instance transform.
    Float3 v0 = applyTransform(v0ModelSpace, t);
    Float3 v1 = applyTransform(v1ModelSpace, t);
    Float3 v2 = applyTransform(v2ModelSpace, t);

    // Create a TriangleLight. We choose v0 as the "base" and compute edge vectors:
    TriangleLight triLight;
    triLight.base = v0;       // Base point.
    triLight.edge1 = v1 - v0; // Edge from v0 to v1.
    triLight.edge2 = v2 - v0; // Edge from v0 to v2.

    // Compute the area of the triangle.
    Float3 crossVal = cross(triLight.edge1, triLight.edge2);
    float area = 0.5f * length(crossVal);

    triLight.radiance = radiance;

    // Pack the triangle light data into a LightInfo using the provided interface.
    LightInfo li = triLight.Store();

    // Write out the result.
    globalLights[globalOffset + globalId] = li;
}

//-----------------------------------------------------------------------------
// Example host code to launch the kernel.
//
// Note: Adjust grid/block sizes as needed and ensure that the device arrays
//       (d_vertices, d_indices, d_transforms, d_lightInfos) are allocated and filled.
void launchGenerateLightInfos(const VertexAttributes *d_vertices,
                              const unsigned int *d_indices,
                              unsigned int numIndices,
                              const float *d_transforms, // device array with numInstances*12 floats
                              unsigned int numInstances,
                              Float3 radiance,
                              LightInfo *d_lightInfos,
                              unsigned int &currentGlobalOffset) // output buffer of size (numIndices/3)*numInstances
{
    unsigned int numTriangles = numIndices / 3;
    unsigned int totalWork = numTriangles * numInstances; // Number of triangle lights for the current type of geometry
    const unsigned int blockSize = 256;
    unsigned int numBlocks = (totalWork + blockSize - 1) / blockSize;

    generateLightInfosKernel<<<numBlocks, blockSize>>>(d_vertices,
                                                       d_indices,
                                                       numIndices,
                                                       d_transforms,
                                                       numInstances,
                                                       radiance,
                                                       d_lightInfos,
                                                       currentGlobalOffset);

    currentGlobalOffset += totalWork;

    // Check for errors and synchronize as needed.
    cudaDeviceSynchronize();
}

__global__ void extractRadianceKernel(const LightInfo *lights, float *d_radiance, unsigned int numLights)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numLights)
    {
        TriangleLight triLight = TriangleLight::Create(lights[idx]);
        d_radiance[idx] = luminance(triLight.radiance) * triLight.surfaceArea;
    }
}

//-----------------------------------------------------------------------------
// Step 2. Build the device radiance array and update the alias table.
void buildAliasTable(LightInfo *d_lights, unsigned int totalLights, AliasTable &aliasTable)
{
    // Allocate device memory to store one radiance weight per light.
    float *d_radiance = nullptr;
    cudaError_t err = cudaMalloc((void **)&d_radiance, totalLights * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "Error allocating radiance buffer: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Launch a kernel to extract radiance weights from the merged LightInfo buffer.
    const unsigned int blockSize = 256;
    unsigned int numBlocks = (totalLights + blockSize - 1) / blockSize;
    extractRadianceKernel<<<numBlocks, blockSize>>>(d_lights, d_radiance, totalLights);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "Error synchronizing after extractRadianceKernel: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_radiance);
        return;
    }

    // Build the alias table based on the radiance weights.
    float sumWeight = 0.0f;
    aliasTable.update(d_radiance, totalLights, sumWeight);

    // Optionally, you can free the d_radiance buffer if it is no longer needed.
    cudaFree(d_radiance);

    // (For debugging, you might want to print out sumWeight or check the alias table.)
}

unsigned int PositionToInstanceId(unsigned int offset, unsigned int geometryId, unsigned int x, unsigned int y, unsigned int z, unsigned int width)
{
    unsigned int linearId = GetLinearId(x, y, z, width);
    return offset + geometryId * width * width * width + linearId;
}

static void MouseButtonCallback(int button, int action, int mods)
{
    if (button == 0 && action == 1)
    {
        VoxelEngine::Get().leftMouseButtonClicked = true;
    }
}

VoxelEngine::~VoxelEngine()
{
}

void VoxelEngine::initInstanceGeometry()
{
    auto &scene = Scene::Get();
    auto &sceneGeometryAttributes = scene.m_geometryAttibutes;
    auto &sceneGeometryIndices = scene.m_geometryIndices;
    auto &sceneGeometryAttributeSize = scene.m_geometryAttibuteSize;
    auto &sceneGeometryIndicesSize = scene.m_geometryIndicesSize;

    for (unsigned int objectId = GetInstancedObjectIdBegin(); objectId < GetInstancedObjectIdEnd(); ++objectId)
    {
        sceneGeometryAttributeSize[objectId] = 0;
        sceneGeometryIndicesSize[objectId] = 0;

        unsigned int blockId = ObjectIdToBlockId(objectId);

        std::string modelFileName = GetModelFileName(blockId);

        loadModel(&(sceneGeometryAttributes[objectId]),
                  &(sceneGeometryIndices[objectId]),
                  sceneGeometryAttributeSize[objectId],
                  sceneGeometryIndicesSize[objectId],
                  modelFileName);
    }
}

void VoxelEngine::updateInstances()
{
    auto &scene = Scene::Get();
    auto &sceneGeometryAttributes = scene.m_geometryAttibutes;
    auto &sceneGeometryIndices = scene.m_geometryIndices;
    auto &sceneGeometryIndicesSize = scene.m_geometryIndicesSize;

    // Load models for the instanced meshes
    std::unordered_map<int, std::set<int>> &geometryInstanceIdMap = scene.geometryInstanceIdMap;
    std::unordered_map<int, std::array<float, 12>> &instanceTransformMatrices = scene.instanceTransformMatrices;
    geometryInstanceIdMap.clear();
    instanceTransformMatrices.clear();

    // Load geometry of instance object
    unsigned int totalNumTriLights = 0;
    for (unsigned int objectId = GetInstancedObjectIdBegin(); objectId < GetInstancedObjectIdEnd(); ++objectId)
    {
        unsigned int blockId = ObjectIdToBlockId(objectId);
        for (unsigned int x = 0; x < voxelChunk.width; ++x)
        {
            for (unsigned int y = 0; y < voxelChunk.width; ++y)
            {
                for (unsigned int z = 0; z < voxelChunk.width; ++z)
                {
                    auto val = voxelChunk.get(x, y, z);
                    bool specialCase = (val.id == BlockTypeTestLightBase) && (blockId == BlockTypeTestLight);
                    if (val.id == blockId || specialCase)
                    {
                        unsigned int instanceId = PositionToInstanceId(GetNumUninstancedBlockTypes(), objectId, x, y, z, voxelChunk.width);

                        geometryInstanceIdMap[objectId].insert(instanceId);

                        std::array<float, 12> transform = {1.0f, 0.0f, 0.0f, (float)x,
                                                           0.0f, 1.0f, 0.0f, (float)y,
                                                           0.0f, 0.0f, 1.0f, (float)z};

                        instanceTransformMatrices[instanceId] = transform;
                    }
                }
            }
        }

        // Accumulate light count
        if (IsBlockEmissive(blockId))
        {
            unsigned int numInstances = geometryInstanceIdMap[objectId].size();
            unsigned int numTriPerInstance = sceneGeometryIndicesSize[objectId] / 3;
            unsigned int numTriLight = numTriPerInstance * numInstances;
            totalNumTriLights += numTriLight;
        }
    }

    // Allocate light info
    if (scene.m_lights != nullptr)
    {
        cudaFree(scene.m_lights);
    }
    cudaMalloc((void **)&scene.m_lights, totalNumTriLights * sizeof(LightInfo));

    // Generate light info
    unsigned int currentGlobalOffset = 0;
    scene.instanceLightMapping.clear();
    for (unsigned int objectId = GetInstancedObjectIdBegin(); objectId < GetInstancedObjectIdEnd(); ++objectId)
    {
        unsigned int blockId = ObjectIdToBlockId(objectId);
        if (IsBlockEmissive(blockId))
        {
            unsigned int numInstances = geometryInstanceIdMap[objectId].size();
            unsigned int numTriPerInstance = sceneGeometryIndicesSize[objectId] / 3;
            unsigned int numTriLight = numTriPerInstance * numInstances;

            std::vector<std::array<float, 12>> transforms;
            unsigned int lightOffset = currentGlobalOffset;
            for (unsigned int instanceId : geometryInstanceIdMap[objectId])
            {
                transforms.push_back(instanceTransformMatrices[instanceId]);
                scene.instanceLightMapping.push_back(InstanceLightMapping{instanceId, lightOffset, numTriPerInstance});
                lightOffset += numTriPerInstance;
            }

            Float3 radiance = GetEmissiveRadiance(blockId);

            float *d_transforms = nullptr;
            size_t transformsSizeInBytes = numInstances * 12 * sizeof(float);
            cudaMalloc((void **)&d_transforms, transformsSizeInBytes);
            cudaMemcpy(d_transforms, &(transforms[0][0]), transformsSizeInBytes, cudaMemcpyHostToDevice);

            launchGenerateLightInfos(sceneGeometryAttributes[objectId], sceneGeometryIndices[objectId], sceneGeometryIndicesSize[objectId], d_transforms, numInstances, radiance, scene.m_lights, currentGlobalOffset);

            cudaFree(d_transforms);
        }
    }

    // Upload instance light mapping
    if (scene.d_instanceLightMapping != nullptr)
    {
        cudaFree(scene.d_instanceLightMapping);
    }
    scene.numInstancedLightMesh = scene.instanceLightMapping.size();
    size_t mappingSizeInBytes = scene.numInstancedLightMesh * sizeof(InstanceLightMapping);
    cudaMalloc((void **)&scene.d_instanceLightMapping, mappingSizeInBytes);
    cudaMemcpy(scene.d_instanceLightMapping, scene.instanceLightMapping.data(), mappingSizeInBytes, cudaMemcpyHostToDevice);

    // Build alias table. If we have an existing table and the size is different
    if (scene.lightAliasTable.initialized() && scene.lightAliasTable.size() != totalNumTriLights)
    {
        scene.lightAliasTable = AliasTable(); // trigger the destructor and constructor
    }
    buildAliasTable(scene.m_lights, totalNumTriLights, scene.lightAliasTable);
    cudaMemcpy(scene.d_lightAliasTable, &scene.lightAliasTable, sizeof(AliasTable), cudaMemcpyHostToDevice);
}

void VoxelEngine::updateUninstancedMeshes(Voxel *d_data)
{
    auto &scene = Scene::Get();
    auto &sceneGeometryAttributes = scene.m_geometryAttibutes;
    auto &sceneGeometryIndices = scene.m_geometryIndices;
    auto &sceneGeometryAttributeSize = scene.m_geometryAttibuteSize;
    auto &sceneGeometryIndicesSize = scene.m_geometryIndicesSize;

    for (int objectId = GetUninstancedObjectIdBegin(); objectId < GetUninstancedObjectIdEnd(); ++objectId)
    {
        int blockId = ObjectIdToBlockId(objectId);

        sceneGeometryAttributeSize[objectId] = 0;
        sceneGeometryIndicesSize[objectId] = 0;

        currentFaceCount[objectId] = 0;
        maxFaceCount[objectId] = 0;

        generateMesh(
            &(sceneGeometryAttributes[objectId]),
            &(sceneGeometryIndices[objectId]),
            faceLocation[objectId],
            sceneGeometryAttributeSize[objectId],
            sceneGeometryIndicesSize[objectId],
            currentFaceCount[objectId],
            maxFaceCount[objectId],
            voxelChunk,
            d_data,
            blockId);
    }
}

void VoxelEngine::init()
{
    // Input handling
    auto &inputHandler = InputHandler::Get();
    inputHandler.setMouseButtonCallbackFunc(MouseButtonCallback);

    auto &scene = Scene::Get();
    auto &sceneGeometryAttributes = scene.m_geometryAttibutes;
    auto &sceneGeometryIndices = scene.m_geometryIndices;
    auto &sceneGeometryAttributeSize = scene.m_geometryAttibuteSize;
    auto &sceneGeometryIndicesSize = scene.m_geometryIndicesSize;

    scene.uninstancedGeometryCount = GetNumUninstancedBlockTypes();
    scene.instancedGeometryCount = GetNumInstancedBlockTypes();

    sceneGeometryAttributes.resize(GetNumUninstancedBlockTypes() + GetNumInstancedBlockTypes());
    sceneGeometryIndices.resize(GetNumUninstancedBlockTypes() + GetNumInstancedBlockTypes());
    sceneGeometryAttributeSize.resize(GetNumUninstancedBlockTypes() + GetNumInstancedBlockTypes());
    sceneGeometryIndicesSize.resize(GetNumUninstancedBlockTypes() + GetNumInstancedBlockTypes());

    faceLocation.resize(GetNumUninstancedBlockTypes());
    currentFaceCount.resize(GetNumUninstancedBlockTypes());
    maxFaceCount.resize(GetNumUninstancedBlockTypes());
    freeFaces.resize(GetNumUninstancedBlockTypes());

    // Init voxel
    Voxel *d_data;
    initVoxels(voxelChunk, &d_data);

    // Create geoemetry mesh based on the voxel grid
    updateUninstancedMeshes(d_data);
    freeDeviceVoxelData(d_data);

    // Init and update instanced meshes
    initInstanceGeometry();
    updateInstances();
}

void VoxelEngine::reload()
{
    // A hack to make sure all materials have a valid geometry
    for (int blockId = 1; blockId < BlockTypeNum; ++blockId)
    {
        voxelChunk.data[GetLinearId(1, 1, blockId, voxelChunk.width)] = blockId;
    }

    // Upload voxel data from disk load
    Voxel *d_data;
    size_t totalVoxels = voxelChunk.width * voxelChunk.width * voxelChunk.width;
    cudaMalloc(&d_data, totalVoxels * sizeof(Voxel));
    cudaMemcpy(d_data, voxelChunk.data, totalVoxels * sizeof(Voxel), cudaMemcpyHostToDevice);

    // Generate uninstanced meshes
    updateUninstancedMeshes(d_data);
    freeDeviceVoxelData(d_data);

    // Update instances
    updateInstances();

    Scene::Get().needSceneUpdate = true;
    Scene::Get().needSceneReloadUpdate = true;
}

void VoxelEngine::update()
{
    auto &camera = RenderCamera::Get().camera;
    Ray ray{camera.pos, camera.dir};

    bool hasSpaceToCreate = false;
    bool hitSurface = false;
    Int3 createPos(-1, -1, -1);
    Int3 deletePos(-1, -1, -1);
    int deleteBlockId = -1;

    cudaDeviceSynchronize();

    // Normalize the ray direction
    {
        float len = std::sqrt(ray.dir.x * ray.dir.x + ray.dir.y * ray.dir.y + ray.dir.z * ray.dir.z);
        if (len > 1e-8f)
        {
            ray.dir.x /= len;
            ray.dir.y /= len;
            ray.dir.z /= len;
        }
        else
        {
            // Degenerate direction vector: no traversal possible
            return; // Or handle error
        }
    }

    // Find the starting voxel indices
    int x = static_cast<int>(std::floor(ray.orig.x));
    int y = static_cast<int>(std::floor(ray.orig.y));
    int z = static_cast<int>(std::floor(ray.orig.z));

    // Determine step direction along each axis
    int stepX = (ray.dir.x > 0.0f) ? 1 : -1;
    int stepY = (ray.dir.y > 0.0f) ? 1 : -1;
    int stepZ = (ray.dir.z > 0.0f) ? 1 : -1;

    // Compute tDeltaX, tDeltaY, tDeltaZ
    float tDeltaX = (std::fabs(ray.dir.x) < 1e-8f) ? FLT_MAX : (1.0f / std::fabs(ray.dir.x));
    float tDeltaY = (std::fabs(ray.dir.y) < 1e-8f) ? FLT_MAX : (1.0f / std::fabs(ray.dir.y));
    float tDeltaZ = (std::fabs(ray.dir.z) < 1e-8f) ? FLT_MAX : (1.0f / std::fabs(ray.dir.z));

    float nextBoundaryX = (stepX > 0) ? (float)(x + 1) : (float)(x);
    float nextBoundaryY = (stepY > 0) ? (float)(y + 1) : (float)(y);
    float nextBoundaryZ = (stepZ > 0) ? (float)(z + 1) : (float)(z);

    float tMaxX = (std::fabs(ray.dir.x) < 1e-8f) ? FLT_MAX : (nextBoundaryX - ray.orig.x) / ray.dir.x;
    float tMaxY = (std::fabs(ray.dir.y) < 1e-8f) ? FLT_MAX : (nextBoundaryY - ray.orig.y) / ray.dir.y;
    float tMaxZ = (std::fabs(ray.dir.z) < 1e-8f) ? FLT_MAX : (nextBoundaryZ - ray.orig.z) / ray.dir.z;

    int iterationCount = 0;
    int maxIteration = 1000;

    int hitAxis = -1; // 0=x,1=y,2=z
    int hitX = -1, hitY = -1, hitZ = -1;

    // Traverse the voxel grid
    while (iterationCount++ < maxIteration)
    {
        // Check if the voxel is within bounds
        if (x < 0 || x >= voxelChunk.width ||
            y < 0 || y >= voxelChunk.width ||
            z < 0 || z >= voxelChunk.width)
        {
            // Out of bounds, stop
            break;
        }

        Voxel voxel = voxelChunk.get(x, y, z);

        if (voxel.id == 0)
        {
            // Empty voxel: we can potentially place something here
            hasSpaceToCreate = true;
            createPos = Int3(x, y, z);
            // Continue traversal to find a solid surface
        }
        else
        {
            // Solid voxel hit
            hitSurface = true;
            hitX = x;
            hitY = y;
            hitZ = z;
            deletePos = Int3(x, y, z);
            deleteBlockId = voxel.id;
            // We stop here
            break;
        }

        // Move to the next voxel:
        // Choose the axis with the smallest tMax
        if (tMaxX < tMaxY)
        {
            if (tMaxX < tMaxZ)
            {
                x += stepX;
                tMaxX += tDeltaX;
                hitAxis = 0; // moved along x-axis
            }
            else
            {
                z += stepZ;
                tMaxZ += tDeltaZ;
                hitAxis = 2; // moved along z-axis
            }
        }
        else
        {
            if (tMaxY < tMaxZ)
            {
                y += stepY;
                tMaxY += tDeltaY;
                hitAxis = 1; // moved along y-axis
            }
            else
            {
                z += stepZ;
                tMaxZ += tDeltaZ;
                hitAxis = 2; // moved along z-axis
            }
        }
    }

    if (hitSurface && hitX >= 0 && hitY >= 0 && hitZ >= 0)
    {
        // Determine which face was hit based on hitAxis and step directions
        Float3 corners[4];

        // Each voxel spans from (hitX, hitY, hitZ) to (hitX+1, hitY+1, hitZ+1).
        if (hitAxis == 0)
        {
            // Hit along X-axis
            // If we moved in +X, we hit the voxel's left face at x=hitX
            // If we moved in -X, we hit the voxel's right face at x=hitX+1
            float faceX = (stepX > 0) ? (float)hitX : (float)(hitX + 1);
            corners[0] = Float3(faceX, (float)hitY, (float)hitZ);
            corners[1] = Float3(faceX, (float)hitY + 1, (float)hitZ);
            corners[2] = Float3(faceX, (float)hitY + 1, (float)hitZ + 1);
            corners[3] = Float3(faceX, (float)hitY, (float)hitZ + 1);
        }
        else if (hitAxis == 1)
        {
            // Hit along Y-axis
            float faceY = (stepY > 0) ? (float)hitY : (float)(hitY + 1);
            corners[0] = Float3((float)hitX, faceY, (float)hitZ);
            corners[1] = Float3((float)hitX + 1, faceY, (float)hitZ);
            corners[2] = Float3((float)hitX + 1, faceY, (float)hitZ + 1);
            corners[3] = Float3((float)hitX, faceY, (float)hitZ + 1);
        }
        else if (hitAxis == 2)
        {
            // Hit along Z-axis
            float faceZ = (stepZ > 0) ? (float)hitZ : (float)(hitZ + 1);
            corners[0] = Float3((float)hitX, (float)hitY, faceZ);
            corners[1] = Float3((float)hitX + 1, (float)hitY, faceZ);
            corners[2] = Float3((float)hitX + 1, (float)hitY + 1, faceZ);
            corners[3] = Float3((float)hitX, (float)hitY + 1, faceZ);
        }

        // Now store these corners in edgeToHighlight
        Scene::Get().edgeToHighlight[0] = corners[0];
        Scene::Get().edgeToHighlight[1] = corners[1];
        Scene::Get().edgeToHighlight[2] = corners[2];
        Scene::Get().edgeToHighlight[3] = corners[3];
    }

    if (leftMouseButtonClicked)
    {
        leftMouseButtonClicked = false;

        int blockId = InputHandler::Get().currentSelectedBlockId;

        auto &scene = Scene::Get();

        if (blockId == 0) // delete a block
        {
            if (hitSurface)
            {
                auto &sceneGeometryAttributes = scene.m_geometryAttibutes;
                auto &sceneGeometryIndices = scene.m_geometryIndices;
                auto &sceneGeometryAttributeSize = scene.m_geometryAttibuteSize;
                auto &sceneGeometryIndicesSize = scene.m_geometryIndicesSize;

                unsigned int newVal = 0;
                int objectId = BlockIdToObjectId(deleteBlockId);

                if (IsUninstancedBlockType(deleteBlockId))
                {
                    updateSingleVoxel(
                        deletePos.x, deletePos.y, deletePos.z,
                        newVal,
                        voxelChunk,
                        sceneGeometryAttributes[objectId],
                        sceneGeometryIndices[objectId],
                        faceLocation[objectId],
                        sceneGeometryAttributeSize[objectId],
                        sceneGeometryIndicesSize[objectId],
                        currentFaceCount[objectId],
                        maxFaceCount[objectId],
                        freeFaces[objectId]);

                    scene.needSceneUpdate = true;
                    scene.sceneUpdateObjectId.push_back(objectId);
                    scene.sceneUpdateInstanceId.push_back(0);
                }
                else if (IsInstancedBlockType(deleteBlockId))
                {
                    unsigned int vid = GetLinearId(deletePos.x, deletePos.y, deletePos.z, voxelChunk.width);
                    Voxel oldVoxel = voxelChunk.data[vid];
                    voxelChunk.data[vid].id = newVal;

                    std::unordered_map<int, std::set<int>> &geometryInstanceIdMap = scene.geometryInstanceIdMap;

                    unsigned int instanceId = PositionToInstanceId(GetNumUninstancedBlockTypes(), objectId, deletePos.x, deletePos.y, deletePos.z, voxelChunk.width);
                    geometryInstanceIdMap[objectId].erase(instanceId);

                    scene.needSceneUpdate = true;
                    scene.sceneUpdateObjectId.push_back(objectId);
                    scene.sceneUpdateInstanceId.push_back(instanceId);

                    if (IsBaseLightBlockType(deleteBlockId))
                    {
                        int childBlockId = BlockTypeTestLight;
                        int childObjectIdx = BlockIdToObjectId(childBlockId);

                        unsigned int childInstanceId = PositionToInstanceId(GetNumUninstancedBlockTypes(), childObjectIdx, deletePos.x, deletePos.y, deletePos.z, voxelChunk.width);
                        geometryInstanceIdMap[childObjectIdx].erase(childInstanceId);

                        scene.sceneUpdateObjectId.push_back(childObjectIdx);
                        scene.sceneUpdateInstanceId.push_back(childInstanceId);
                    }
                }
            }
        }
        else // create a block
        {
            if (hasSpaceToCreate && hitSurface)
            {
                auto &scene = Scene::Get();

                auto &sceneGeometryAttributes = scene.m_geometryAttibutes;
                auto &sceneGeometryIndices = scene.m_geometryIndices;
                auto &sceneGeometryAttributeSize = scene.m_geometryAttibuteSize;
                auto &sceneGeometryIndicesSize = scene.m_geometryIndicesSize;

                unsigned int newVal = blockId;
                int objectId = BlockIdToObjectId(blockId);

                if (IsUninstancedBlockType(blockId))
                {
                    updateSingleVoxel(
                        createPos.x, createPos.y, createPos.z,
                        newVal,
                        voxelChunk,
                        sceneGeometryAttributes[objectId],
                        sceneGeometryIndices[objectId],
                        faceLocation[objectId],
                        sceneGeometryAttributeSize[objectId],
                        sceneGeometryIndicesSize[objectId],
                        currentFaceCount[objectId],
                        maxFaceCount[objectId],
                        freeFaces[objectId]);

                    scene.needSceneUpdate = true;
                    scene.sceneUpdateObjectId.push_back(objectId);
                    scene.sceneUpdateInstanceId.push_back(0);
                }
                else if (IsInstancedBlockType(blockId))
                {
                    unsigned int vid = GetLinearId(createPos.x, createPos.y, createPos.z, voxelChunk.width);
                    Voxel oldVoxel = voxelChunk.data[vid];
                    voxelChunk.data[vid].id = newVal;

                    std::unordered_map<int, std::set<int>> &geometryInstanceIdMap = scene.geometryInstanceIdMap;
                    std::unordered_map<int, std::array<float, 12>> &instanceTransformMatrices = scene.instanceTransformMatrices;

                    unsigned int instanceId = PositionToInstanceId(GetNumUninstancedBlockTypes(), objectId, createPos.x, createPos.y, createPos.z, voxelChunk.width);
                    geometryInstanceIdMap[objectId].insert(instanceId);

                    std::array<float, 12> transform = {1.0f, 0.0f, 0.0f, (float)createPos.x,
                                                       0.0f, 1.0f, 0.0f, (float)createPos.y,
                                                       0.0f, 0.0f, 1.0f, (float)createPos.z};
                    instanceTransformMatrices[instanceId] = transform;

                    scene.needSceneUpdate = true;
                    scene.sceneUpdateObjectId.push_back(objectId);
                    scene.sceneUpdateInstanceId.push_back(instanceId);

                    if (IsBaseLightBlockType(deleteBlockId))
                    {
                        int childBlockId = BlockTypeTestLight;
                        int childObjectIdx = childBlockId - 1;

                        unsigned int childInstanceId = PositionToInstanceId(GetNumUninstancedBlockTypes(), childObjectIdx, createPos.x, createPos.y, createPos.z, voxelChunk.width);
                        geometryInstanceIdMap[childObjectIdx].insert(childInstanceId);

                        instanceTransformMatrices[childInstanceId] = transform;

                        scene.sceneUpdateObjectId.push_back(childObjectIdx);
                        scene.sceneUpdateInstanceId.push_back(childInstanceId);
                    }
                }
            }
        }
    }
}
