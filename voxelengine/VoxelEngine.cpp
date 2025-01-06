#include "VoxelEngine.h"
#include "VoxelMath.h"

#include "core/Scene.h"
#include "core/InputHandler.h"
#include "core/RenderCamera.h"

namespace vox
{

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

    void VoxelEngine::init()
    {
        using namespace jazzfusion;

        // Input handling
        auto &inputHandler = jazzfusion::InputHandler::Get();
        inputHandler.setMouseButtonCallbackFunc(MouseButtonCallback);

        auto &scene = jazzfusion::Scene::Get();
        auto &sceneGeometryAttributes = scene.m_geometryAttibutes;
        auto &sceneGeometryIndices = scene.m_geometryIndices;
        auto &sceneGeometryAttributeSize = scene.m_geometryAttibuteSize;
        auto &sceneGeometryIndicesSize = scene.m_geometryIndicesSize;

        totalNumBlockTypes = 6;
        // totalNumGeometries = totalNumBlockTypes;
        totalNumGeometries = totalNumBlockTypes + 1;

        sceneGeometryAttributes.resize(totalNumGeometries);
        sceneGeometryIndices.resize(totalNumGeometries);
        sceneGeometryAttributeSize.resize(totalNumGeometries);
        sceneGeometryIndicesSize.resize(totalNumGeometries);

        faceLocation.resize(totalNumBlockTypes);
        currentFaceCount.resize(totalNumBlockTypes);
        maxFaceCount.resize(totalNumBlockTypes);
        freeFaces.resize(totalNumBlockTypes);

        Voxel *d_data;
        initVoxels(voxelChunk, &d_data);

        for (int i = 0; i < totalNumBlockTypes; ++i)
        {
            sceneGeometryAttributeSize[i] = 0;
            sceneGeometryIndicesSize[i] = 0;

            currentFaceCount[i] = 0;
            maxFaceCount[i] = 0;

            generateMesh(
                &(sceneGeometryAttributes[i]),
                &(sceneGeometryIndices[i]),
                faceLocation[i],
                sceneGeometryAttributeSize[i],
                sceneGeometryIndicesSize[i],
                currentFaceCount[i],
                maxFaceCount[i],
                voxelChunk,
                d_data,
                i + 1);
        }

        freeDeviceVoxelData(d_data);

        // Generate geometry for sea
        int seaIndex = totalNumGeometries - 1;
        generateSea(
            &(sceneGeometryAttributes[seaIndex]),
            &(sceneGeometryIndices[seaIndex]),
            sceneGeometryAttributeSize[seaIndex],
            sceneGeometryIndicesSize[seaIndex],
            voxelChunk.width);
    }

    void VoxelEngine::reload()
    {
        Voxel *d_data;
        size_t totalVoxels = voxelChunk.width * voxelChunk.width * voxelChunk.width;
        cudaMalloc(&d_data, totalVoxels * sizeof(Voxel));
        cudaMemcpy(d_data, voxelChunk.data, totalVoxels * sizeof(Voxel), cudaMemcpyHostToDevice);

        auto &scene = jazzfusion::Scene::Get();
        auto &sceneGeometryAttributes = scene.m_geometryAttibutes;
        auto &sceneGeometryIndices = scene.m_geometryIndices;
        auto &sceneGeometryAttributeSize = scene.m_geometryAttibuteSize;
        auto &sceneGeometryIndicesSize = scene.m_geometryIndicesSize;

        for (int i = 0; i < totalNumBlockTypes; ++i)
        {
            sceneGeometryAttributeSize[i] = 0;
            sceneGeometryIndicesSize[i] = 0;

            currentFaceCount[i] = 0;
            maxFaceCount[i] = 0;

            generateMesh(
                &(sceneGeometryAttributes[i]),
                &(sceneGeometryIndices[i]),
                faceLocation[i],
                sceneGeometryAttributeSize[i],
                sceneGeometryIndicesSize[i],
                currentFaceCount[i],
                maxFaceCount[i],
                voxelChunk,
                d_data,
                i + 1);

            jazzfusion::Scene::Get().sceneUpdateObjectId.push_back(i);
        }
        jazzfusion::Scene::Get().needSceneUpdate = true;

        freeDeviceVoxelData(d_data);
    }

    void VoxelEngine::update()
    {
        using namespace jazzfusion;

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
            jazzfusion::Scene::Get().edgeToHighlight[0] = corners[0];
            jazzfusion::Scene::Get().edgeToHighlight[1] = corners[1];
            jazzfusion::Scene::Get().edgeToHighlight[2] = corners[2];
            jazzfusion::Scene::Get().edgeToHighlight[3] = corners[3];
        }

        if (leftMouseButtonClicked)
        {
            leftMouseButtonClicked = false;

            int blockId = InputHandler::Get().currentSelectedBlockId;

            if (blockId == 0) // delete a block
            {
                if (hitSurface)
                {
                    auto &scene = jazzfusion::Scene::Get();

                    auto &sceneGeometryAttributes = scene.m_geometryAttibutes;
                    auto &sceneGeometryIndices = scene.m_geometryIndices;
                    auto &sceneGeometryAttributeSize = scene.m_geometryAttibuteSize;
                    auto &sceneGeometryIndicesSize = scene.m_geometryIndicesSize;

                    unsigned int newVal = 0;
                    int idx = deleteBlockId - 1;

                    updateSingleVoxel(
                        deletePos.x, deletePos.y, deletePos.z,
                        newVal,
                        voxelChunk,
                        sceneGeometryAttributes[idx],
                        sceneGeometryIndices[idx],
                        faceLocation[idx],
                        sceneGeometryAttributeSize[idx],
                        sceneGeometryIndicesSize[idx],
                        currentFaceCount[idx],
                        maxFaceCount[idx],
                        freeFaces[idx]);

                    jazzfusion::Scene::Get().needSceneUpdate = true;
                    jazzfusion::Scene::Get().sceneUpdateObjectId.push_back(idx);
                }
            }
            else // create a block
            {
                if (hasSpaceToCreate && hitSurface)
                {
                    auto &scene = jazzfusion::Scene::Get();

                    auto &sceneGeometryAttributes = scene.m_geometryAttibutes;
                    auto &sceneGeometryIndices = scene.m_geometryIndices;
                    auto &sceneGeometryAttributeSize = scene.m_geometryAttibuteSize;
                    auto &sceneGeometryIndicesSize = scene.m_geometryIndicesSize;

                    unsigned int newVal = blockId;
                    int idx = blockId - 1;

                    updateSingleVoxel(
                        createPos.x, createPos.y, createPos.z,
                        newVal,
                        voxelChunk,
                        sceneGeometryAttributes[idx],
                        sceneGeometryIndices[idx],
                        faceLocation[idx],
                        sceneGeometryAttributeSize[idx],
                        sceneGeometryIndicesSize[idx],
                        currentFaceCount[idx],
                        maxFaceCount[idx],
                        freeFaces[idx]);

                    jazzfusion::Scene::Get().needSceneUpdate = true;
                    jazzfusion::Scene::Get().sceneUpdateObjectId.push_back(idx);
                }
            }
        }
    }
}