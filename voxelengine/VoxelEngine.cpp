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

        sceneGeometryAttributes.resize(1);
        sceneGeometryIndices.resize(1);
        sceneGeometryAttributeSize.resize(1);
        sceneGeometryIndicesSize.resize(1);

        sceneGeometryAttributes[0] = nullptr;
        sceneGeometryIndices[0] = nullptr;
        sceneGeometryAttributeSize[0] = 0;
        sceneGeometryIndicesSize[0] = 0;

        initVoxelChunk(voxelChunk);
        generateMesh(&(sceneGeometryAttributes[0]), &(sceneGeometryIndices[0]), sceneGeometryAttributeSize[0], sceneGeometryIndicesSize[0], voxelChunk);
    }

    void VoxelEngine::update()
    {
        using namespace jazzfusion;

        if (leftMouseButtonClicked)
        {
            leftMouseButtonClicked = false;

            auto &camera = RenderCamera::Get().camera;

            Ray ray{camera.pos, camera.dir};

            bool hasSpaceToCreate = false;
            bool hitSurface = false;
            Int3 createPos;

            RayVoxelGridTraversal(ray, [&](int x, int y, int z) -> bool
                                  {
                auto& voxelEngine = VoxelEngine::Get();
                VoxelChunk& voxelChunk = voxelEngine.voxelChunk;

                if (x < 0 || x >= voxelChunk.width || y < 0 || y >= voxelChunk.width || z < 0 || z >= voxelChunk.width) {
                    return false;
                }

                Voxel voxel = voxelChunk.get(x, y, z);

                if (voxel.id == 0)
                {
                    hasSpaceToCreate = true;
                    createPos = Int3(x, y, z);

                    return true;
                }
                else
                {
                    hitSurface = true;
                    return false;
                } });

            if (hasSpaceToCreate && hitSurface)
            {
                auto &scene = jazzfusion::Scene::Get();

                voxelChunk.set(createPos.x, createPos.y, createPos.z, 1);

                auto &sceneGeometryAttributes = scene.m_geometryAttibutes;
                auto &sceneGeometryIndices = scene.m_geometryIndices;
                auto &sceneGeometryAttributeSize = scene.m_geometryAttibuteSize;
                auto &sceneGeometryIndicesSize = scene.m_geometryIndicesSize;
                generateMesh(&(sceneGeometryAttributes[0]), &(sceneGeometryIndices[0]), sceneGeometryAttributeSize[0], sceneGeometryIndicesSize[0], voxelChunk);

                scene.m_updateCallback(0);
            }
        }
    }

}