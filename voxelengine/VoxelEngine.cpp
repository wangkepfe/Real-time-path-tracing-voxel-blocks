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

        sceneGeometryAttributes.resize(1);
        sceneGeometryIndices.resize(1);

        generateMesh(sceneGeometryAttributes[0], sceneGeometryIndices[0], voxelChunk);
    }

    void VoxelEngine::update()
    {
        using namespace jazzfusion;

        // if (leftMouseButtonClicked)
        // {
        //     leftMouseButtonClicked = false;

        //     auto &camera = RenderCamera::Get().camera;

        //     Ray ray{camera.pos, camera.dir};

        //     bool hasSpaceToCreate = false;
        //     bool hitSurface = false;
        //     Int3 createPos;

        //     RayVoxelGridTraversal(ray, [&](int x, int y, int z) -> bool
        //                           {
        //         auto& voxelEngine = VoxelEngine::Get();
        //         voxelChunk& voxelChunk = voxelEngine.voxelChunk;
        //         auto voxel = voxelChunk.get(x, y, z);
        //         if (voxel == std::nullopt)
        //         {
        //             // std::cout << "out of bound traversal: " << x << " " << y << " " << z << std::endl;
        //             return true;
        //         }

        //         if (voxel->id == 0)
        //         {
        //             hasSpaceToCreate = true;
        //             createPos = Int3(x, y, z);

        //             // std::cout << "in bound traversal: " << x << " " << y << " " << z << std::endl;
        //             return true;
        //         }
        //         else
        //         {
        //             hitSurface = true;

        //             // std::cout << "hit: " << x << " " << y << " " << z << std::endl;
        //             return false;
        //         } });

        //     if (hasSpaceToCreate)
        //     {
        //         blockMeshers[0].update(Voxel(1), createPos.x, createPos.y, createPos.z);

        //         auto &scene = jazzfusion::Scene::Get();
        //         scene.m_updateCallback(0);
        //     }
        // }
    }

}