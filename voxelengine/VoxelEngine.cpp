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
{}

void VoxelEngine::init()
{
    using namespace jazzfusion;

    // Input handling
    auto& inputHandler = jazzfusion::InputHandler::Get();
    inputHandler.setMouseButtonCallbackFunc(MouseButtonCallback);

    // Generate scene voxel data
    VoxelChunk& voxelchunk = data[0];
    voxelchunk.clear();
    voxelchunk.set(Voxel(1), 0, 0, 0);
    voxelchunk.set(Voxel(1), 1, 0, 0);
    voxelchunk.set(Voxel(1), 0, 1, 0);

    auto& scene = jazzfusion::Scene::Get();
    auto& sceneGeometryAttributes = scene.m_geometryAttibutes;
    auto& sceneGeometryIndices = scene.m_geometryIndices;

    sceneGeometryAttributes.resize(1);
    sceneGeometryIndices.resize(1);

    // Meshing
    blockMeshers.emplace_back(voxelchunk, sceneGeometryAttributes[0], sceneGeometryIndices[0]);
    blockMeshers[0].process();

    // Square geometry test
    //
    // sceneGeometryAttributes.resize(1);
    // sceneGeometryIndices.resize(1);

    // auto& attr = sceneGeometryAttributes[0];
    // auto& indi = sceneGeometryIndices[0];

    // attr.resize(4);
    // attr[0].vertex = Float3(0, 0, 0);
    // attr[1].vertex = Float3(1, 0, 0);
    // attr[2].vertex = Float3(1, 0, 1);
    // attr[3].vertex = Float3(0, 0, 1);

    // indi.resize(6);
    // indi[0] = 0;
    // indi[1] = 1;
    // indi[2] = 2;
    // indi[3] = 0;
    // indi[4] = 2;
    // indi[5] = 3;
}

void VoxelEngine::update()
{
    using namespace jazzfusion;

    if (leftMouseButtonClicked)
    {
        leftMouseButtonClicked = false;

        auto& camera = RenderCamera::Get().camera;

        Ray ray{ camera.pos, camera.dir };

        bool hasSpaceToCreate = false;
        bool hitSurface = false;
        Int3 createPos;

        RayVoxelGridTraversal(ray, [&](int x, int y, int z)->bool
            {
                auto& voxelEngine = VoxelEngine::Get();
                VoxelChunk& voxelchunk = voxelEngine.data[0];
                auto voxel = voxelchunk.get(x, y, z);
                if (voxel == std::nullopt)
                {
                    // std::cout << "out of bound traversal: " << x << " " << y << " " << z << std::endl;
                    return true;
                }

                if (voxel->id == 0)
                {
                    hasSpaceToCreate = true;
                    createPos = Int3(x, y, z);

                    // std::cout << "in bound traversal: " << x << " " << y << " " << z << std::endl;
                    return true;
                }
                else
                {
                    hitSurface = true;

                    // std::cout << "hit: " << x << " " << y << " " << z << std::endl;
                    return false;
                }
            });

        if (hasSpaceToCreate)
        {
            blockMeshers[0].update(Voxel(1), createPos.x, createPos.y, createPos.z);

            auto& scene = jazzfusion::Scene::Get();
            scene.m_updateCallback(0);
        }
    }
}

}