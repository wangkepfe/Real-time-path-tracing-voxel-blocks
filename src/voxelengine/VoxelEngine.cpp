#include "VoxelEngine.h"
#include "BlockMesher.h"

#include "core/Scene.h"
#include "core/InputHandler.h"

namespace vox
{

static void MouseButtonCallback(int button, int action, int mods)
{
    std::cout << button << "\n";
}

VoxelEngine::~VoxelEngine()
{

}

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
    auto& sceneGeometryAttributes = scene.getGeometryAttibutes();
    auto& sceneGeometryIndices = scene.getGeometryIndices();

    sceneGeometryAttributes.resize(1);
    sceneGeometryIndices.resize(1);

    // Meshing
    BlockMesher blockMesher(voxelchunk, sceneGeometryAttributes[0], sceneGeometryIndices[0]);
    blockMesher.process();


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

}