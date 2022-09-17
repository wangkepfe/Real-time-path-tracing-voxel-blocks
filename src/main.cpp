

#include "shaders/LinearMath.h"

#include "core/UI.h"
#include "core/Backend.h"
#include "core/OptixRenderer.h"
#include "core/BufferManager.h"
#include "util/TextureUtils.h"

#include "voxelengine.h"

#include <IL/il.h>

int main(int argc, char* argv[])
{
    auto& backend = jazzfusion::Backend::Get();
    auto& renderer = jazzfusion::OptixRenderer::Get();

    try
    {
        auto& ui = jazzfusion::UI::Get();
        auto& bufferManager = jazzfusion::BufferManager::Get();
        auto& textureManager = jazzfusion::TextureManager::Get();
        auto& voxelengine = vox::VoxelEngine::Get();

        voxelengine.init();
        backend.init();
        bufferManager.init();
        textureManager.init();
        renderer.init();
        ui.init();

        backend.mainloop(vox::UpdateFunc);


    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    renderer.clear();
    backend.clear();

    return 0;
}
