

#include "shaders/LinearMath.h"

#include "core/UI.h"
#include "core/Backend.h"
#include "core/OptixRenderer.h"
#include "core/BufferManager.h"
#include "util/TextureUtils.h"

#include "voxelengine.h"

int main(int argc, char *argv[])
{
    auto &backend = Backend::Get();
    auto &renderer = OptixRenderer::Get();

    try
    {
        auto &ui = UI::Get();
        auto &bufferManager = BufferManager::Get();
        auto &textureManager = TextureManager::Get();
        auto &voxelengine = VoxelEngine::Get();

        voxelengine.init();
        backend.init();
        bufferManager.init();
        textureManager.init();
        renderer.init();
        ui.init();

        backend.mainloop();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }

    renderer.clear();
    backend.clear();

    return 0;
}
