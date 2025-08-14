

#include "shaders/LinearMath.h"

#include "core/UI.h"
#include "core/Backend.h"
#include "core/OptixRenderer.h"
#include "core/BufferManager.h"
#include "core/GlobalSettings.h"
#include "assets/TextureManager.h"
#include "assets/ModelManager.h"
#include "assets/BlockManager.h"

#include "voxelengine.h"

int main(int argc, char *argv[])
{
    auto &backend = Backend::Get();
    auto &renderer = OptixRenderer::Get();

    try
    {
        auto &globalSettings = GlobalSettings::Get();
        auto &ui = UI::Get();
        auto &bufferManager = BufferManager::Get();

        auto &voxelengine = VoxelEngine::Get();

        // Load global settings from YAML file at startup
        globalSettings.LoadFromYAML("data/settings/global_settings.yaml");

        // Initialize ModelManager first so VoxelEngine can use it
        Assets::ModelManager::Get().initialize();
        Assets::BlockManager::Get().initialize();

        voxelengine.init();
        backend.init();
        bufferManager.init();
        // Textures are now initialized through OptixRenderer's asset management system
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
