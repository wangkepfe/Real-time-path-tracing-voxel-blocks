

#include "shaders/ShaderCommon.h"

#include "core/UI.h"
#include "core/Backend.h"
#include "core/OptixRenderer.h"

#include <IL/il.h>

int main(int argc, char* argv[])
{
    auto& backend = jazzfusion::Backend::Get();
    auto& renderer = jazzfusion::OptixRenderer::Get();
    auto& ui = jazzfusion::UI::Get();

    try
    {
        backend.init();
        renderer.init();
        ui.init();

        backend.mainloop();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    ui.clear();
    renderer.clear();
    backend.clear();

    return 0;
}
