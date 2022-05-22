

#include "shaders/ShaderCommon.h"

#include "core/UI.h"
#include "core/Backend.h"
#include "core/OptixRenderer.h"

#include <IL/il.h>

int main(int argc, char* argv[])
{
    auto& backend = jazzfusion::Backend::Get();
    auto& optix = jazzfusion::OptixRenderer::Get();
    auto& ui = jazzfusion::UI::Get();

    try
    {
        backend.init();
        optix.init();
        ui.init();

        backend.mainloop();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    ui.clear();
    optix.clear();
    backend.clear();

    return 0;
}
