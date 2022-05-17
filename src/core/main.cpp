

#include "shaders/app_config.h"

#include "core/Options.h"
#include "core/Application.h"
#include "core/Backend.h"
#include "core/OptixRenderer.h"

#include <IL/il.h>

int main(int argc, char *argv[])
{
    auto& backend = jazzfusion::Backend::Get();
    auto& optix = jazzfusion::OptixRenderer::Get();
    auto& app = Application::Get();

    try
    {
        backend.init();
        optix.init();
        app.init(backend.getWindow());

        backend.mainloop();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    optix.clear();
    backend.clear();
    return 0;
}
