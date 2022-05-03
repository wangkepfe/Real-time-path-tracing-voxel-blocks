

#include "shaders/app_config.h"

#include "core/Options.h"
#include "core/Application.h"
#include "core/Backend.h"

#include <IL/il.h>

int main(int argc, char *argv[])
{
    auto& backend = jazzfusion::Backend::Get();
    auto& app = Application::Get();

    try
    {
        backend.init();

        app.init(backend.getWindow());

        if (!app.isValid())
        {
            throw std::runtime_error("Application failed to initialize successfully.");
        }

        backend.mainloop();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}
