

#include "shaders/app_config.h"

#include "core/Options.h"
#include "core/Application.h"
#include "core/Backend.h"

#include <IL/il.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

static void error_callback(int error, const char* description)
{
    std::cerr << "Error: "<< error << ": " << description << '\n';
}

int main(int argc, char *argv[])
{
    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
    {
        std::cerr << "Error: GLFW failed to initialize.\n";
        return -1;
    }

    auto& backend = jazzfusion::Backend::Get();
    backend.run();

    glfwTerminate();

    return 0;
}
