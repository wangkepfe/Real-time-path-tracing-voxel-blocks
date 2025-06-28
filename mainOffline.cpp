#include "shaders/LinearMath.h"

#include "core/OfflineBackend.h"
#include "core/OptixRenderer.h"
#include "core/BufferManager.h"
#include "util/TextureUtils.h"
#include "core/GlobalSettings.h"
#include "core/RenderCamera.h"
#include "core/SceneConfig.h"

#include "voxelengine.h"

#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <filesystem>

int main(int argc, char *argv[])
{
    // Parse command line arguments
    int width = 1920;
    int height = 1080;
    int numFrames = 1;
    std::string outputPrefix = "offline_render";
    std::string sceneFile = "";

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc)
        {
            width = std::atoi(argv[++i]);
        }
        else if (arg == "--height" && i + 1 < argc)
        {
            height = std::atoi(argv[++i]);
        }
        else if (arg == "--frames" && i + 1 < argc)
        {
            numFrames = std::atoi(argv[++i]);
        }
        else if (arg == "--output" && i + 1 < argc)
        {
            outputPrefix = argv[++i];
        }
        else if (arg == "--scene" && i + 1 < argc)
        {
            sceneFile = argv[++i];
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Offline Voxel Path Tracer\n";
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --width <int>     Output width (default: 1920)\n";
            std::cout << "  --height <int>    Output height (default: 1080)\n";
            std::cout << "  --frames <int>    Number of frames to render (default: 1)\n";
            std::cout << "  --output <string> Output filename prefix (default: offline_render)\n";
            std::cout << "  --scene <file>    Scene configuration YAML file (optional)\n";
            std::cout << "  --help, -h        Show this help message\n";
            return 0;
        }
    }

    std::cout << "=== Offline Voxel Path Tracer ===" << std::endl;
    std::cout << "Resolution: " << width << "x" << height << std::endl;
    std::cout << "Frames to render: " << numFrames << std::endl;
    std::cout << "Output prefix: " << outputPrefix << std::endl;

    // Set offline mode
    GlobalSettings::SetOfflineMode(true);

    auto &offlineBackend = OfflineBackend::Get();
    auto &renderer = OptixRenderer::Get();

    try
    {
        auto &bufferManager = BufferManager::Get();
        auto &textureManager = TextureManager::Get();
        auto &voxelengine = VoxelEngine::Get();

        // Initialize components in order
        std::cout << "Initializing voxel engine..." << std::endl;
        voxelengine.init();

        std::cout << "Initializing offline backend..." << std::endl;
        offlineBackend.init(width, height);

        std::cout << "Initializing buffer manager..." << std::endl;
        bufferManager.init();

        std::cout << "Initializing texture manager..." << std::endl;
        textureManager.init();

        std::cout << "Initializing OptixRenderer..." << std::endl;
        renderer.init();

        // Load scene configuration
        SceneConfig sceneConfig;
        if (!sceneFile.empty() && std::filesystem::exists(sceneFile))
        {
            if (!SceneConfigParser::LoadFromFile(sceneFile, sceneConfig))
            {
                std::cerr << "Warning: Failed to load scene config, using defaults" << std::endl;
                sceneConfig = SceneConfigParser::CreateDefault();
            }
        }
        else
        {
            if (!sceneFile.empty())
            {
                std::cout << "Scene file not found: " << sceneFile << ", using defaults" << std::endl;
            }
            sceneConfig = SceneConfigParser::CreateDefault();

            // Create a default scene file for reference
            if (sceneFile.empty())
            {
                SceneConfigParser::SaveToFile("scene_default.yaml", sceneConfig);
                std::cout << "Created default scene config: scene_default.yaml" << std::endl;
            }
        }

        // Initialize camera to a nice position
        auto &camera = RenderCamera::Get().camera;
        camera.init(width, height);

        // Set camera from scene configuration
        camera.pos = sceneConfig.camera.position;

        // Convert direction vector to yaw/pitch if needed
        Float3 dir = normalize(sceneConfig.camera.direction);
        camera.dir = dir;  // Set direction directly for now

        // Note: Camera class calculates its own up vector internally
        camera.update();

        std::cout << "Starting rendering..." << std::endl;

        // Render frames
        for (int frame = 0; frame < numFrames; frame++)
        {
            std::cout << "\n=== Rendering Frame " << (frame + 1) << "/" << numFrames << " ===" << std::endl;

            // Generate output filename
            std::stringstream filename;
            filename << outputPrefix;
            if (numFrames > 1)
            {
                filename << "_" << std::setfill('0') << std::setw(4) << frame;
            }
            filename << ".png";

            // Render and save frame
            offlineBackend.renderFrame(filename.str());

            // Optional: Move camera slightly for animation
            if (numFrames > 1)
            {
                float angle = (frame + 1) * (2.0f * 3.14159f / numFrames);
                float radius = 25.0f;
                camera.pos = Float3(
                    radius * cos(angle),
                    15.0f,
                    radius * sin(angle)
                );
                camera.dir = normalize(Float3(-camera.pos.x, -camera.pos.y * 0.3f, -camera.pos.z));
                camera.update();

                // Reset accumulation for new view
                offlineBackend.resetAccumulationCounter();
            }
        }

        std::cout << "\n=== Rendering Complete ===" << std::endl;
        std::cout << "Output files saved with prefix: " << outputPrefix << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Cleanup
    renderer.clear();
    offlineBackend.clear();

    return 0;
}