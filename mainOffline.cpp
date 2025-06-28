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
    // Movement parameters for multi-frame animation
    constexpr float tangentialMovementSpeed = 0.0f;

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

        // Convert direction vector to yaw/pitch like the GUI does
        Float3 dir = normalize(sceneConfig.camera.direction);
        Float2 yawPitch = DirToYawPitch(dir);
        camera.yaw = yawPitch.x;
        camera.pitch = yawPitch.y;

        // Apply FOV from scene configuration
        float fovX = sceneConfig.camera.fov * Pi_over_180;
        float fovY = fovX * (camera.resolution.y / camera.resolution.x);
        camera.tanHalfFov = Float2(tanf(fovX * 0.5f), tanf(fovY * 0.5f));

        // Update camera to recalculate matrices with correct yaw/pitch and FOV
        camera.update();

        // Store the target final position (from scene config) for multi-frame animation
        Float3 targetFinalPosition = sceneConfig.camera.position;

        // Calculate camera's right vector for tangential movement
        Float3 up = Float3(0.0f, 1.0f, 0.0f);  // Assuming Y is up
        Float3 cameraDir = normalize(sceneConfig.camera.direction);
        Float3 rightVector = normalize(cross(cameraDir, up));

        // For multi-frame rendering, adjust initial camera position to start left of target
        if (numFrames > 1)
        {
            // Start position: move left from target by total movement distance
            float totalMovement = (numFrames - 1) * tangentialMovementSpeed;
            camera.pos = targetFinalPosition - rightVector * totalMovement;
            camera.update();
        }

        std::cout << "Camera setup - Position: (" << camera.pos.x << ", " << camera.pos.y << ", " << camera.pos.z << ")" << std::endl;
        std::cout << "Camera setup - Direction: (" << camera.dir.x << ", " << camera.dir.y << ", " << camera.dir.z << ")" << std::endl;
        std::cout << "Camera setup - FOV: " << sceneConfig.camera.fov << " degrees" << std::endl;

        if (numFrames > 1)
        {
            std::cout << "Multi-frame animation: tangential movement with speed " << tangentialMovementSpeed << std::endl;
            std::cout << "Target final position: (" << targetFinalPosition.x << ", " << targetFinalPosition.y << ", " << targetFinalPosition.z << ")" << std::endl;
        }

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

            // Move camera tangentially for multi-frame animation
            // Move camera tangentially to the right, approaching target final position
            camera.pos = targetFinalPosition - rightVector * tangentialMovementSpeed * (numFrames - 1 - frame);
            camera.update();
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