#include "shaders/LinearMath.h"

#include "core/OfflineBackend.h"
#include "core/OptixRenderer.h"
#include "core/BufferManager.h"
#include "assets/TextureManager.h"
#include "assets/ModelManager.h"
#include "assets/BlockManager.h"
#include "core/GlobalSettings.h"
#include "core/RenderCamera.h"
#include "core/SceneConfig.h"
#include "core/Scene.h"
#include "util/ImageDiff.h"
#include "util/PerformanceTracker.h"

#include "voxelengine.h"
#include "voxelengine/VoxelSceneGen.h"
#include "voxelengine/VoxelMath.h"

#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <vector>
#include <algorithm>

int main(int argc, char *argv[])
{
    // Movement parameters for multi-frame animation (disabled)
    constexpr float tangentialMovementSpeed = 0.0f;

    // Parse command line arguments
    int width = 3840;
    int height = 2160;
    std::string outputPrefix = "offline_render";
    std::string sceneFile = "data/scene/scene_export.yaml";
    bool testCanonical = false;
    bool updateCanonical = false;
    std::string canonicalImagePath = "../../data/canonical/canonical_render.png";
    std::string runComment = "default run";

    // Frame configuration - can be overridden via command line
    int totalFrames = 64;
    std::vector<int> savedFrames = {1, 4, 16, 64}; // 1-indexed frame numbers to save

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

        else if (arg == "--output" && i + 1 < argc)
        {
            outputPrefix = argv[++i];
        }
        else if (arg == "--scene" && i + 1 < argc)
        {
            sceneFile = argv[++i];
        }
        else if (arg == "--test-canonical" || arg == "--test")
        {
            testCanonical = true;
        }
        else if (arg == "--update-canonical")
        {
            updateCanonical = true;
        }
        else if (arg == "--canonical-image" && i + 1 < argc)
        {
            canonicalImagePath = argv[++i];
        }
        else if (arg == "--comment" && i + 1 < argc)
        {
            runComment = argv[++i];
        }
        else if (arg == "--frames" && i + 1 < argc)
        {
            totalFrames = std::atoi(argv[++i]);
            if (totalFrames == 1)
            {
                savedFrames = {1}; // Only save frame 1 for single frame render
            }
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Offline Voxel Path Tracer\n";
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --width <int>        Output width (default: 1920)\n";
            std::cout << "  --height <int>       Output height (default: 1080)\n";
            std::cout << "  --output <string>    Output filename prefix (default: offline_render)\n";
            std::cout << "  --scene <file>       Scene configuration YAML file (optional)\n";
            std::cout << "  --test-canonical     Compare output with canonical image\n";
            std::cout << "  --update-canonical   Update the canonical reference image\n";
            std::cout << "  --canonical-image    Path to canonical image (default: ../../data/canonical/canonical_render.png)\n";
            std::cout << "  --comment <text>     Comment for performance report (default: default run)\n";
            std::cout << "  --frames <int>       Number of frames to render (default: 64, use 1 for single frame)\n";
            std::cout << "  --help, -h           Show this help message\n";
            return 0;
        }
    }

    std::cout << "=== Offline Voxel Path Tracer ===" << std::endl;
    std::cout << "Resolution: " << width << "x" << height << std::endl;
    std::cout << "Frames to render: " << totalFrames << " (saving frames: 1, 4, 16, 64)" << std::endl;
    std::cout << "Output prefix: " << outputPrefix << std::endl;

    // Load global settings from YAML file at startup
    auto &globalSettings = GlobalSettings::Get();
    globalSettings.LoadFromYAML("data/settings/global_settings.yaml");

    // Set offline mode
    GlobalSettings::SetOfflineMode(true);

    auto &offlineBackend = OfflineBackend::Get();
    auto &renderer = OptixRenderer::Get();

    try
    {
        auto &bufferManager = BufferManager::Get();

        auto &voxelengine = VoxelEngine::Get();

        // Initialize components in order
        // Initialize ModelManager first so VoxelEngine can use it
        std::cout << "Initializing ModelManager..." << std::endl;
        Assets::ModelManager::Get().initialize();

        std::cout << "Initializing BlockManager..." << std::endl;
        Assets::BlockManager::Get().initialize();

        std::cout << "Initializing voxel engine..." << std::endl;
        voxelengine.init();

        std::cout << "Initializing offline backend..." << std::endl;
        offlineBackend.init(width, height);

        std::cout << "Initializing buffer manager..." << std::endl;
        bufferManager.init();

        // Textures are now initialized through OptixRenderer's asset management system

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

        // Camera movement disabled - keep camera static
        Float3 staticCameraPosition = sceneConfig.camera.position;

        std::cout << "Camera setup - Position: (" << camera.pos.x << ", " << camera.pos.y << ", " << camera.pos.z << ")" << std::endl;
        std::cout << "Camera setup - Direction: (" << camera.dir.x << ", " << camera.dir.y << ", " << camera.dir.z << ")" << std::endl;
        std::cout << "Camera setup - FOV: " << sceneConfig.camera.fov << " degrees" << std::endl;
        std::cout << "Camera movement: DISABLED (static camera)" << std::endl;


        std::cout << "Starting rendering..." << std::endl;

        auto &perfTracker = PerformanceTracker::Get();

        // Render frames (selective saving - only save specific frames)
        for (int frame = 0; frame < totalFrames; frame++)
        {
            int frameNumber = frame + 1; // Convert to 1-indexed

            // Check if this frame should be saved
            bool shouldSave = std::find(savedFrames.begin(), savedFrames.end(), frameNumber) != savedFrames.end();

            // Generate comment for this frame
            std::string frameComment;
            if (shouldSave)
            {
                frameComment = "Saved frame " + std::to_string(frameNumber) + "/" + std::to_string(totalFrames);
            }
            else
            {
                frameComment = "Convergence frame " + std::to_string(frameNumber) + "/" + std::to_string(totalFrames);
            }

            // Begin performance tracking for this frame
            perfTracker.beginFrame(frameNumber, width, height, frameComment);

            // Update unified time management for this frame
            GlobalSettings::UpdateTime();

            if (shouldSave)
            {
                // Generate output filename for saved frames
                std::stringstream filename;
                filename << outputPrefix << "_" << std::setfill('0') << std::setw(4) << frame << ".png";

                // Render and save frame
                offlineBackend.renderFrame(filename.str());
            }
            else
            {
                // Render frame but don't save (for convergence)
                offlineBackend.renderFrame("");
            }

            // End performance tracking and print stats
            perfTracker.endFrame();

            // Add test blocks after first frame is rendered
            if (frameNumber >= 2 && frameNumber <= 6)
            {
                std::cout << "Simulating mouse click #" << (frameNumber - 1) << " to place plank block at camera center..." << std::endl;
                
                // Simulate multiple mouse clicks to place light blocks along the ray
                voxelengine.leftMouseButtonClicked = true;
                
                std::cout << "Mouse click simulated - VoxelEngine will handle placement on next update." << std::endl;
            }

            // Print performance stats for saved frames or every 16th frame
            if (shouldSave || (frameNumber % 16 == 0))
            {
                std::cout << "Frame " << frameNumber << "/" << totalFrames << " completed";
                if (shouldSave)
                    std::cout << " (SAVED)";
                std::cout << std::endl;
            }

            // Camera stays static (no movement)
            // camera.pos remains unchanged
        }

        std::cout << "\n=== Rendering Complete - Writing all frames to disk ===" << std::endl;

        // Write all batched frames at once
        offlineBackend.writeAllBatchedFrames();

        std::cout << "Output files saved with prefix: " << outputPrefix << std::endl;

        // Save performance report
        std::cout << "\n=== Performance Report ===" << std::endl;
        perfTracker.saveReport("../../data/perf/performance_report.txt", runComment);
        std::cout << "Performance data saved to: data/perf/performance_report.txt" << std::endl;

        // Handle canonical image testing/updating
        if (testCanonical || updateCanonical)
        {
            // Use the 64th frame (frame index 63) for canonical testing
            std::stringstream canonicalFramePath;
            canonicalFramePath << outputPrefix << "_" << std::setfill('0') << std::setw(4) << (totalFrames - 1) << ".png";
            std::string testImagePath = canonicalFramePath.str();

            if (updateCanonical)
            {
                std::cout << "\n=== Updating Canonical Image ===" << std::endl;
                try
                {
                    if (std::filesystem::copy_file(testImagePath, canonicalImagePath, std::filesystem::copy_options::overwrite_existing))
                    {
                        std::cout << "Canonical image updated: " << canonicalImagePath << std::endl;
                    }
                    else
                    {
                        std::cerr << "Failed to update canonical image" << std::endl;
                    }
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error updating canonical image: " << e.what() << std::endl;
                }
            }

            if (testCanonical)
            {
                std::cout << "\n=== Canonical Image Testing ===" << std::endl;

                if (!std::filesystem::exists(canonicalImagePath))
                {
                    std::cout << "Warning: Canonical image not found at " << canonicalImagePath << std::endl;
                    std::cout << "Use --update-canonical to create it from current render" << std::endl;
                }
                else if (!std::filesystem::exists(testImagePath))
                {
                    std::cerr << "Error: Test image not found at " << testImagePath << std::endl;
                }
                else
                {
                    std::cout << "Comparing: " << testImagePath << " vs " << canonicalImagePath << std::endl;

                    // Perform image comparison
                    ImageDiffResult result = ImageDiff::compare(testImagePath, canonicalImagePath);
                    result.print();

                    // Generate diff image
                    std::string diffImagePath = outputPrefix + "_diff.png";
                    if (ImageDiff::generateDiffImage(testImagePath, canonicalImagePath, diffImagePath))
                    {
                        std::cout << "Difference visualization saved to: " << diffImagePath << std::endl;
                    }
                    else
                    {
                        std::cerr << "Failed to generate difference image" << std::endl;
                    }

                    // Return appropriate exit code based on results
                    if (!result.isIdentical && !result.isVeryClose)
                    {
                        std::cout << "\nWarning: Significant differences detected from canonical image!" << std::endl;
                        std::cout << "This may indicate a regression or intentional change." << std::endl;
                        if (!result.isClose)
                        {
                            std::cout << "Consider investigating the differences." << std::endl;
                        }
                    }
                    else
                    {
                        std::cout << "\nImage matches canonical reference within acceptable tolerance." << std::endl;
                    }
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Cleanup
    renderer.clear();
    offlineBackend.clearBatchedFrames();
    offlineBackend.clear();

    return 0;
}