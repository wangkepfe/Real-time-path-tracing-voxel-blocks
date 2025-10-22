#include "core/WorldSceneManager.h"

#include "core/SceneConfig.h"
#include "core/InputHandler.h"
#include "core/Character.h"
#include "core/Scene.h"
#include "core/RenderCamera.h"
#include "voxelengine/VoxelEngine.h"
#include "shaders/Camera.h"
#include "shaders/LinearMath.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <cstdint>

namespace
{
    std::filesystem::path GetChunkStorageDirectory()
    {
        std::filesystem::path chunkDir;
#ifdef _WIN32
        const char *appData = std::getenv("APPDATA");
        if (appData)
        {
            chunkDir = std::filesystem::path(appData) / "wotw" / "chunks";
        }
        else
        {
            chunkDir = "chunks";
        }
#else
        const char *home = std::getenv("HOME");
        if (home)
        {
            chunkDir = std::filesystem::path(home) / ".local" / "share" / "wotw" / "chunks";
        }
        else
        {
            chunkDir = "chunks";
        }
#endif
        return chunkDir;
    }

    std::string ComputeChunkHash(const VoxelChunk &chunk)
    {
        constexpr unsigned long long kFnvOffsetBasis = 1469598103934665603ull;
        constexpr unsigned long long kFnvPrime = 1099511628211ull;

        auto byteCount = static_cast<size_t>(chunk.size());
        const auto *data = reinterpret_cast<const uint8_t *>(chunk.data);

        unsigned long long hash = kFnvOffsetBasis;
        for (size_t i = 0; i < byteCount; ++i)
        {
            hash ^= static_cast<unsigned long long>(data[i]);
            hash *= kFnvPrime;
        }

        std::ostringstream oss;
        oss << std::hex << std::setw(16) << std::setfill('0') << hash;
        return oss.str();
    }

    bool WriteChunkToFile(const VoxelChunk &chunk, const std::filesystem::path &filepath)
    {
        std::ofstream out(filepath, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!out.is_open())
        {
            std::cout << "Failed to open chunk file for writing: " << filepath << std::endl;
            return false;
        }

        auto byteCount = static_cast<std::streamsize>(chunk.size());
        out.write(reinterpret_cast<const char *>(chunk.data), byteCount);
        if (!out.good())
        {
            std::cout << "Failed to write chunk data to file: " << filepath << std::endl;
            return false;
        }

        return true;
    }

    bool ReadChunkFromFile(VoxelChunk &chunk, const std::filesystem::path &filepath)
    {
        std::error_code ec;
        auto expectedSize = static_cast<uintmax_t>(chunk.size());
        auto fileSize = std::filesystem::file_size(filepath, ec);
        if (ec || fileSize != expectedSize)
        {
            std::cout << "Chunk file size mismatch or read error: " << filepath << std::endl;
            return false;
        }

        std::ifstream in(filepath, std::ios::binary | std::ios::in);
        if (!in.is_open())
        {
            std::cout << "Failed to open chunk file for reading: " << filepath << std::endl;
            return false;
        }

        auto byteCount = static_cast<std::streamsize>(chunk.size());
        in.read(reinterpret_cast<char *>(chunk.data), byteCount);
        if (in.gcount() != byteCount)
        {
            std::cout << "Failed to read chunk data from file: " << filepath << std::endl;
            return false;
        }

        return true;
    }
} // namespace

bool WorldSceneManager::SaveScene(const std::string &filepath, const Camera &camera, InputHandler &inputHandler)
{
    SceneConfig currentScene;
    currentScene.camera.position = camera.pos;
    currentScene.camera.direction = camera.dir;
    currentScene.camera.up = Float3(0.0f, 1.0f, 0.0f); // Standard up vector
    currentScene.camera.fov = 90.0f;                   // Default FOV, should match camera default

    if (auto *character = inputHandler.getCharacter())
    {
        const EntityTransform &characterTransform = character->getTransform();
        currentScene.character.position = characterTransform.position;
        currentScene.character.rotation = characterTransform.rotation;
        currentScene.character.scale = characterTransform.scale;
    }

    auto &voxelEngine = VoxelEngine::Get();
    currentScene.chunkConfig.chunksX = voxelEngine.chunkConfig.chunksX;
    currentScene.chunkConfig.chunksY = voxelEngine.chunkConfig.chunksY;
    currentScene.chunkConfig.chunksZ = voxelEngine.chunkConfig.chunksZ;

    const unsigned int totalChunks = voxelEngine.chunkConfig.getTotalChunks();
    currentScene.chunkRecords.reserve(totalChunks);

    std::filesystem::path chunkDir = GetChunkStorageDirectory();
    std::error_code dirError;
    std::filesystem::create_directories(chunkDir, dirError);
    if (dirError)
    {
        std::cout << "Failed to create chunk directory: " << dirError.message() << std::endl;
    }

    bool allChunksSaved = true;
    for (unsigned int chunkIndex = 0; chunkIndex < totalChunks && chunkIndex < voxelEngine.voxelChunks.size(); ++chunkIndex)
    {
        const VoxelChunk &chunk = voxelEngine.voxelChunks[chunkIndex];
        std::string chunkHash = ComputeChunkHash(chunk);
        std::filesystem::path chunkFile = chunkDir / (chunkHash + ".bin");

        if (!WriteChunkToFile(chunk, chunkFile))
        {
            std::cout << "Failed to write chunk " << chunkIndex << " to " << chunkFile << std::endl;
            allChunksSaved = false;
        }

        ChunkRecord record;
        record.index = chunkIndex;
        record.hash = chunkHash;
        currentScene.chunkRecords.push_back(record);
    }

    SceneConfigParser::SaveToFile(filepath, currentScene);
    return allChunksSaved;
}

bool WorldSceneManager::LoadScene(const std::string &filepath, InputHandler &inputHandler)
{
    SceneConfig loadedScene;
    if (!SceneConfigParser::LoadFromFile(filepath, loadedScene))
    {
        return false;
    }

    auto &currentCamera = RenderCamera::Get().camera;
    currentCamera.pos = loadedScene.camera.position;

    Float3 normalizedDir = loadedScene.camera.direction.normalize();
    Float2 yawPitch = DirToYawPitch(normalizedDir);
    currentCamera.yaw = yawPitch.x;
    currentCamera.pitch = yawPitch.y;

    currentCamera.update();

    bool success = true;
    if (auto *character = inputHandler.getCharacter())
    {
        EntityTransform transform = character->getTransform();
        transform.position = loadedScene.character.position;
        transform.rotation = loadedScene.character.rotation;
        transform.scale = loadedScene.character.scale;
        character->setTransform(transform);
        character->setYaw(loadedScene.character.rotation.y);
        character->checkGroundCollision();

        auto &scene = Scene::Get();
        scene.needSceneUpdate = true;
        scene.needSceneReloadUpdate = true;
    }

    auto &voxelEngine = VoxelEngine::Get();
    if (loadedScene.chunkConfig.chunksX != 0 &&
        loadedScene.chunkConfig.chunksY != 0 &&
        loadedScene.chunkConfig.chunksZ != 0)
    {
        if (loadedScene.chunkConfig.chunksX != voxelEngine.chunkConfig.chunksX ||
            loadedScene.chunkConfig.chunksY != voxelEngine.chunkConfig.chunksY ||
            loadedScene.chunkConfig.chunksZ != voxelEngine.chunkConfig.chunksZ)
        {
            std::cout << "Scene chunk configuration mismatch. Using runtime configuration." << std::endl;
        }
    }

    if (!loadedScene.chunkRecords.empty())
    {
        std::filesystem::path chunkDir = GetChunkStorageDirectory();
        std::error_code dirError;
        std::filesystem::create_directories(chunkDir, dirError);
        if (dirError)
        {
            std::cout << "Failed to create chunk directory: " << dirError.message() << std::endl;
        }

        bool anyChunkLoaded = false;
        for (const auto &record : loadedScene.chunkRecords)
        {
            if (record.index >= voxelEngine.voxelChunks.size())
            {
                std::cout << "Chunk index out of range in scene file: " << record.index << std::endl;
                success = false;
                continue;
            }

            std::filesystem::path chunkFile = chunkDir / (record.hash + ".bin");
            if (!std::filesystem::exists(chunkFile))
            {
                std::cout << "Chunk file not found: " << chunkFile << std::endl;
                success = false;
                continue;
            }

            if (ReadChunkFromFile(voxelEngine.voxelChunks[record.index], chunkFile))
            {
                anyChunkLoaded = true;
            }
            else
            {
                std::cout << "Failed to load chunk data from: " << chunkFile << std::endl;
                success = false;
            }
        }

        if (anyChunkLoaded)
        {
            voxelEngine.reload();
        }
    }

    return success;
}

