#include "core/WorldSceneManager.h"

#include "core/SceneConfig.h"
#include "core/InputHandler.h"
#include "core/Character.h"
#include "core/Scene.h"
#include "core/RenderCamera.h"
#include "voxelengine/VoxelEngine.h"
#include "shaders/Camera.h"
#include "shaders/LinearMath.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <cstdint>
#include <ctime>

namespace
{
    std::filesystem::path GetAppDataDirectory()
    {
        std::filesystem::path chunkDir;
#ifdef _WIN32
        const char *appData = std::getenv("APPDATA");
        if (appData)
        {
            chunkDir = std::filesystem::path(appData) / "wotw";
        }
        else
        {
            chunkDir = "wotw";
        }
#else
        const char *home = std::getenv("HOME");
        if (home)
        {
            chunkDir = std::filesystem::path(home) / ".local" / "share" / "wotw";
        }
        else
        {
            chunkDir = "wotw";
        }
#endif
        return chunkDir;
    }

    std::filesystem::path GetChunkStorageDirectory()
    {
        return GetAppDataDirectory() / "chunks";
    }

    std::filesystem::path GetWorldsDirectory()
    {
        return GetAppDataDirectory() / "worlds";
    }

    std::filesystem::path GetMetadataFilePath()
    {
        return GetAppDataDirectory() / "metadata.yaml";
    }

    std::filesystem::path GetWorldScenePath(const std::string &worldName)
    {
        if (worldName.empty())
        {
            return {};
        }
        return GetWorldsDirectory() / worldName / "scene.yaml";
    }

    bool EnsureDirectoryExists(const std::filesystem::path &path)
    {
        if (path.empty())
        {
            return false;
        }

        std::error_code ec;
        std::filesystem::create_directories(path, ec);
        if (ec)
        {
            std::cout << "Failed to create directory: " << path << " (" << ec.message() << ")" << std::endl;
            return false;
        }
        return true;
    }

    std::string TrimCopy(const std::string &value)
    {
        const auto begin = value.find_first_not_of(" \t\r\n");
        if (begin == std::string::npos)
        {
            return {};
        }

        const auto end = value.find_last_not_of(" \t\r\n");
        return value.substr(begin, end - begin + 1);
    }

    bool ContainsIllegalCharacters(const std::string &value)
    {
#ifdef _WIN32
        constexpr const char *kIllegal = "<>:\"/\\|?*";
#else
        constexpr const char *kIllegal = "/";
#endif
        return value.find_first_of(kIllegal) != std::string::npos;
    }

#ifdef _WIN32
    bool EqualsIgnoreCase(const std::string &lhs, const std::string &rhs)
    {
        if (lhs.size() != rhs.size())
        {
            return false;
        }

        for (size_t i = 0; i < lhs.size(); ++i)
        {
            unsigned char left = static_cast<unsigned char>(lhs[i]);
            unsigned char right = static_cast<unsigned char>(rhs[i]);
            if (std::tolower(left) != std::tolower(right))
            {
                return false;
            }
        }
        return true;
    }

    bool IsReservedFilename(const std::string &value)
    {
        static const std::array<std::string, 22> kReserved = {
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"};

        for (const auto &reserved : kReserved)
        {
            if (EqualsIgnoreCase(value, reserved))
            {
                return true;
            }
        }
        return false;
    }
#else
    bool IsReservedFilename(const std::string &)
    {
        return false;
    }
#endif

    void GenerateDefaultChunks(VoxelEngine &engine)
    {
        const unsigned int totalChunks = engine.chunkConfig.getTotalChunks();
        if (engine.voxelChunks.size() != totalChunks)
        {
            engine.voxelChunks.resize(totalChunks);
        }

        std::vector<Voxel *> deviceChunks(totalChunks, nullptr);
        for (unsigned int chunkIndex = 0; chunkIndex < totalChunks; ++chunkIndex)
        {
            initVoxelsMultiChunk(engine.voxelChunks[chunkIndex], &deviceChunks[chunkIndex], chunkIndex, engine.chunkConfig);
        }

        for (auto *chunkDevicePtr : deviceChunks)
        {
            if (chunkDevicePtr)
            {
                freeDeviceVoxelData(chunkDevicePtr);
            }
        }
    }

    std::optional<std::string> ReadLastWorldFromMetadata()
    {
        const auto metadataPath = GetMetadataFilePath();
        std::ifstream in(metadataPath);
        if (!in.is_open())
        {
            return std::nullopt;
        }

        std::string line;
        while (std::getline(in, line))
        {
            std::string trimmed = TrimCopy(line);
            if (trimmed.rfind("last_world", 0) != 0)
            {
                continue;
            }

            size_t colon = trimmed.find(':');
            if (colon == std::string::npos)
            {
                continue;
            }

            std::string value = TrimCopy(trimmed.substr(colon + 1));
            if (!value.empty() && value.front() == '"' && value.back() == '"' && value.size() >= 2)
            {
                value = value.substr(1, value.size() - 2);
            }
            return value;
        }

        return std::nullopt;
    }

    bool WriteLastWorldToMetadata(const std::string &worldName)
    {
        const auto metadataPath = GetMetadataFilePath();
        if (!EnsureDirectoryExists(metadataPath.parent_path()))
        {
            return false;
        }

        std::ofstream out(metadataPath, std::ios::out | std::ios::trunc);
        if (!out.is_open())
        {
            std::cout << "Failed to write metadata file: " << metadataPath << std::endl;
            return false;
        }

        out << "# Game metadata - automatically generated\n";
        out << "last_world: " << worldName << "\n";
        return true;
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

bool WorldSceneManager::SaveWorld(const std::string &worldName, const Camera &camera, InputHandler &inputHandler)
{
    std::string normalized = TrimCopy(worldName);
    if (normalized.empty())
    {
        return false;
    }

    const auto scenePath = GetWorldScenePath(normalized);
    if (!EnsureDirectoryExists(scenePath.parent_path()))
    {
        return false;
    }

    return SaveScene(scenePath.string(), camera, inputHandler);
}

bool WorldSceneManager::LoadWorld(const std::string &worldName, InputHandler &inputHandler)
{
    std::string normalized = TrimCopy(worldName);
    if (normalized.empty())
    {
        return false;
    }

    const auto scenePath = GetWorldScenePath(normalized);
    if (!std::filesystem::exists(scenePath))
    {
        std::cout << "World scene file not found: " << scenePath << std::endl;
        return false;
    }

    if (!LoadScene(scenePath.string(), inputHandler))
    {
        return false;
    }

    SetLastPlayedWorld(normalized);
    return true;
}

bool WorldSceneManager::CreateWorld(const std::string &worldName, InputHandler &inputHandler)
{
    std::string normalized;
    std::string error;
    if (!ValidateWorldName(worldName, normalized, error))
    {
        std::cout << "Invalid world name: " << error << std::endl;
        return false;
    }

    if (WorldExists(normalized))
    {
        std::cout << "World already exists: " << normalized << std::endl;
        return false;
    }

    const auto scenePath = GetWorldScenePath(normalized);
    if (!EnsureDirectoryExists(scenePath.parent_path()))
    {
        return false;
    }

    auto &voxelEngine = VoxelEngine::Get();
    GenerateDefaultChunks(voxelEngine);

    SceneConfig defaultScene = SceneConfigParser::CreateDefault();
    Camera defaultCamera{};
    defaultCamera.pos = defaultScene.camera.position;
    defaultCamera.dir = defaultScene.camera.direction;

    if (!SaveScene(scenePath.string(), defaultCamera, inputHandler))
    {
        return false;
    }

    if (!LoadScene(scenePath.string(), inputHandler))
    {
        return false;
    }

    SetLastPlayedWorld(normalized);
    return true;
}

bool WorldSceneManager::WorldExists(const std::string &worldName)
{
    std::string normalized = TrimCopy(worldName);
    if (normalized.empty())
    {
        return false;
    }

    const auto scenePath = GetWorldScenePath(normalized);
    std::error_code ec;
    return std::filesystem::exists(scenePath, ec) && !ec;
}

std::vector<std::string> WorldSceneManager::ListWorlds()
{
    namespace fs = std::filesystem;
    std::vector<std::string> result;

    const auto worldsDir = GetWorldsDirectory();
    std::error_code ec;
    fs::create_directories(worldsDir, ec);

    if (ec)
    {
        std::cout << "Failed to prepare worlds directory: " << worldsDir << " (" << ec.message() << ")" << std::endl;
        return result;
    }

    fs::directory_iterator it(worldsDir, ec);
    if (ec)
    {
        return result;
    }

    fs::directory_iterator end;
    for (; it != end; it.increment(ec))
    {
        if (ec)
        {
            break;
        }

        if (!it->is_directory())
        {
            continue;
        }

        fs::path sceneFile = it->path() / "scene.yaml";
        if (fs::exists(sceneFile))
        {
            result.push_back(it->path().filename().string());
        }
    }

    std::sort(result.begin(), result.end(), [](const std::string &lhs, const std::string &rhs) {
        return std::lexicographical_compare(
            lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
            [](unsigned char a, unsigned char b) { return std::tolower(a) < std::tolower(b); });
    });

    return result;
}

std::optional<std::string> WorldSceneManager::GetLastPlayedWorld()
{
    return ReadLastWorldFromMetadata();
}

bool WorldSceneManager::SetLastPlayedWorld(const std::string &worldName)
{
    std::string normalized = TrimCopy(worldName);
    if (normalized.empty())
    {
        return false;
    }

    return WriteLastWorldToMetadata(normalized);
}

std::string WorldSceneManager::GenerateDefaultWorldName()
{
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const std::time_t currentTime = clock::to_time_t(now);
    std::tm localTime{};
#ifdef _WIN32
    localtime_s(&localTime, &currentTime);
#else
    localtime_r(&currentTime, &localTime);
#endif

    std::ostringstream oss;
    oss << "World-" << std::put_time(&localTime, "%Y%m%d-%H%M%S");
    return oss.str();
}

bool WorldSceneManager::ValidateWorldName(const std::string &candidate, std::string &normalized, std::string &errorMessage)
{
    normalized = TrimCopy(candidate);
    if (normalized.empty())
    {
        errorMessage = "World name cannot be empty.";
        return false;
    }

    if (normalized.size() > 64)
    {
        errorMessage = "World name must be 64 characters or fewer.";
        return false;
    }

    if (ContainsIllegalCharacters(normalized))
    {
        errorMessage = "World name contains unsupported characters.";
        return false;
    }

    if (normalized == "." || normalized == "..")
    {
        errorMessage = "World name is not allowed.";
        return false;
    }

#ifdef _WIN32
    if (!normalized.empty() && normalized.back() == '.')
    {
        errorMessage = "World name cannot end with a period.";
        return false;
    }
#endif

    if (IsReservedFilename(normalized))
    {
        errorMessage = "World name is reserved by the operating system.";
        return false;
    }

    return true;
}

