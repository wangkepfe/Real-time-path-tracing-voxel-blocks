#pragma once

#include "shaders/LinearMath.h"
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

struct CameraConfig
{
    Float3 position = Float3(20.0f, 15.0f, 20.0f);
    Float3 direction = Float3(-1.0f, -0.3f, -1.0f);
    Float3 up = Float3(0.0f, 1.0f, 0.0f);
    float fov = 90.0f;

    // Normalize direction vector
    void normalize()
    {
        direction = ::normalize(direction);
    }
};

struct CharacterConfig
{
    Float3 position = Float3(16.0f, 10.0f, 16.0f);
    Float3 rotation = Float3(0.0f, 0.0f, 0.0f);
    Float3 scale = Float3(1.0f, 1.0f, 1.0f);
};

struct ChunkConfigData
{
    unsigned int chunksX = 0;
    unsigned int chunksY = 0;
    unsigned int chunksZ = 0;
};

struct ChunkRecord
{
    unsigned int index = 0;
    std::string hash;
};

struct SceneConfig
{
    CameraConfig camera;
    CharacterConfig character;
    ChunkConfigData chunkConfig;
    std::vector<ChunkRecord> chunkRecords;

    // Add more scene elements here in the future
    // LightConfig lights;
    // EnvironmentConfig environment;
    // ObjectConfig objects;
};

class SceneConfigParser
{
public:
    static bool LoadFromFile(const std::string& filepath, SceneConfig& config);
    static void SaveToFile(const std::string& filepath, const SceneConfig& config);
    static SceneConfig CreateDefault();

private:
    static std::string trim(const std::string& str);
    static bool parseFloat3(const std::string& value, Float3& result);
    static bool parseFloat(const std::string& value, float& result);
    static bool parseUnsignedInt(const std::string& value, unsigned int& result);
    static std::string float3ToString(const Float3& vec);
};
