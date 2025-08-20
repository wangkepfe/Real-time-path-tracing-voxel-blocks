#include "core/SceneConfig.h"
#include <iostream>
#include <algorithm>
#include <cctype>

bool SceneConfigParser::LoadFromFile(const std::string &filepath, SceneConfig &config)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Failed to open scene config file: " << filepath << std::endl;
        return false;
    }

    std::string line;
    std::string currentSection;

    while (std::getline(file, line))
    {
        line = trim(line);

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#')
            continue;

        // Check for section headers (e.g., "camera:")
        if (line.back() == ':')
        {
            currentSection = line.substr(0, line.length() - 1);
            continue;
        }

        // Parse key-value pairs
        size_t colonPos = line.find(':');
        if (colonPos == std::string::npos)
            continue;

        std::string key = trim(line.substr(0, colonPos));
        std::string value = trim(line.substr(colonPos + 1));

        // Parse camera settings
        if (currentSection == "camera")
        {
            if (key == "position")
            {
                parseFloat3(value, config.camera.position);
            }
            else if (key == "direction")
            {
                parseFloat3(value, config.camera.direction);
            }
            else if (key == "up")
            {
                parseFloat3(value, config.camera.up);
            }
            else if (key == "fov")
            {
                parseFloat(value, config.camera.fov);
            }
        }
        // Parse character settings
        else if (currentSection == "character")
        {
            if (key == "position")
            {
                parseFloat3(value, config.character.position);
            }
            else if (key == "rotation")
            {
                parseFloat3(value, config.character.rotation);
            }
            else if (key == "scale")
            {
                parseFloat3(value, config.character.scale);
            }
        }
    }

    // Normalize camera direction
    config.camera.normalize();

    std::cout << "Loaded scene config from: " << filepath << std::endl;

    return true;
}

void SceneConfigParser::SaveToFile(const std::string &filepath, const SceneConfig &config)
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Failed to create scene config file: " << filepath << std::endl;
        return;
    }

    file << "# Scene Configuration File\n";
    file << "# Generated automatically\n\n";

    file << "camera:\n";
    file << "  position: " << float3ToString(config.camera.position) << "\n";
    file << "  direction: " << float3ToString(config.camera.direction) << "\n";
    file << "  up: " << float3ToString(config.camera.up) << "\n";
    file << "  fov: " << config.camera.fov << "\n\n";

    file << "character:\n";
    file << "  position: " << float3ToString(config.character.position) << "\n";
    file << "  rotation: " << float3ToString(config.character.rotation) << "\n";
    file << "  scale: " << float3ToString(config.character.scale) << "\n";

    std::cout << "Saved scene config to: " << filepath << std::endl;
}

SceneConfig SceneConfigParser::CreateDefault()
{
    SceneConfig config;
    config.camera.position = Float3(20.0f, 15.0f, 20.0f);
    config.camera.direction = Float3(1.0f, -0.3f, 1.0f);
    config.camera.up = Float3(0.0f, 1.0f, 0.0f);
    config.camera.fov = 90.0f;
    config.camera.normalize();

    config.character.position = Float3(16.0f, 10.0f, 16.0f);
    config.character.rotation = Float3(0.0f, 0.0f, 0.0f);
    config.character.scale = Float3(1.0f, 1.0f, 1.0f);

    return config;
}

std::string SceneConfigParser::trim(const std::string &str)
{
    size_t start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
        return "";

    size_t end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

bool SceneConfigParser::parseFloat3(const std::string &value, Float3 &result)
{
    // Expected format: [x, y, z] or x, y, z
    std::string cleanValue = value;

    // Remove brackets if present
    if (cleanValue.front() == '[')
        cleanValue = cleanValue.substr(1);
    if (cleanValue.back() == ']')
        cleanValue = cleanValue.substr(0, cleanValue.length() - 1);

    std::istringstream iss(cleanValue);
    std::string token;
    float values[3];
    int count = 0;

    while (std::getline(iss, token, ',') && count < 3)
    {
        token = trim(token);
        try
        {
            values[count] = std::stof(token);
            count++;
        }
        catch (const std::exception &)
        {
            return false;
        }
    }

    if (count == 3)
    {
        result = Float3(values[0], values[1], values[2]);
        return true;
    }

    return false;
}

bool SceneConfigParser::parseFloat(const std::string &value, float &result)
{
    try
    {
        result = std::stof(trim(value));
        return true;
    }
    catch (const std::exception &)
    {
        return false;
    }
}

std::string SceneConfigParser::float3ToString(const Float3 &vec)
{
    std::ostringstream oss;
    oss << "[" << vec.x << ", " << vec.y << ", " << vec.z << "]";
    return oss.str();
}