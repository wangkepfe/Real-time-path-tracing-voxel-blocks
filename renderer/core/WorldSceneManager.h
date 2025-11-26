#pragma once

#include <optional>
#include <string>
#include <vector>

struct Camera;
class InputHandler;

class WorldSceneManager
{
public:
    static bool SaveScene(const std::string &filepath, const Camera &camera, InputHandler &inputHandler);
    static bool LoadScene(const std::string &filepath, InputHandler &inputHandler);

    static bool SaveWorld(const std::string &worldName, const Camera &camera, InputHandler &inputHandler);
    static bool LoadWorld(const std::string &worldName, InputHandler &inputHandler);
    static bool CreateWorld(const std::string &worldName, InputHandler &inputHandler);

    static bool WorldExists(const std::string &worldName);
    static std::vector<std::string> ListWorlds();

    static std::optional<std::string> GetLastPlayedWorld();
    static bool SetLastPlayedWorld(const std::string &worldName);

    static std::string GenerateDefaultWorldName();
    static bool ValidateWorldName(const std::string &candidate, std::string &normalized, std::string &errorMessage);
};
