#pragma once

#include <string>

struct Camera;
class InputHandler;

class WorldSceneManager
{
public:
    static bool SaveScene(const std::string &filepath, const Camera &camera, InputHandler &inputHandler);
    static bool LoadScene(const std::string &filepath, InputHandler &inputHandler);
};

