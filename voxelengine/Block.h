#pragma once

#include <vector>
#include <string>
#include <unordered_map>

enum BlockType
{
    BlockTypeEmpty,

    BlockTypeSand,
    BlockTypeSoil,
    BlockTypeCliff,

    BlockTypeTrunk,
    BlockTypeUnused1,
    BlockTypeUnused2,

    BlockTypeRocks,
    BlockTypeFloor,
    BlockTypeBrick,
    BlockTypeWall,
    BlockTypePlank,

    BlockTypeWater,

    BlockTypeTest1,
    BlockTypeLeaves,

    BlockTypeNum,
};

inline const std::vector<std::string> &GetTextureFiles()
{
    static const std::vector<std::string> textureFiles = {
        "rocky_trail",         // Sand
        "brown_mud_leaves_01", // Soil
        "aerial_rocks_02",     // Cliff

        "bark_willow_02",  // Trunk
        "rocky_trail",     // unused1
        "aerial_beach_01", // unused2

        "gray_rocks",          // Rocks
        "stone_tiles_02",      // Floor
        "seaworn_stone_tiles", // Brick
        "beige_wall_001",      // Wall
        "wood_planks",         // Plank
    };
    return textureFiles;
}

inline const std::string &GetModelFileName(int blockType)
{
    static const std::unordered_map<int, std::string> modelFiles = {
        {BlockTypeTest1, "data/test_plane.obj"},
        {BlockTypeLeaves, "data/leavesCube4.obj"},
    };
    return modelFiles.at(blockType);
}