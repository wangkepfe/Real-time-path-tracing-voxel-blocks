#pragma once

#include <vector>
#include <string>

enum BlockType
{
    BlockTypeEmpty,

    BlockTypeSand,
    BlockTypeSoil,
    BlockTypeCliff,

    BlockTypeTrunk,
    BlockTypeLeaves,
    BlockTypeGrass,

    BlockTypeRocks,
    BlockTypeFloor,
    BlockTypeBrick,
    BlockTypeWall,
    BlockTypePlank,

    BlockTypeMaxNum,
};

inline const std::vector<std::string> &GetTextureFiles()
{
    static const std::vector<std::string> textureFiles = {
        "rocky_trail",         // Sand
        "brown_mud_leaves_01", // Soil
        "aerial_rocks_02",     // Cliff

        "bark_willow_02",  // Trunk
        "rocky_trail",     // Leaves - unused
        "aerial_beach_01", // Grass - unused

        "gray_rocks",          // Rocks
        "stone_tiles_02",      // Floor
        "seaworn_stone_tiles", // Brick
        "beige_wall_001",      // Wall
        "wood_planks",         // Plank
    };
    return textureFiles;
}