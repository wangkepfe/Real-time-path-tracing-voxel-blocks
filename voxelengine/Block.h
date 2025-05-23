#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "shaders/LinearMath.h"

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
    BlockTypePlank2,

    BlockTypeTest1,
    BlockTypeLeaves,
    BlockTypeTestLightBase,
    BlockTypeTestLight,

    BlockTypeNum,
};

inline bool IsInstancedBlockType(unsigned int blockId) { return blockId >= BlockTypeTest1; }
inline bool IsUninstancedBlockType(unsigned int blockId) { return blockId < BlockTypeTest1; }
inline bool IsTransparentBlockType(unsigned int blockId) { return false; }
inline bool IsBaseLightBlockType(unsigned int blockId) { return blockId == BlockTypeTestLightBase; }

inline unsigned int GetNumInstancedBlockTypes() { return BlockTypeNum - BlockTypeTest1; }
inline unsigned int GetNumUninstancedBlockTypes() { return BlockTypeTest1 - 1; }
inline unsigned int GetNumBlockTypes() { return BlockTypeNum; }

inline unsigned int GetUninstancedObjectIdBegin() { return 0; }
inline unsigned int GetUninstancedObjectIdEnd() { return BlockTypeTest1 - 1; }

inline unsigned int GetInstancedObjectIdBegin() { return BlockTypeTest1 - 1; }
inline unsigned int GetInstancedObjectIdEnd() { return BlockTypeNum - 1; }

inline unsigned int ObjectIdToBlockId(unsigned int i) { return i + 1; }
inline unsigned int BlockIdToObjectId(unsigned int i) { return i - 1; }

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
        "wood_planks",         // Plank2
    };
    return textureFiles;
}

inline const std::string &GetModelFileName(int blockType)
{
    static const std::unordered_map<int, std::string> modelFiles = {
        {BlockTypeTest1, "data/test_plane.obj"},
        {BlockTypeLeaves, "data/leavesCube4.obj"},
        {BlockTypeTestLightBase, "data/lanternBase.obj"},
        {BlockTypeTestLight, "data/lanternLight.obj"},
    };
    return modelFiles.at(blockType);
}

inline bool IsBlockEmissive(int blockType)
{
    static const std::unordered_set<int> emissiveBlocks = {
        BlockTypeTestLight,
    };
    return emissiveBlocks.count(blockType);
}

inline Float3 GetEmissiveRadiance(int blockType)
{
    static const std::unordered_map<int, Float3> emissiveBlockRadiance = {
        {BlockTypeTestLight, Float3(255.0f, 182.0f, 78.0f) / 255.0f * 10000.0f / (683.0f * 108.0f)}, // 1500 Kelvin color temperature, 10000 Lux
    };
    return emissiveBlockRadiance.at(blockType);
}
