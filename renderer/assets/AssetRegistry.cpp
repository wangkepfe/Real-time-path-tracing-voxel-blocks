#include "AssetRegistry.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <json.hpp>

using json = nlohmann::json;

namespace Assets {

bool AssetRegistry::loadFromYAML(const std::string& assetDirectory) {
    clear();
    
    std::filesystem::path assetPath(assetDirectory);
    
    // For now, we'll use JSON files instead of YAML since we have json.hpp available
    // Load materials
    if (!loadMaterials((assetPath / "materials.json").string())) {
        // Fall back to hardcoded materials if JSON doesn't exist
        loadHardcodedMaterials();
    }
    
    // Load models
    if (!loadModels((assetPath / "models.json").string())) {
        // Fall back to hardcoded models if JSON doesn't exist
        loadHardcodedModels();
    }
    
    // Load blocks
    if (!loadBlocks((assetPath / "blocks.json").string())) {
        // Fall back to hardcoded blocks if JSON doesn't exist
        loadHardcodedBlocks();
    }
    
    std::cout << "Loaded " << m_materials.size() << " materials, " 
              << m_models.size() << " models, "
              << m_blocks.size() << " blocks" << std::endl;
    
    return true;
}

void AssetRegistry::loadHardcodedMaterials() {
    // CRITICAL: Material order MUST match the original system exactly!
    // The original creates materials in this specific order:
    // 0-11: Block texture materials from GetTextureFiles()
    // 12: Test material
    // 13: Leaves material  
    // 14: Lantern base material
    // 15: Lantern light material
    // 16: Minecraft character material
    
    std::vector<std::string> textureFiles = {
        "rocky_trail", "brown_mud_leaves_01", "aerial_rocks_02",
        "bark_willow_02", "rocky_trail", "aerial_beach_01",
        "gray_rocks", "stone_tiles_02", "seaworn_stone_tiles",
        "beige_wall_001", "wood_planks", "wood_planks"
    };
    
    // Materials 0-11: Create materials for each texture file (matches original loop)
    for (size_t i = 0; i < textureFiles.size(); ++i) {
        MaterialDefinition mat;
        mat.id = "material_" + std::to_string(i);
        mat.name = textureFiles[i];
        mat.textures.albedo = "data/textures/" + textureFiles[i] + "_albedo.png";
        mat.textures.normal = "data/textures/" + textureFiles[i] + "_normal.png";
        mat.textures.roughness = "data/textures/" + textureFiles[i] + "_rough.png";
        mat.properties.uv_scale = 2.5f;
        mat.properties.use_world_grid_uv = true;
        // Note: Original doesn't set roughness in properties, textures provide it
        
        m_materialIndex[mat.id] = m_materials.size();
        m_materials.push_back(mat);
    }
    
    // Material 12: Test material
    {
        MaterialDefinition mat;
        mat.id = "test1";
        mat.name = "Test Material";
        mat.properties.albedo = Float3(1.0f);
        mat.properties.roughness = 0.0f;
        mat.properties.translucency = 0.0f;
        
        m_materialIndex[mat.id] = m_materials.size();
        m_materials.push_back(mat);
    }
    
    // Material 13: Leaves material
    {
        MaterialDefinition mat;
        mat.id = "leaves";
        mat.name = "Leaves Material";
        mat.textures.albedo = "data/textures/GreenLeaf10_4K_back_albedo.png";
        mat.textures.normal = "data/textures/GreenLeaf10_4K_back_normal.png";
        mat.textures.roughness = "data/textures/GreenLeaf10_4K_back_rough.png";
        mat.properties.translucency = 0.5f;
        mat.properties.is_thinfilm = true;
        
        m_materialIndex[mat.id] = m_materials.size();
        m_materials.push_back(mat);
    }
    
    // Material 14: Lantern base
    {
        MaterialDefinition mat;
        mat.id = "lantern_base";
        mat.name = "Lantern Base Material";
        mat.textures.albedo = "data/textures/beaten-up-metal1_albedo.png";
        mat.textures.normal = "data/textures/beaten-up-metal1_normal.png";
        mat.textures.roughness = "data/textures/beaten-up-metal1_rough.png";
        mat.textures.metallic = "data/textures/beaten-up-metal1_metal.png";
        mat.properties.use_world_grid_uv = true;
        
        m_materialIndex[mat.id] = m_materials.size();
        m_materials.push_back(mat);
    }
    
    // Material 15: Lantern light (emissive)
    {
        MaterialDefinition mat;
        mat.id = "lantern_light";
        mat.name = "Lantern Light Material";
        mat.properties.emissive_radiance = Float3(255.0f, 182.0f, 78.0f) / 255.0f * 10000.0f / (683.0f * 108.0f);
        mat.properties.is_emissive = true;
        
        m_materialIndex[mat.id] = m_materials.size();
        m_materials.push_back(mat);
    }
    
    // Material 16: Minecraft character
    {
        MaterialDefinition mat;
        mat.id = "minecraft_character";
        mat.name = "Minecraft Character Material";
        mat.textures.albedo = "data/textures/high_fidelity_pink_smoothie_albedo.png";
        mat.textures.normal = "data/textures/high_fidelity_pink_smoothie_normal.png";
        mat.textures.roughness = "data/textures/high_fidelity_pink_smoothie_roughness.png";
        mat.properties.albedo = Float3(1.0f, 1.0f, 1.0f);
        mat.properties.roughness = 1.0f;
        mat.properties.metallic = 0.0f;
        mat.properties.use_world_grid_uv = false;
        mat.properties.uv_scale = 1.0f;
        
        m_materialIndex[mat.id] = m_materials.size();
        m_materials.push_back(mat);
    }
}

void AssetRegistry::loadHardcodedModels() {
    // Test plane model
    {
        ModelDefinition model;
        model.id = "test_plane";
        model.name = "Test Plane Model";
        model.file = "models/test_plane.obj";
        model.type = "instanced";
        model.block_type = 13; // BlockTypeTest1
        
        m_modelIndex[model.id] = m_models.size();
        m_models.push_back(model);
    }
    
    // Leaves cube
    {
        ModelDefinition model;
        model.id = "leaves_cube";
        model.name = "Leaves Cube Model";
        model.file = "models/leavesCube4.obj";
        model.type = "instanced";
        model.block_type = 14; // BlockTypeLeaves
        
        m_modelIndex[model.id] = m_models.size();
        m_models.push_back(model);
    }
    
    // Lantern base
    {
        ModelDefinition model;
        model.id = "lantern_base_model";
        model.name = "Lantern Base Model";
        model.file = "models/lanternBase.obj";
        model.type = "instanced";
        model.block_type = 15; // BlockTypeTestLightBase
        
        m_modelIndex[model.id] = m_models.size();
        m_models.push_back(model);
    }
    
    // Lantern light
    {
        ModelDefinition model;
        model.id = "lantern_light_model";
        model.name = "Lantern Light Model";
        model.file = "models/lanternLight.obj";
        model.type = "instanced";
        model.block_type = 16; // BlockTypeTestLight
        
        m_modelIndex[model.id] = m_models.size();
        m_models.push_back(model);
    }
    
    // Minecraft character
    {
        ModelDefinition model;
        model.id = "minecraft_character_model";
        model.name = "Minecraft Character Model";
        model.file = "models/character-pink-smoothie.gltf";
        model.type = "entity";
        model.entity_type = 0; // EntityTypeMinecraftCharacter
        model.has_animation = true;
        
        m_modelIndex[model.id] = m_models.size();
        m_models.push_back(model);
    }
}

void AssetRegistry::loadHardcodedBlocks() {
    // Block definitions matching the original BlockType enum
    const std::vector<std::tuple<int, std::string, std::string, std::string, std::string, bool, bool>> blockDefs = {
        {0, "Empty", "BlockTypeEmpty", "", "", false, false},
        {1, "Sand", "BlockTypeSand", "material_0", "", false, false},
        {2, "Soil", "BlockTypeSoil", "material_1", "", false, false},
        {3, "Cliff", "BlockTypeCliff", "material_2", "", false, false},
        {4, "Trunk", "BlockTypeTrunk", "material_3", "", false, false},
        {5, "Unused1", "BlockTypeUnused1", "material_4", "", false, false},
        {6, "Unused2", "BlockTypeUnused2", "material_5", "", false, false},
        {7, "Rocks", "BlockTypeRocks", "material_6", "", false, false},
        {8, "Floor", "BlockTypeFloor", "material_7", "", false, false},
        {9, "Brick", "BlockTypeBrick", "material_8", "", false, false},
        {10, "Wall", "BlockTypeWall", "material_9", "", false, false},
        {11, "Plank", "BlockTypePlank", "material_10", "", false, false},
        {12, "Plank2", "BlockTypePlank2", "material_11", "", false, false},
        {13, "Test1", "BlockTypeTest1", "test1", "test_plane", true, false},
        {14, "Leaves", "BlockTypeLeaves", "leaves", "leaves_cube", true, false},
        {15, "TestLightBase", "BlockTypeTestLightBase", "lantern_base", "lantern_base_model", true, true},
        {16, "TestLight", "BlockTypeTestLight", "lantern_light", "lantern_light_model", true, false}
    };
    
    for (const auto& [id, name, type, materialId, modelId, isInstanced, isBaseLight] : blockDefs) {
        BlockDefinition block;
        block.id = id;
        block.name = name;
        block.type = type;
        if (!materialId.empty()) block.material_id = materialId;
        if (!modelId.empty()) block.model_id = modelId;
        block.is_instanced = isInstanced;
        block.is_base_light = isBaseLight;
        block.is_emissive = (id == 16); // BlockTypeTestLight
        
        m_blockIndex[block.id] = m_blocks.size();
        m_blockIdIndex[block.id] = m_blocks.size();
        m_blocks.push_back(block);
    }
}

bool AssetRegistry::loadMaterials(const std::string& filepath) {
    // Try to load from JSON file
    if (!std::filesystem::exists(filepath)) {
        return false;
    }
    
    try {
        std::ifstream file(filepath);
        json j;
        file >> j;
        
        // Parse JSON materials - implementation would go here
        // For now, return false to use hardcoded
        return false;
    } catch (...) {
        return false;
    }
}

bool AssetRegistry::loadModels(const std::string& filepath) {
    // Try to load from JSON file
    if (!std::filesystem::exists(filepath)) {
        return false;
    }
    
    return false; // Use hardcoded for now
}

bool AssetRegistry::loadBlocks(const std::string& filepath) {
    // Try to load from JSON file
    if (!std::filesystem::exists(filepath)) {
        return false;
    }
    
    return false; // Use hardcoded for now
}

const MaterialDefinition* AssetRegistry::getMaterial(const std::string& id) const {
    auto it = m_materialIndex.find(id);
    if (it != m_materialIndex.end()) {
        return &m_materials[it->second];
    }
    return nullptr;
}

MaterialDefinition* AssetRegistry::getMaterialMutable(const std::string& id) {
    auto it = m_materialIndex.find(id);
    if (it != m_materialIndex.end()) {
        return &m_materials[it->second];
    }
    return nullptr;
}

const ModelDefinition* AssetRegistry::getModel(const std::string& id) const {
    auto it = m_modelIndex.find(id);
    if (it != m_modelIndex.end()) {
        return &m_models[it->second];
    }
    return nullptr;
}

ModelDefinition* AssetRegistry::getModelMutable(const std::string& id) {
    auto it = m_modelIndex.find(id);
    if (it != m_modelIndex.end()) {
        return &m_models[it->second];
    }
    return nullptr;
}

const BlockDefinition* AssetRegistry::getBlock(int blockType) const {
    auto it = m_blockIndex.find(blockType);
    if (it != m_blockIndex.end()) {
        return &m_blocks[it->second];
    }
    return nullptr;
}

const BlockDefinition* AssetRegistry::getBlockById(int id) const {
    auto it = m_blockIdIndex.find(id);
    if (it != m_blockIdIndex.end()) {
        return &m_blocks[it->second];
    }
    return nullptr;
}

const MaterialDefinition* AssetRegistry::getMaterialForBlock(int blockType) const {
    const BlockDefinition* block = getBlock(blockType);
    if (block && block->material_id.has_value()) {
        return getMaterial(block->material_id.value());
    }
    return nullptr;
}

const ModelDefinition* AssetRegistry::getModelForBlock(int blockType) const {
    const BlockDefinition* block = getBlock(blockType);
    if (block && block->model_id.has_value()) {
        return getModel(block->model_id.value());
    }
    return nullptr;
}

void AssetRegistry::clear() {
    m_materials.clear();
    m_models.clear();
    m_blocks.clear();
    m_materialIndex.clear();
    m_modelIndex.clear();
    m_blockIndex.clear();
    m_blockIdIndex.clear();
}

} // namespace Assets