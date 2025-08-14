#include "AssetRegistry.h"
#include "TextureManager.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <yaml-cpp/yaml.h>

namespace Assets {


bool AssetRegistry::loadFromYAML(const std::string& assetDirectory) {
    // If already loaded, return success without reloading
    if (m_isLoaded) {
        return true;
    }
    
    clear();
    
    std::filesystem::path assetPath(assetDirectory);
    
    // Load YAML files - blocks first so models can reference block types
    if (!loadBlocks((assetPath / "blocks.yaml").string())) {
        std::cerr << "ERROR: Failed to load blocks from YAML: " << (assetPath / "blocks.yaml").string() << std::endl;
        return false;
    }
    
    if (!loadMaterials((assetPath / "materials.yaml").string())) {
        std::cerr << "ERROR: Failed to load materials from YAML: " << (assetPath / "materials.yaml").string() << std::endl;
        return false;
    }
    
    if (!loadModels((assetPath / "models.yaml").string())) {
        std::cerr << "ERROR: Failed to load models from YAML: " << (assetPath / "models.yaml").string() << std::endl;
        return false;
    }
    
    std::cout << "Successfully loaded " << m_materials.size() << " materials, " 
              << m_models.size() << " models, "
              << m_blocks.size() << " blocks from YAML" << std::endl;
    
    // Initialize texture manager with collected texture paths
    initializeTextureManager();
    
    // Mark as loaded to prevent duplicate loading
    m_isLoaded = true;
    
    return true;
}

bool AssetRegistry::loadMaterials(const std::string& filepath) {
    if (!std::filesystem::exists(filepath)) {
        return false;
    }
    
    try {
        YAML::Node root = YAML::LoadFile(filepath);
        
        if (!root["materials"] || !root["materials"].IsSequence()) {
            std::cerr << "Invalid YAML structure: missing 'materials' sequence in " << filepath << std::endl;
            return false;
        }
        
        const auto& materialsNode = root["materials"];
        
        for (const auto& materialNodePtr : materialsNode) {
            const auto& materialNode = *materialNodePtr;
            if (!materialNode.IsMap()) continue;
            
            MaterialDefinition material;
            
            // Parse basic properties
            if (materialNode["id"]) {
                material.id = materialNode["id"].as<std::string>();
            }
            if (materialNode["name"]) {
                material.name = materialNode["name"].as<std::string>();
            }
            
            // Parse textures
            
            if (materialNode["textures"]) {
                if (materialNode["textures"].IsMap()) {
                    // Parse textures from YAML (when YAML parser is fixed)
                    const auto& texturesNode = materialNode["textures"];
                    
                    if (texturesNode["albedo"]) {
                        material.textures.albedo = "data/" + texturesNode["albedo"].as<std::string>();
                    }
                    if (texturesNode["normal"]) {
                        material.textures.normal = "data/" + texturesNode["normal"].as<std::string>();
                    }
                    if (texturesNode["roughness"]) {
                        material.textures.roughness = "data/" + texturesNode["roughness"].as<std::string>();
                    }
                    if (texturesNode["metallic"]) {
                        material.textures.metallic = "data/" + texturesNode["metallic"].as<std::string>();
                    }
                    if (texturesNode["emissive"]) {
                        material.textures.emissive = "data/" + texturesNode["emissive"].as<std::string>();
                    }
                }
            }
            
            // Parse properties
            if (materialNode["properties"] && materialNode["properties"].IsMap()) {
                const auto& propsNode = materialNode["properties"];
                
                if (propsNode["albedo"] && propsNode["albedo"].IsSequence()) {
                    auto albedoSeq = propsNode["albedo"];
                    if (albedoSeq.size() >= 3) {
                        material.properties.albedo.x = albedoSeq[std::size_t(0)].as<float>();
                        material.properties.albedo.y = albedoSeq[std::size_t(1)].as<float>();
                        material.properties.albedo.z = albedoSeq[std::size_t(2)].as<float>();
                    }
                }
                if (propsNode["roughness"]) {
                    material.properties.roughness = propsNode["roughness"].as<float>();
                }
                if (propsNode["metallic"]) {
                    material.properties.metallic = propsNode["metallic"].as<float>();
                }
                if (propsNode["uv_scale"]) {
                    material.properties.uv_scale = propsNode["uv_scale"].as<float>();
                }
                if (propsNode["translucency"]) {
                    material.properties.translucency = propsNode["translucency"].as<float>();
                }
                if (propsNode["is_emissive"]) {
                    material.properties.is_emissive = propsNode["is_emissive"].as<bool>();
                }
                if (propsNode["is_thinfilm"]) {
                    material.properties.is_thinfilm = propsNode["is_thinfilm"].as<bool>();
                }
                if (propsNode["use_world_grid_uv"]) {
                    material.properties.use_world_grid_uv = propsNode["use_world_grid_uv"].as<bool>();
                }
                if (propsNode["emissive_radiance"] && propsNode["emissive_radiance"].IsSequence()) {
                    auto emissiveSeq = propsNode["emissive_radiance"];
                    if (emissiveSeq.size() >= 3) {
                        material.properties.emissive_radiance.x = emissiveSeq[std::size_t(0)].as<float>();
                        material.properties.emissive_radiance.y = emissiveSeq[std::size_t(1)].as<float>();
                        material.properties.emissive_radiance.z = emissiveSeq[std::size_t(2)].as<float>();
                    }
                }
            }
            
            // Store the material
            if (!material.id.empty()) {
                m_materialIndex[material.id] = m_materials.size();
                m_materials.push_back(material);
            }
        }
        
        std::cout << "Loaded " << m_materials.size() << " materials from YAML: " << filepath << std::endl;
        return !m_materials.empty();
        
    } catch (const YAML::Exception& e) {
        std::cerr << "YAML parsing error in " << filepath << ": " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error loading materials from " << filepath << ": " << e.what() << std::endl;
        return false;
    }
}

bool AssetRegistry::loadModels(const std::string& filepath) {
    if (!std::filesystem::exists(filepath)) {
        return false;
    }
    
    try {
        YAML::Node root = YAML::LoadFile(filepath);
        
        if (!root["models"] || !root["models"].IsSequence()) {
            std::cerr << "Invalid YAML structure: missing 'models' sequence in " << filepath << std::endl;
            return false;
        }
        
        const auto& modelsNode = root["models"];
        
        for (const auto& modelNodePtr : modelsNode) {
            const auto& modelNode = *modelNodePtr;
            if (!modelNode.IsMap()) continue;
            
            ModelDefinition model;
            
            // Parse basic properties
            if (modelNode["id"]) {
                model.id = modelNode["id"].as<std::string>();
            }
            if (modelNode["name"]) {
                model.name = modelNode["name"].as<std::string>();
            }
            if (modelNode["file"]) {
                model.file = modelNode["file"].as<std::string>();
            }
            if (modelNode["type"]) {
                model.type = modelNode["type"].as<std::string>();
            }
            if (modelNode["block_type"]) {
                std::string blockTypeStr = modelNode["block_type"].as<std::string>();
                // Map string block types to integers using loaded block data
                int blockId = getBlockIdFromType(blockTypeStr);
                if (blockId != -1) {
                    model.block_type = blockId;
                } else {
                    std::cerr << "Unknown block type: " << blockTypeStr << std::endl;
                    model.block_type = -1;
                }
            }
            if (modelNode["entity_type"]) {
                std::string entityTypeStr = modelNode["entity_type"].as<std::string>();
                // Map string entity types to integers  
                if (entityTypeStr == "EntityTypeMinecraftCharacter") model.entity_type = 0;
                else {
                    std::cerr << "Unknown entity type: " << entityTypeStr << std::endl;
                    model.entity_type = -1;
                }
            }
            if (modelNode["has_animation"]) {
                model.has_animation = modelNode["has_animation"].as<bool>();
            }
            
            // ModelDefinition doesn't have properties - those are handled individually above
            
            // Store the model
            if (!model.id.empty()) {
                m_modelIndex[model.id] = m_models.size();
                m_models.push_back(model);
            }
        }
        
        std::cout << "Loaded " << m_models.size() << " models from YAML: " << filepath << std::endl;
        return !m_models.empty();
        
    } catch (const YAML::Exception& e) {
        std::cerr << "YAML parsing error in " << filepath << ": " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error loading models from " << filepath << ": " << e.what() << std::endl;
        return false;
    }
}

bool AssetRegistry::loadBlocks(const std::string& filepath) {
    if (!std::filesystem::exists(filepath)) {
        return false;
    }
    
    try {
        YAML::Node root = YAML::LoadFile(filepath);
        
        if (!root["blocks"] || !root["blocks"].IsSequence()) {
            std::cerr << "Invalid YAML structure: missing 'blocks' sequence in " << filepath << std::endl;
            return false;
        }
        
        const auto& blocksNode = root["blocks"];
        
        for (const auto& blockNodePtr : blocksNode) {
            const auto& blockNode = *blockNodePtr;
            if (!blockNode.IsMap()) continue;
            
            BlockDefinition block;
            
            // Parse basic properties
            if (blockNode["id"]) {
                block.id = blockNode["id"].as<int>();
            }
            if (blockNode["name"]) {
                block.name = blockNode["name"].as<std::string>();
            }
            if (blockNode["type"]) {
                block.type = blockNode["type"].as<std::string>();
            }
            if (blockNode["material"]) {
                block.material_id = blockNode["material"].as<std::string>();
            }
            if (blockNode["model_id"]) {
                block.model_id = blockNode["model_id"].as<std::string>();
            }
            
            // Parse boolean properties
            if (blockNode["is_instanced"]) {
                block.is_instanced = blockNode["is_instanced"].as<bool>();
            }
            if (blockNode["is_base_light"]) {
                block.is_base_light = blockNode["is_base_light"].as<bool>();
            }
            if (blockNode["is_emissive"]) {
                block.is_emissive = blockNode["is_emissive"].as<bool>();
            }
            // cast_shadows field removed - not in BlockDefinition
            if (blockNode["is_transparent"]) {
                block.is_transparent = blockNode["is_transparent"].as<bool>();
            }
            
            // Store the block
            m_blockIndex[block.id] = m_blocks.size();
            m_blockIdIndex[block.id] = m_blocks.size();
            // Build block type string to ID mapping
            if (!block.type.empty()) {
                m_blockTypeToId[block.type] = block.id;
            }
            m_blocks.push_back(block);
        }
        
        std::cout << "Loaded " << m_blocks.size() << " blocks from YAML: " << filepath << std::endl;
        return !m_blocks.empty();
        
    } catch (const YAML::Exception& e) {
        std::cerr << "YAML parsing error in " << filepath << ": " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error loading blocks from " << filepath << ": " << e.what() << std::endl;
        return false;
    }
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
    m_blockTypeToId.clear();
    m_isLoaded = false;
}

int AssetRegistry::getBlockIdFromType(const std::string& blockType) const {
    auto it = m_blockTypeToId.find(blockType);
    if (it != m_blockTypeToId.end()) {
        return it->second;
    }
    return -1;  // Not found
}

std::unordered_set<std::string> AssetRegistry::collectTexturePaths() const {
    std::unordered_set<std::string> texturePaths;
    
    for (const auto& material : m_materials) {
        if (material.textures.albedo.has_value()) {
            texturePaths.insert(material.textures.albedo.value());
        }
        if (material.textures.normal.has_value()) {
            texturePaths.insert(material.textures.normal.value());
        }
        if (material.textures.roughness.has_value()) {
            texturePaths.insert(material.textures.roughness.value());
        }
        if (material.textures.metallic.has_value()) {
            texturePaths.insert(material.textures.metallic.value());
        }
    }
    
    // Add minecraft character textures
    texturePaths.insert("data/textures/high_fidelity_pink_smoothie_albedo.png");
    texturePaths.insert("data/textures/high_fidelity_pink_smoothie_normal.png");
    texturePaths.insert("data/textures/high_fidelity_pink_smoothie_roughness.png");
    
    return texturePaths;
}

void AssetRegistry::initializeTextureManager() {
    std::unordered_set<std::string> texturePaths = collectTexturePaths();
    
    std::cout << "AssetRegistry: Collected " << texturePaths.size() << " texture paths from YAML materials" << std::endl;
    
    auto& textureManager = TextureManager::Get();
    textureManager.initWithMaterialPaths(texturePaths);
}

} // namespace Assets