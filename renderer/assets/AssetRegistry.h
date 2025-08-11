#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <optional>
#include "MaterialDefinition.h"

namespace Assets {

class AssetRegistry {
public:
    static AssetRegistry& Get() {
        static AssetRegistry instance;
        return instance;
    }
    
    AssetRegistry(const AssetRegistry&) = delete;
    AssetRegistry& operator=(const AssetRegistry&) = delete;
    
    // Load all asset definitions from YAML files
    bool loadFromYAML(const std::string& assetDirectory = "data/assets");
    
    // Material access
    const MaterialDefinition* getMaterial(const std::string& id) const;
    MaterialDefinition* getMaterialMutable(const std::string& id);
    const std::vector<MaterialDefinition>& getAllMaterials() const { return m_materials; }
    std::vector<MaterialDefinition>& getAllMaterialsMutable() { return m_materials; }
    
    // Model access
    const ModelDefinition* getModel(const std::string& id) const;
    ModelDefinition* getModelMutable(const std::string& id);
    const std::vector<ModelDefinition>& getAllModels() const { return m_models; }
    std::vector<ModelDefinition>& getAllModelsMutable() { return m_models; }
    
    // Block access
    const BlockDefinition* getBlock(int blockType) const;
    const BlockDefinition* getBlockById(int id) const;
    const std::vector<BlockDefinition>& getAllBlocks() const { return m_blocks; }
    
    // Utility functions
    const MaterialDefinition* getMaterialForBlock(int blockType) const;
    const ModelDefinition* getModelForBlock(int blockType) const;
    
    // Clear all registries
    void clear();
    
private:
    AssetRegistry() = default;
    ~AssetRegistry() = default;
    
    bool loadMaterials(const std::string& filepath);
    bool loadModels(const std::string& filepath);
    bool loadBlocks(const std::string& filepath);
    
    void loadHardcodedMaterials();
    void loadHardcodedModels();
    void loadHardcodedBlocks();
    
    std::vector<MaterialDefinition> m_materials;
    std::vector<ModelDefinition> m_models;
    std::vector<BlockDefinition> m_blocks;
    
    std::unordered_map<std::string, size_t> m_materialIndex;
    std::unordered_map<std::string, size_t> m_modelIndex;
    std::unordered_map<int, size_t> m_blockIndex;  // Maps block type to index
    std::unordered_map<int, size_t> m_blockIdIndex;  // Maps block ID to index
};

} // namespace Assets