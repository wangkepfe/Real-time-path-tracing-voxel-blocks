#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <cuda_runtime.h>
#include "MaterialDefinition.h"
#include "shaders/SystemParameter.h"

namespace Assets {

class MaterialManager {
public:
    static MaterialManager& Get() {
        static MaterialManager instance;
        return instance;
    }
    
    MaterialManager(const MaterialManager&) = delete;
    MaterialManager& operator=(const MaterialManager&) = delete;
    
    // Initialize the material manager and load all materials
    bool initialize();
    
    // Cleanup resources
    void cleanup();
    
    // Create GPU materials from asset definitions
    bool createGPUMaterials();
    
    // Update a specific material on the GPU
    bool updateMaterial(const std::string& materialId);
    bool updateMaterial(unsigned int index);
    
    // Dynamic material management
    unsigned int createDynamicMaterial(const MaterialProperties& properties);
    bool updateDynamicMaterial(unsigned int dynamicId, const MaterialProperties& properties);
    void destroyDynamicMaterial(unsigned int dynamicId);
    
    // Get material data
    MaterialParameter* getGPUMaterialsPointer() const { return m_d_materials; }
    unsigned int getMaterialCount() const { return m_materialCount; }
    
    // Get material index by various lookups
    unsigned int getMaterialIndex(const std::string& materialId) const;
    unsigned int getMaterialIndexForBlock(int blockType) const;
    unsigned int getMaterialIndexForEntity(int entityType) const;
    unsigned int getMaterialIndexForObjectId(unsigned int objectId) const;
    
    
private:
    MaterialManager() = default;
    ~MaterialManager();
    
    // Convert material definition to GPU material parameter
    MaterialParameter createMaterialParameter(const MaterialDefinition& def);
    
    // GPU material data
    MaterialParameter* m_d_materials = nullptr;
    std::vector<MaterialParameter> m_cpuMaterials;
    unsigned int m_materialCount = 0;
    unsigned int m_materialCapacity = 0;
    
    // Dynamic material management
    std::unordered_map<unsigned int, unsigned int> m_dynamicMaterialMap;  // Dynamic ID -> GPU index
    std::vector<unsigned int> m_freeMaterialSlots;
    unsigned int m_nextDynamicId = 10000;  // Start dynamic IDs at high number
    
    // Lookups
    std::unordered_map<std::string, unsigned int> m_materialIdToIndex;
    std::unordered_map<int, unsigned int> m_blockTypeToMaterialIndex;
    std::unordered_map<int, unsigned int> m_entityTypeToMaterialIndex;
    
    
    // Ensure we have capacity for N materials
    bool ensureCapacity(unsigned int count);
    
    // Upload materials to GPU
    bool uploadMaterials();
};

} // namespace Assets