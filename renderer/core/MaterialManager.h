#pragma once

#include <string>
#include <map>
#include <vector>
#include <unordered_map>

#include "shaders/SystemParameter.h"
#include "util/TextureUtils.h"

// Forward declarations
class TextureManager;

/**
 * Material Definition - Defines a material with all its properties
 */
struct MaterialDefinition {
    std::string name;
    std::string displayName;
    
    // PBR material properties
    Float3 albedo = Float3(1.0f);
    float roughness = 0.5f;
    float metallic = 0.0f;
    float translucency = 0.0f;
    bool isThinfilm = false;
    bool isEmissive = false;
    bool useWorldGridUV = true;
    float uvScale = 1.0f;
    
    // Texture paths (relative to data directory)
    std::string textureAlbedoPath;
    std::string textureNormalPath; 
    std::string textureRoughnessPath;
    std::string textureMetallicPath;
    
    // Categories for organization
    std::string category = "default";
    
    // Constructor for simple materials
    MaterialDefinition(const std::string& materialName, const Float3& color = Float3(1.0f))
        : name(materialName), displayName(materialName), albedo(color) {}
    
    // Constructor for textured materials
    MaterialDefinition(const std::string& materialName, const std::string& textureBaseName, 
                      const std::string& textureDir = "textures/")
        : name(materialName), displayName(materialName)
    {
        textureAlbedoPath = textureDir + textureBaseName + "_albedo.png";
        textureNormalPath = textureDir + textureBaseName + "_normal.png";
        textureRoughnessPath = textureDir + textureBaseName + "_rough.png";
        textureMetallicPath = textureDir + textureBaseName + "_metal.png";
    }
};

/**
 * Material Manager - Centralized material system for the ray tracing engine
 * 
 * Features:
 * - Dynamic material registration and management
 * - Block type to material mapping
 * - Runtime material loading and texture management
 * - Material parameter caching and GPU upload
 * - Support for multiple material types (blocks, entities, instances)
 */
class MaterialManager {
public:
    static MaterialManager& Get() {
        static MaterialManager instance;
        return instance;
    }
    
    MaterialManager(const MaterialManager&) = delete;
    void operator=(const MaterialManager&) = delete;
    
    // Core management functions
    void init();
    void clear();
    void update(); // Call when materials change
    
    // Material definition management
    void registerMaterial(const MaterialDefinition& materialDef);
    void registerBlockMaterial(unsigned int blockType, const std::string& materialName);
    void registerEntityMaterial(unsigned int entityType, const std::string& materialName);
    
    // Material lookup functions
    unsigned int getMaterialIndex(const std::string& materialName) const;
    unsigned int getMaterialIndexForBlock(unsigned int blockType) const;
    unsigned int getMaterialIndexForEntity(unsigned int entityType) const;
    
    // Material parameter access
    const std::vector<MaterialParameter>& getMaterialParameters() const { return m_materialParameters; }
    const MaterialParameter* getMaterialParameter(unsigned int materialIndex) const;
    const MaterialParameter* getMaterialParameterByName(const std::string& materialName) const;
    
    // GPU resource management
    MaterialParameter* getDeviceMaterialParameters() const { return m_d_materialParameters; }
    unsigned int getMaterialCount() const { return static_cast<unsigned int>(m_materialParameters.size()); }
    
    // Debug and utility functions
    void printMaterialInfo() const;
    std::vector<std::string> getAvailableMaterials() const;
    std::vector<std::string> getMaterialsByCategory(const std::string& category) const;

private:
    MaterialManager() = default;
    
    // Internal functions
    void createBuiltinMaterials();
    void loadMaterialDefinitions();
    void uploadMaterialsToGPU();
    MaterialParameter createMaterialParameter(const MaterialDefinition& materialDef) const;
    
    // Data storage
    std::map<std::string, MaterialDefinition> m_materialDefinitions;
    std::vector<MaterialParameter> m_materialParameters;
    std::map<std::string, unsigned int> m_materialNameToIndex;
    
    // Block and entity mappings
    std::map<unsigned int, std::string> m_blockTypeToMaterial;
    std::map<unsigned int, std::string> m_entityTypeToMaterial;
    
    // GPU resources
    MaterialParameter* m_d_materialParameters = nullptr;
    
    // Default materials
    static constexpr unsigned int DEFAULT_MATERIAL_INDEX = 0;
    static constexpr unsigned int MISSING_MATERIAL_INDEX = 1;
};

// Utility functions for material system integration
namespace MaterialUtils {
    // Block type to material name mapping
    std::string getDefaultMaterialForBlock(unsigned int blockType);
    
    // Material validation
    bool validateMaterialDefinition(const MaterialDefinition& materialDef);
    
    // Texture path helpers
    std::string resolveMaterialTexturePath(const std::string& relativePath);
}