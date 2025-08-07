#include "MaterialManager.h"
#include "util/TextureUtils.h"
#include "util/DebugUtils.h"
#include "voxelengine/Block.h"
#include "core/Entity.h"

#include <iostream>
#include <filesystem>
#include <fstream>

void MaterialManager::init()
{
    printf("MATERIAL MANAGER: Initializing dynamic material system...\n");
    
    // Clear any existing data
    clear();
    
    // Create built-in materials first
    createBuiltinMaterials();
    
    // Load material definitions from configuration
    loadMaterialDefinitions();
    
    // Upload to GPU
    uploadMaterialsToGPU();
    
    printf("MATERIAL MANAGER: Initialized with %zu materials\n", m_materialParameters.size());
}

void MaterialManager::clear()
{
    if (m_d_materialParameters) {
        CUDA_CHECK(cudaFree(m_d_materialParameters));
        m_d_materialParameters = nullptr;
    }
    
    m_materialDefinitions.clear();
    m_materialParameters.clear();
    m_materialNameToIndex.clear();
    m_blockTypeToMaterial.clear();
    m_entityTypeToMaterial.clear();
}

void MaterialManager::createBuiltinMaterials()
{
    TextureManager& textureManager = TextureManager::Get();
    
    // Default material (always index 0)
    MaterialDefinition defaultMat("default", Float3(0.8f, 0.8f, 0.8f));
    defaultMat.displayName = "Default Material";
    defaultMat.category = "builtin";
    defaultMat.roughness = 0.5f;
    registerMaterial(defaultMat);
    
    // Missing material (always index 1) - bright magenta for missing textures
    MaterialDefinition missingMat("missing", Float3(1.0f, 0.0f, 1.0f));
    missingMat.displayName = "Missing Material";
    missingMat.category = "builtin";
    missingMat.roughness = 1.0f;
    registerMaterial(missingMat);
    
    // Block materials based on texture files
    const auto& textureFiles = GetTextureFiles();
    for (size_t i = 0; i < textureFiles.size(); ++i) {
        const std::string& textureFile = textureFiles[i];
        
        MaterialDefinition blockMat(textureFile, textureFile, "textures/");
        blockMat.displayName = textureFile; // Can be made more user-friendly
        blockMat.category = "blocks";
        blockMat.useWorldGridUV = true;
        blockMat.uvScale = 2.5f;
        
        registerMaterial(blockMat);
        
        // Map block types to materials (this maintains compatibility with existing system)
        unsigned int blockType = i + 1; // Block types start at 1
        if (blockType < BlockTypeNum) {
            registerBlockMaterial(blockType, textureFile);
        }
    }
    
    // Special materials
    
    // Test material
    MaterialDefinition testMat("test", Float3(1.0f));
    testMat.displayName = "Test Material";
    testMat.category = "special";
    testMat.roughness = 0.0f;
    testMat.translucency = 0.0f;
    registerMaterial(testMat);
    
    // Leaf material
    MaterialDefinition leafMat("leaf");
    leafMat.displayName = "Leaf Material";
    leafMat.category = "special";
    leafMat.textureAlbedoPath = "data/GreenLeaf10_4K_back_albedo.png";
    leafMat.textureNormalPath = "data/GreenLeaf10_4K_back_normal.png";
    leafMat.textureRoughnessPath = "data/GreenLeaf10_4K_back_rough.png";
    leafMat.translucency = 0.5f;
    leafMat.isThinfilm = true;
    leafMat.useWorldGridUV = false;
    leafMat.uvScale = 1.0f;
    registerMaterial(leafMat);
    
    // Metal material
    MaterialDefinition metalMat("metal", "beaten-up-metal1", "data/");
    metalMat.displayName = "Beaten Metal";
    metalMat.category = "special";
    metalMat.useWorldGridUV = true;
    metalMat.uvScale = 1.0f;
    registerMaterial(metalMat);
    
    // Emissive light material
    MaterialDefinition lightMat("light", GetEmissiveRadiance(BlockTypeTestLight));
    lightMat.displayName = "Emissive Light";
    lightMat.category = "special";
    lightMat.isEmissive = true;
    registerMaterial(lightMat);
    
    // Entity materials
    
    // Minecraft character material
    MaterialDefinition characterMat("character", "high_fidelity_pink_smoothie", "textures/");
    characterMat.displayName = "Pink Smoothie Character";
    characterMat.category = "entities";
    characterMat.albedo = Float3(1.0f, 1.0f, 1.0f);
    characterMat.roughness = 1.0f;
    characterMat.metallic = 0.0f;
    characterMat.useWorldGridUV = false;
    characterMat.uvScale = 1.0f;
    registerMaterial(characterMat);
    
    // Register entity material mapping
    registerEntityMaterial(EntityTypeMinecraftCharacter, "character");
    
    printf("MATERIAL MANAGER: Created %zu built-in materials\n", m_materialDefinitions.size());
}

void MaterialManager::loadMaterialDefinitions()
{
    // TODO: Load materials from YAML/JSON configuration files
    // This would allow artists to define materials without recompiling
    
    std::string materialConfigPath = "data/materials/materials.yaml";
    if (std::filesystem::exists(materialConfigPath)) {
        printf("MATERIAL MANAGER: Loading materials from %s\n", materialConfigPath.c_str());
        // Implementation would parse YAML and call registerMaterial() for each definition
    }
}

void MaterialManager::registerMaterial(const MaterialDefinition& materialDef)
{
    if (MaterialUtils::validateMaterialDefinition(materialDef)) {
        // Store the definition
        m_materialDefinitions.emplace(materialDef.name, materialDef);
        
        // Create GPU material parameter
        MaterialParameter materialParam = createMaterialParameter(materialDef);
        
        // Add to arrays
        unsigned int materialIndex = static_cast<unsigned int>(m_materialParameters.size());
        m_materialParameters.push_back(materialParam);
        m_materialNameToIndex[materialDef.name] = materialIndex;
        
        printf("MATERIAL MANAGER: Registered material '%s' at index %u\n", 
               materialDef.name.c_str(), materialIndex);
    }
    else {
        printf("MATERIAL MANAGER ERROR: Failed to validate material '%s'\n", materialDef.name.c_str());
    }
}

void MaterialManager::registerBlockMaterial(unsigned int blockType, const std::string& materialName)
{
    m_blockTypeToMaterial[blockType] = materialName;
    printf("MATERIAL MANAGER: Mapped block type %u to material '%s'\n", blockType, materialName.c_str());
}

void MaterialManager::registerEntityMaterial(unsigned int entityType, const std::string& materialName)
{
    m_entityTypeToMaterial[entityType] = materialName;
    printf("MATERIAL MANAGER: Mapped entity type %u to material '%s'\n", entityType, materialName.c_str());
}

MaterialParameter MaterialManager::createMaterialParameter(const MaterialDefinition& materialDef) const
{
    MaterialParameter param = {};
    TextureManager& textureManager = TextureManager::Get();
    
    // Basic properties
    param.albedo = materialDef.albedo;
    param.roughness = materialDef.roughness;
    param.metallic = materialDef.metallic;
    param.translucency = materialDef.translucency;
    param.isThinfilm = materialDef.isThinfilm;
    param.isEmissive = materialDef.isEmissive;
    param.useWorldGridUV = materialDef.useWorldGridUV;
    param.uvScale = materialDef.uvScale;
    
    // Load textures
    if (!materialDef.textureAlbedoPath.empty()) {
        std::string fullPath = MaterialUtils::resolveMaterialTexturePath(materialDef.textureAlbedoPath);
        param.textureAlbedo = textureManager.GetTexture(fullPath);
    }
    
    if (!materialDef.textureNormalPath.empty()) {
        std::string fullPath = MaterialUtils::resolveMaterialTexturePath(materialDef.textureNormalPath);
        param.textureNormal = textureManager.GetTexture(fullPath);
    }
    
    if (!materialDef.textureRoughnessPath.empty()) {
        std::string fullPath = MaterialUtils::resolveMaterialTexturePath(materialDef.textureRoughnessPath);
        param.textureRoughness = textureManager.GetTexture(fullPath);
    }
    
    if (!materialDef.textureMetallicPath.empty()) {
        std::string fullPath = MaterialUtils::resolveMaterialTexturePath(materialDef.textureMetallicPath);
        param.textureMetallic = textureManager.GetTexture(fullPath);
    }
    
    return param;
}

unsigned int MaterialManager::getMaterialIndex(const std::string& materialName) const
{
    auto it = m_materialNameToIndex.find(materialName);
    if (it != m_materialNameToIndex.end()) {
        return it->second;
    }
    
    printf("MATERIAL MANAGER WARNING: Material '%s' not found, using default\n", materialName.c_str());
    return DEFAULT_MATERIAL_INDEX;
}

unsigned int MaterialManager::getMaterialIndexForBlock(unsigned int blockType) const
{
    auto it = m_blockTypeToMaterial.find(blockType);
    if (it != m_blockTypeToMaterial.end()) {
        return getMaterialIndex(it->second);
    }
    
    // Fall back to default material for block type
    std::string defaultMaterial = MaterialUtils::getDefaultMaterialForBlock(blockType);
    if (!defaultMaterial.empty()) {
        return getMaterialIndex(defaultMaterial);
    }
    
    return DEFAULT_MATERIAL_INDEX;
}

unsigned int MaterialManager::getMaterialIndexForEntity(unsigned int entityType) const
{
    auto it = m_entityTypeToMaterial.find(entityType);
    if (it != m_entityTypeToMaterial.end()) {
        return getMaterialIndex(it->second);
    }
    
    return DEFAULT_MATERIAL_INDEX;
}

void MaterialManager::uploadMaterialsToGPU()
{
    if (m_materialParameters.empty()) {
        printf("MATERIAL MANAGER WARNING: No materials to upload to GPU\n");
        return;
    }
    
    // Free existing GPU memory
    if (m_d_materialParameters) {
        CUDA_CHECK(cudaFree(m_d_materialParameters));
    }
    
    // Allocate and upload
    size_t materialDataSize = sizeof(MaterialParameter) * m_materialParameters.size();
    CUDA_CHECK(cudaMalloc((void**)&m_d_materialParameters, materialDataSize));
    CUDA_CHECK(cudaMemcpy(m_d_materialParameters, m_materialParameters.data(), 
                         materialDataSize, cudaMemcpyHostToDevice));
    
    printf("MATERIAL MANAGER: Uploaded %zu materials to GPU (%zu bytes)\n", 
           m_materialParameters.size(), materialDataSize);
}

void MaterialManager::update()
{
    // Re-upload materials to GPU if they have changed
    uploadMaterialsToGPU();
}

void MaterialManager::printMaterialInfo() const
{
    printf("\\n=== MATERIAL MANAGER INFO ===\\n");
    printf("Total materials: %zu\\n", m_materialParameters.size());
    
    for (const auto& [name, def] : m_materialDefinitions) {
        unsigned int index = getMaterialIndex(name);
        printf("[%2u] %s (%s) - Category: %s\\n", 
               index, def.displayName.c_str(), name.c_str(), def.category.c_str());
    }
    
    printf("\\nBlock type mappings:\\n");
    for (const auto& [blockType, materialName] : m_blockTypeToMaterial) {
        printf("  Block %u -> %s\\n", blockType, materialName.c_str());
    }
    
    printf("\\nEntity type mappings:\\n");
    for (const auto& [entityType, materialName] : m_entityTypeToMaterial) {
        printf("  Entity %u -> %s\\n", entityType, materialName.c_str());
    }
    printf("===========================\\n\\n");
}

// Utility function implementations
namespace MaterialUtils {
    std::string getDefaultMaterialForBlock(unsigned int blockType)
    {
        // Provide sensible defaults based on block type
        if (blockType >= 1 && blockType < BlockTypeNum) {
            const auto& textureFiles = GetTextureFiles();
            size_t textureIndex = blockType - 1;
            if (textureIndex < textureFiles.size()) {
                return textureFiles[textureIndex];
            }
        }
        
        return "default";
    }
    
    bool validateMaterialDefinition(const MaterialDefinition& materialDef)
    {
        if (materialDef.name.empty()) {
            printf("MATERIAL VALIDATION ERROR: Material name cannot be empty\\n");
            return false;
        }
        
        if (materialDef.uvScale <= 0.0f) {
            printf("MATERIAL VALIDATION ERROR: UV scale must be positive (material: %s)\\n", 
                   materialDef.name.c_str());
            return false;
        }
        
        // Validate texture paths exist if specified
        if (!materialDef.textureAlbedoPath.empty()) {
            std::string fullPath = resolveMaterialTexturePath(materialDef.textureAlbedoPath);
            if (!std::filesystem::exists(fullPath)) {
                printf("MATERIAL VALIDATION WARNING: Albedo texture not found: %s (material: %s)\\n", 
                       fullPath.c_str(), materialDef.name.c_str());
                // Don't fail validation, just warn
            }
        }
        
        return true;
    }
    
    std::string resolveMaterialTexturePath(const std::string& relativePath)
    {
        // Handle different path formats
        if (relativePath.find("data/") == 0) {
            return relativePath; // Already has data/ prefix
        }
        else if (relativePath.find("textures/") == 0) {
            return "data/" + relativePath; // Add data/ prefix
        }
        else {
            return "data/textures/" + relativePath; // Assume textures directory
        }
    }
}