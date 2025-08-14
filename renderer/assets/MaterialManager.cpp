#include "MaterialManager.h"
#include "AssetRegistry.h"
#include "TextureManager.h"
#include "util/DebugUtils.h"
#include "voxelengine/BlockType.h"
#include "BlockManager.h"
#include <iostream>
#include <algorithm>

namespace Assets {

MaterialManager::~MaterialManager() {
    cleanup();
}

bool MaterialManager::initialize() {
    std::cout << "Initializing MaterialManager..." << std::endl;
    
    // Load asset definitions from YAML
    if (!AssetRegistry::Get().loadFromYAML()) {
        std::cerr << "Failed to load asset definitions" << std::endl;
        return false;
    }
    
    // Create GPU materials from definitions
    if (!createGPUMaterials()) {
        std::cerr << "Failed to create GPU materials" << std::endl;
        return false;
    }
    
    std::cout << "MaterialManager initialized with " << m_materialCount << " materials" << std::endl;
    return true;
}

void MaterialManager::cleanup() {
    if (m_d_materials) {
        CUDA_CHECK(cudaFree(m_d_materials));
        m_d_materials = nullptr;
    }
    
    m_cpuMaterials.clear();
    m_materialIdToIndex.clear();
    m_blockTypeToMaterialIndex.clear();
    m_entityTypeToMaterialIndex.clear();
    m_dynamicMaterialMap.clear();
    m_freeMaterialSlots.clear();
    m_materialCount = 0;
    m_materialCapacity = 0;
}

bool MaterialManager::createGPUMaterials() {
    auto& registry = AssetRegistry::Get();
    auto& materials = registry.getAllMaterialsMutable();
    
    if (materials.empty()) {
        std::cerr << "No materials found in registry" << std::endl;
        return false;
    }
    
    // Reserve space for static materials plus some dynamic slots
    const unsigned int staticCount = materials.size();
    const unsigned int dynamicReserve = 64;  // Reserve space for dynamic materials
    const unsigned int totalCapacity = staticCount + dynamicReserve;
    
    if (!ensureCapacity(totalCapacity)) {
        return false;
    }
    
    // Create material parameters for each definition
    m_cpuMaterials.clear();
    m_cpuMaterials.reserve(totalCapacity);
    
    for (size_t i = 0; i < materials.size(); ++i) {
        auto& matDef = materials[i];
        
        // Create GPU material parameter
        MaterialParameter param = createMaterialParameter(matDef);
        param.materialId = i;  // Set the material ID
        
        // Store runtime index in definition
        matDef.runtimeIndex = i;
        
        // Add to CPU array
        m_cpuMaterials.push_back(param);
        
        // Update lookup maps
        m_materialIdToIndex[matDef.id] = i;
    }
    
    // Build block type to material index mapping
    const auto& blocks = registry.getAllBlocks();
    for (const auto& block : blocks) {
        if (block.material_id.has_value()) {
            auto it = m_materialIdToIndex.find(block.material_id.value());
            if (it != m_materialIdToIndex.end()) {
                m_blockTypeToMaterialIndex[block.id] = it->second;
            }
        }
    }
    
    // Build entity type to material index mapping
    // Special handling for entity materials
    const std::unordered_map<std::string, int> entityTypeMap = {
        {"minecraft_character", 0}  // EntityTypeMinecraftCharacter = 0
    };
    
    for (const auto& [matId, entityType] : entityTypeMap) {
        auto it = m_materialIdToIndex.find(matId);
        if (it != m_materialIdToIndex.end()) {
            m_entityTypeToMaterialIndex[entityType] = it->second;
        }
    }
    
    m_materialCount = m_cpuMaterials.size();
    
    // Upload to GPU
    return uploadMaterials();
}

MaterialParameter MaterialManager::createMaterialParameter(const MaterialDefinition& def) {
    MaterialParameter param{};
    
    // Set basic properties
    param.albedo = def.properties.albedo;
    param.roughness = def.properties.roughness;
    param.metallic = def.properties.metallic;
    param.uvScale = def.properties.uv_scale;
    param.translucency = def.properties.translucency;
    param.useWorldGridUV = def.properties.use_world_grid_uv;
    param.isThinfilm = def.properties.is_thinfilm;
    param.isEmissive = def.properties.is_emissive;
    
    if (param.isEmissive) {
        param.albedo = def.properties.emissive_radiance;
    }
    
    // Set texture handles using the existing TextureManager
    auto& textureManager = TextureManager::Get();
    if (def.textures.albedo.has_value()) {
        param.textureAlbedo = textureManager.GetTexture(def.textures.albedo.value());
        std::cout << "Material " << def.id << " albedo texture: " << param.textureAlbedo << " from " << def.textures.albedo.value() << std::endl;
    }
    if (def.textures.normal.has_value()) {
        param.textureNormal = textureManager.GetTexture(def.textures.normal.value());
    }
    if (def.textures.roughness.has_value()) {
        param.textureRoughness = textureManager.GetTexture(def.textures.roughness.value());
    }
    if (def.textures.metallic.has_value()) {
        param.textureMetallic = textureManager.GetTexture(def.textures.metallic.value());
    }
    
    return param;
}

bool MaterialManager::ensureCapacity(unsigned int count) {
    if (count <= m_materialCapacity) {
        return true;
    }
    
    // Free old buffer if exists
    if (m_d_materials) {
        CUDA_CHECK(cudaFree(m_d_materials));
        m_d_materials = nullptr;
    }
    
    // Allocate new buffer
    size_t bufferSize = sizeof(MaterialParameter) * count;
    CUDA_CHECK(cudaMalloc((void**)&m_d_materials, bufferSize));
    
    if (!m_d_materials) {
        std::cerr << "Failed to allocate GPU memory for materials" << std::endl;
        return false;
    }
    
    m_materialCapacity = count;
    return true;
}

bool MaterialManager::uploadMaterials() {
    if (!m_d_materials || m_cpuMaterials.empty()) {
        return false;
    }
    
    size_t uploadSize = sizeof(MaterialParameter) * m_materialCount;
    CUDA_CHECK(cudaMemcpy(m_d_materials, m_cpuMaterials.data(), uploadSize, cudaMemcpyHostToDevice));
    
    return true;
}

bool MaterialManager::updateMaterial(const std::string& materialId) {
    auto it = m_materialIdToIndex.find(materialId);
    if (it == m_materialIdToIndex.end()) {
        std::cerr << "Material not found: " << materialId << std::endl;
        return false;
    }
    
    return updateMaterial(it->second);
}

bool MaterialManager::updateMaterial(unsigned int index) {
    if (index >= m_materialCount) {
        std::cerr << "Material index out of range: " << index << std::endl;
        return false;
    }
    
    // Get the material definition from registry
    auto& registry = AssetRegistry::Get();
    const auto& materials = registry.getAllMaterials();
    
    if (index >= materials.size()) {
        // This might be a dynamic material
        return false;
    }
    
    // Recreate the material parameter
    MaterialParameter param = createMaterialParameter(materials[index]);
    param.materialId = index;
    
    // Update CPU copy
    m_cpuMaterials[index] = param;
    
    // Upload single material to GPU
    size_t offset = sizeof(MaterialParameter) * index;
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<char*>(m_d_materials) + offset,
        &param,
        sizeof(MaterialParameter),
        cudaMemcpyHostToDevice
    ));
    
    return true;
}

unsigned int MaterialManager::createDynamicMaterial(const MaterialProperties& properties) {
    unsigned int gpuIndex;
    
    // Find a free slot
    if (!m_freeMaterialSlots.empty()) {
        gpuIndex = m_freeMaterialSlots.back();
        m_freeMaterialSlots.pop_back();
    } else if (m_materialCount < m_materialCapacity) {
        gpuIndex = m_materialCount++;
    } else {
        // Need to grow the buffer
        if (!ensureCapacity(m_materialCapacity * 2)) {
            return 0;  // Failed to allocate
        }
        gpuIndex = m_materialCount++;
    }
    
    // Create material parameter
    MaterialParameter param{};
    param.albedo = properties.albedo;
    param.roughness = properties.roughness;
    param.metallic = properties.metallic;
    param.uvScale = properties.uv_scale;
    param.translucency = properties.translucency;
    param.useWorldGridUV = properties.use_world_grid_uv;
    param.isThinfilm = properties.is_thinfilm;
    param.isEmissive = properties.is_emissive;
    param.materialId = gpuIndex;
    
    if (param.isEmissive) {
        param.albedo = properties.emissive_radiance;
    }
    
    // Store in CPU array
    if (gpuIndex >= m_cpuMaterials.size()) {
        m_cpuMaterials.resize(gpuIndex + 1);
    }
    m_cpuMaterials[gpuIndex] = param;
    
    // Upload to GPU
    size_t offset = sizeof(MaterialParameter) * gpuIndex;
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<char*>(m_d_materials) + offset,
        &param,
        sizeof(MaterialParameter),
        cudaMemcpyHostToDevice
    ));
    
    // Create dynamic ID and store mapping
    unsigned int dynamicId = m_nextDynamicId++;
    m_dynamicMaterialMap[dynamicId] = gpuIndex;
    
    return dynamicId;
}

bool MaterialManager::updateDynamicMaterial(unsigned int dynamicId, const MaterialProperties& properties) {
    auto it = m_dynamicMaterialMap.find(dynamicId);
    if (it == m_dynamicMaterialMap.end()) {
        return false;
    }
    
    unsigned int gpuIndex = it->second;
    
    // Create updated material parameter
    MaterialParameter param{};
    param.albedo = properties.albedo;
    param.roughness = properties.roughness;
    param.metallic = properties.metallic;
    param.uvScale = properties.uv_scale;
    param.translucency = properties.translucency;
    param.useWorldGridUV = properties.use_world_grid_uv;
    param.isThinfilm = properties.is_thinfilm;
    param.isEmissive = properties.is_emissive;
    param.materialId = gpuIndex;
    
    if (param.isEmissive) {
        param.albedo = properties.emissive_radiance;
    }
    
    // Update CPU copy
    m_cpuMaterials[gpuIndex] = param;
    
    // Upload to GPU
    size_t offset = sizeof(MaterialParameter) * gpuIndex;
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<char*>(m_d_materials) + offset,
        &param,
        sizeof(MaterialParameter),
        cudaMemcpyHostToDevice
    ));
    
    return true;
}

void MaterialManager::destroyDynamicMaterial(unsigned int dynamicId) {
    auto it = m_dynamicMaterialMap.find(dynamicId);
    if (it == m_dynamicMaterialMap.end()) {
        return;
    }
    
    unsigned int gpuIndex = it->second;
    
    // Add slot to free list for reuse
    m_freeMaterialSlots.push_back(gpuIndex);
    
    // Remove from dynamic map
    m_dynamicMaterialMap.erase(it);
}

unsigned int MaterialManager::getMaterialIndex(const std::string& materialId) const {
    auto it = m_materialIdToIndex.find(materialId);
    if (it != m_materialIdToIndex.end()) {
        return it->second;
    }
    return 0;  // Default material
}

unsigned int MaterialManager::getMaterialIndexForBlock(int blockType) const {
    auto it = m_blockTypeToMaterialIndex.find(blockType);
    if (it != m_blockTypeToMaterialIndex.end()) {
        return it->second;
    }
    return 0;  // Default material
}

unsigned int MaterialManager::getMaterialIndexForEntity(int entityType) const {
    auto it = m_entityTypeToMaterialIndex.find(entityType);
    if (it != m_entityTypeToMaterialIndex.end()) {
        return it->second;
    }
    return 0;  // Default material
}

unsigned int MaterialManager::getMaterialIndexForObjectId(unsigned int objectId) const {
    // Convert object ID to block ID and use existing block-to-material mapping
    int blockType = static_cast<int>(Assets::BlockManager::Get().objectIdToBlockId(objectId));
    unsigned int materialIndex = getMaterialIndexForBlock(blockType);
    
    return materialIndex;
}

} // namespace Assets