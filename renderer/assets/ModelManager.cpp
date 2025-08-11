#include "ModelManager.h"
#include "AssetRegistry.h"
#include "util/ModelUtils.h"
#include "util/GLTFUtils.h"
#include "util/DebugUtils.h"
#include "voxelengine/Block.h"
#include <iostream>
#include <cuda_runtime.h>

namespace Assets {

ModelManager::~ModelManager() {
    cleanup();
}

bool ModelManager::initialize() {
    std::cout << "Initializing ModelManager..." << std::endl;
    
    // Load all models from registry
    if (!loadModels()) {
        std::cerr << "Failed to load models" << std::endl;
        return false;
    }
    
    std::cout << "ModelManager initialized with " << m_geometries.size() << " models" << std::endl;
    return true;
}

void ModelManager::cleanup() {
    for (auto& geometry : m_geometries) {
        if (geometry.ownsData) {
            if (geometry.d_attributes) {
                CUDA_CHECK(cudaFree(geometry.d_attributes));
            }
            if (geometry.d_indices) {
                CUDA_CHECK(cudaFree(geometry.d_indices));
            }
        }
    }
    
    m_geometries.clear();
    m_modelIdToIndex.clear();
    m_blockTypeToGeometryIndex.clear();
    m_entityTypeToGeometryIndex.clear();
}

bool ModelManager::loadModels() {
    auto& registry = AssetRegistry::Get();
    auto& models = registry.getAllModelsMutable();
    
    // Map string block/entity types to enum values
    const std::unordered_map<std::string, int> blockTypeMap = {
        {"BlockTypeTest1", BlockTypeTest1},
        {"BlockTypeLeaves", BlockTypeLeaves},
        {"BlockTypeTestLightBase", BlockTypeTestLightBase},
        {"BlockTypeTestLight", BlockTypeTestLight}
    };
    
    const std::unordered_map<std::string, int> entityTypeMap = {
        {"EntityTypeMinecraftCharacter", 0}  // EntityTypeMinecraftCharacter
    };
    
    for (auto& modelDef : models) {
        // Resolve block/entity type enums
        auto blockIt = blockTypeMap.find(modelDef.block_type == -1 ? "" : std::to_string(modelDef.block_type));
        if (modelDef.type == "instanced" && modelDef.block_type == -1) {
            // Try to find in the YAML-defined block_type string
            for (const auto& [typeStr, typeEnum] : blockTypeMap) {
                if (modelDef.file.find(typeStr) != std::string::npos || 
                    modelDef.id.find(typeStr) != std::string::npos) {
                    modelDef.block_type = typeEnum;
                    break;
                }
            }
        }
        
        if (modelDef.type == "entity" && modelDef.entity_type == -1) {
            // Try to find in the YAML-defined entity_type string
            for (const auto& [typeStr, typeEnum] : entityTypeMap) {
                if (modelDef.file.find(typeStr) != std::string::npos ||
                    modelDef.id.find(typeStr) != std::string::npos) {
                    modelDef.entity_type = typeEnum;
                    break;
                }
            }
        }
        
        // Load the model file
        if (!loadModelFile(modelDef.id, "data/" + modelDef.file, modelDef.has_animation)) {
            std::cerr << "Failed to load model: " << modelDef.id << " from " << modelDef.file << std::endl;
            continue;
        }
        
        // Update mappings
        size_t geometryIndex = m_modelIdToIndex[modelDef.id];
        
        if (modelDef.block_type >= 0) {
            m_blockTypeToGeometryIndex[modelDef.block_type] = geometryIndex;
        }
        
        if (modelDef.entity_type >= 0) {
            m_entityTypeToGeometryIndex[modelDef.entity_type] = geometryIndex;
        }
        
        // Store runtime geometry pointer
        modelDef.runtimeGeometry = &m_geometries[geometryIndex];
    }
    
    return !m_geometries.empty();
}

bool ModelManager::loadModelFile(const std::string& modelId, const std::string& filepath, bool hasAnimation) {
    LoadedGeometry geometry;
    
    bool success = false;
    
    // Determine file type and load accordingly
    if (filepath.find(".obj") != std::string::npos || filepath.find(".OBJ") != std::string::npos) {
        success = loadOBJModel(geometry, filepath);
    } else if (filepath.find(".gltf") != std::string::npos || filepath.find(".glb") != std::string::npos) {
        success = loadGLTFModel(geometry, filepath, hasAnimation);
    } else {
        std::cerr << "Unsupported model format: " << filepath << std::endl;
        return false;
    }
    
    if (!success) {
        return false;
    }
    
    // Store the geometry
    size_t index = m_geometries.size();
    m_geometries.push_back(std::move(geometry));
    m_modelIdToIndex[modelId] = index;
    
    return true;
}

bool ModelManager::loadOBJModel(LoadedGeometry& geometry, const std::string& filepath) {
    // Use the loadModel function from ModelUtils
    unsigned int* d_tempIndices = nullptr;
    unsigned int attrSize = 0;
    unsigned int indicesSize = 0;
    
    loadModel(&geometry.d_attributes, &d_tempIndices, attrSize, indicesSize, filepath);
    
    if (!geometry.d_attributes || !d_tempIndices || attrSize == 0 || indicesSize == 0) {
        if (geometry.d_attributes) CUDA_CHECK(cudaFree(geometry.d_attributes));
        if (d_tempIndices) CUDA_CHECK(cudaFree(d_tempIndices));
        return false;
    }
    
    // Convert unsigned int indices to Int3 format
    // OBJ files use triangle indices, so indicesSize should be divisible by 3
    size_t numTriangles = indicesSize / 3;
    size_t indexBufferSize = numTriangles * sizeof(Int3);
    
    CUDA_CHECK(cudaMalloc((void**)&geometry.d_indices, indexBufferSize));
    
    // Copy and convert indices from unsigned int to Int3
    // We need to do this on the CPU side
    std::vector<unsigned int> tempIndices(indicesSize);
    std::vector<Int3> int3Indices(numTriangles);
    
    CUDA_CHECK(cudaMemcpy(tempIndices.data(), d_tempIndices, indicesSize * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    for (size_t i = 0; i < numTriangles; ++i) {
        int3Indices[i].x = tempIndices[i * 3 + 0];
        int3Indices[i].y = tempIndices[i * 3 + 1];
        int3Indices[i].z = tempIndices[i * 3 + 2];
    }
    
    CUDA_CHECK(cudaMemcpy(geometry.d_indices, int3Indices.data(), indexBufferSize, cudaMemcpyHostToDevice));
    
    // Free the temporary indices
    CUDA_CHECK(cudaFree(d_tempIndices));
    
    geometry.attributeSize = attrSize;
    geometry.indicesSize = numTriangles;
    geometry.vertexCount = attrSize;
    geometry.triangleCount = numTriangles;
    geometry.ownsData = true;
    
    return true;
}

bool ModelManager::loadGLTFModel(LoadedGeometry& geometry, const std::string& filepath, bool hasAnimation) {
    if (hasAnimation) {
        // For animated models, we'll need special handling
        // This will be integrated with the animation system
        // For now, just load the base geometry
        
        // Placeholder - the actual implementation would need skeleton and animation data
        std::cerr << "Animated GLTF loading not yet integrated with new system" << std::endl;
        return false;
    } else {
        // Load static GLTF model
        unsigned int* d_tempIndices = nullptr;
        unsigned int attrSize = 0;
        unsigned int indicesSize = 0;
        
        if (!GLTFUtils::loadGLTFModel(&geometry.d_attributes, &d_tempIndices, attrSize, indicesSize, filepath)) {
            return false;
        }
        
        // Convert unsigned int indices to Int3 format
        size_t numTriangles = indicesSize / 3;
        size_t indexBufferSize = numTriangles * sizeof(Int3);
        
        CUDA_CHECK(cudaMalloc((void**)&geometry.d_indices, indexBufferSize));
        
        // Copy and convert indices
        std::vector<unsigned int> tempIndices(indicesSize);
        std::vector<Int3> int3Indices(numTriangles);
        
        CUDA_CHECK(cudaMemcpy(tempIndices.data(), d_tempIndices, indicesSize * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        
        for (size_t i = 0; i < numTriangles; ++i) {
            int3Indices[i].x = tempIndices[i * 3 + 0];
            int3Indices[i].y = tempIndices[i * 3 + 1];
            int3Indices[i].z = tempIndices[i * 3 + 2];
        }
        
        CUDA_CHECK(cudaMemcpy(geometry.d_indices, int3Indices.data(), indexBufferSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaFree(d_tempIndices));
        
        geometry.attributeSize = attrSize;
        geometry.indicesSize = numTriangles;
        geometry.vertexCount = attrSize;
        geometry.triangleCount = numTriangles;
        geometry.ownsData = true;
        
        return true;
    }
}

const LoadedGeometry* ModelManager::getGeometry(const std::string& modelId) const {
    auto it = m_modelIdToIndex.find(modelId);
    if (it != m_modelIdToIndex.end()) {
        return &m_geometries[it->second];
    }
    return nullptr;
}

LoadedGeometry* ModelManager::getGeometryMutable(const std::string& modelId) {
    auto it = m_modelIdToIndex.find(modelId);
    if (it != m_modelIdToIndex.end()) {
        return &m_geometries[it->second];
    }
    return nullptr;
}

const LoadedGeometry* ModelManager::getGeometryForBlock(int blockType) const {
    auto it = m_blockTypeToGeometryIndex.find(blockType);
    if (it != m_blockTypeToGeometryIndex.end()) {
        return &m_geometries[it->second];
    }
    return nullptr;
}

const LoadedGeometry* ModelManager::getGeometryForEntity(int entityType) const {
    auto it = m_entityTypeToGeometryIndex.find(entityType);
    if (it != m_entityTypeToGeometryIndex.end()) {
        return &m_geometries[it->second];
    }
    return nullptr;
}

} // namespace Assets