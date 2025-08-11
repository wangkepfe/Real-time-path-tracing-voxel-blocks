#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include "MaterialDefinition.h"
#include "shaders/SystemParameter.h"

namespace Assets {

struct LoadedGeometry {
    VertexAttributes* d_attributes = nullptr;
    Int3* d_indices = nullptr;
    size_t attributeSize = 0;
    size_t indicesSize = 0;
    size_t vertexCount = 0;
    size_t triangleCount = 0;
    bool ownsData = true;  // Whether this geometry owns the GPU memory
};

class ModelManager {
public:
    static ModelManager& Get() {
        static ModelManager instance;
        return instance;
    }
    
    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;
    
    // Initialize and load all models
    bool initialize();
    
    // Cleanup resources
    void cleanup();
    
    // Load models from asset registry
    bool loadModels();
    
    // Get geometry for a specific model
    const LoadedGeometry* getGeometry(const std::string& modelId) const;
    LoadedGeometry* getGeometryMutable(const std::string& modelId);
    
    // Get geometry for block type
    const LoadedGeometry* getGeometryForBlock(int blockType) const;
    
    // Get geometry for entity type
    const LoadedGeometry* getGeometryForEntity(int entityType) const;
    
    // Load a specific model file
    bool loadModelFile(const std::string& modelId, const std::string& filepath, bool hasAnimation = false);
    
    // Get all loaded geometries
    const std::vector<LoadedGeometry>& getAllGeometries() const { return m_geometries; }
    
private:
    ModelManager() = default;
    ~ModelManager();
    
    // Load OBJ model
    bool loadOBJModel(LoadedGeometry& geometry, const std::string& filepath);
    
    // Load GLTF model
    bool loadGLTFModel(LoadedGeometry& geometry, const std::string& filepath, bool hasAnimation);
    
    // Storage
    std::vector<LoadedGeometry> m_geometries;
    std::unordered_map<std::string, size_t> m_modelIdToIndex;
    std::unordered_map<int, size_t> m_blockTypeToGeometryIndex;
    std::unordered_map<int, size_t> m_entityTypeToGeometryIndex;
};

} // namespace Assets