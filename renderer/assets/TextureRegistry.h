#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <memory>
#include <cuda_runtime.h>
#include "shaders/LinearMath.h"
#include "util/DebugUtils.h"

namespace Assets {

struct Texture2D {
    Texture2D() = default;
    ~Texture2D() {
        if (bufferArray != nullptr) {
            CUDA_CHECK(cudaFreeMipmappedArray(bufferArray));
            bufferArray = nullptr;
        }
    }
    
    int width{};
    int height{};
    int channel{};
    cudaChannelFormatDesc format{};
    cudaResourceDesc resourceDesc{};
    cudaTextureDesc texDesc{};
    cudaMipmappedArray_t bufferArray{};
};

class TextureRegistry {
public:
    static TextureRegistry& Get() {
        static TextureRegistry instance;
        return instance;
    }
    
    TextureRegistry(const TextureRegistry&) = delete;
    TextureRegistry& operator=(const TextureRegistry&) = delete;
    
    // Initialize texture registry
    bool initialize();
    
    // Cleanup all textures
    void cleanup();
    
    // Load textures used by materials
    bool loadMaterialTextures();
    
    // Load a single texture
    TexObj loadTexture(const std::string& filepath, bool generateMipmaps = true);
    
    // Get texture by path
    TexObj getTexture(const std::string& filepath);
    
    // Check if texture is loaded
    bool hasTexture(const std::string& filepath) const;
    
    // Get all loaded texture paths
    std::vector<std::string> getLoadedTextures() const;
    
    // Texture statistics
    size_t getTextureCount() const { return m_textures.size(); }
    size_t getTotalTextureMemory() const;
    
private:
    TextureRegistry() = default;
    ~TextureRegistry();
    
    // Internal texture loading implementation
    bool loadTextureInternal(const std::string& filepath, bool generateMipmaps);
    
    // Generate mipmaps for a texture
    bool generateMipmaps(Texture2D& texture, unsigned char* data, int width, int height, int channels);
    
    // Create texture object from Texture2D
    TexObj createTextureObject(const Texture2D& texture);
    
    // Texture storage
    std::unordered_map<std::string, size_t> m_texturePathToIndex;
    std::vector<std::unique_ptr<Texture2D>> m_textures;
    std::vector<TexObj> m_textureObjects;
    
    // Track which textures need to be loaded
    std::unordered_set<std::string> m_requiredTextures;
    
    // Default/fallback textures
    TexObj m_defaultAlbedo = 0;
    TexObj m_defaultNormal = 0;
    TexObj m_defaultRoughness = 0;
    TexObj m_defaultMetallic = 0;
    
    // Create default textures
    void createDefaultTextures();
};

} // namespace Assets