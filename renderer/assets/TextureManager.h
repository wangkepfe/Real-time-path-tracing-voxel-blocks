#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>

#include "shaders/LinearMath.h"
#include "util/DebugUtils.h"

namespace Assets {

static constexpr int MAX_NUM_TEXTURES = 64;

struct Texture2D
{
    Texture2D() = default;
    ~Texture2D()
    {
        if (bufferArray != nullptr)
        {
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

struct ShaderTexture
{
    TexObj texObjs[MAX_NUM_TEXTURES]{};
};

class TextureManager
{
public:
    static TextureManager &Get()
    {
        static TextureManager instance;
        return instance;
    }
    TextureManager(TextureManager const &) = delete;
    void operator=(TextureManager const &) = delete;
    ~TextureManager()
    {
        for (auto &texObj : m_shaderTextures.texObjs)
        {
            if (texObj != 0)
            {
                cudaDestroyTextureObject(texObj);
            }
        }
    }

    // void init(); // Legacy method - replaced by initWithMaterialPaths()
    
    // Initialize with material texture paths from YAML
    void initWithMaterialPaths(const std::unordered_set<std::string>& texturePaths);
    
    // Internal method to load textures from a list of file paths
    void LoadTexturesFromPaths(const std::vector<std::string>& filePaths);
    
    TexObj GetTexture(const std::string &name)
    {
        assert(textureNameIdLookup.find(name) != textureNameIdLookup.end());
        int id = textureNameIdLookup[name];
        assert(id >= 0 && id < MAX_NUM_TEXTURES);
        return m_shaderTextures.texObjs[id];
    }
    
    // Get texture by full path
    TexObj GetTextureByPath(const std::string& filepath);
    
    // Load a specific texture file
    bool LoadTexture(const std::string& filepath);

private:
    TextureManager() {}

    ShaderTexture m_shaderTextures{};
    std::vector<Texture2D> m_textures{};
    std::unordered_map<std::string, int> textureNameIdLookup;
    std::unordered_map<std::string, int> texturePathIdLookup;  // Full path to texture ID lookup
};

} // namespace Assets