#include "TextureRegistry.h"
#include "AssetRegistry.h"
#include "util/Timer.h"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <unordered_set>
#include <algorithm>

#include "ext/stb/stb_image.h"

namespace Assets {

TextureRegistry::~TextureRegistry() {
    cleanup();
}

bool TextureRegistry::initialize() {
    std::cout << "Initializing TextureRegistry..." << std::endl;
    
    // Create default textures first
    createDefaultTextures();
    
    // Load all textures required by materials
    if (!loadMaterialTextures()) {
        std::cerr << "Failed to load material textures" << std::endl;
        return false;
    }
    
    std::cout << "TextureRegistry initialized with " << m_textures.size() << " textures" << std::endl;
    return true;
}

void TextureRegistry::cleanup() {
    // Destroy texture objects
    for (auto& texObj : m_textureObjects) {
        if (texObj != 0) {
            cudaDestroyTextureObject(texObj);
        }
    }
    m_textureObjects.clear();
    
    // Textures are automatically cleaned up by unique_ptr destructors
    m_textures.clear();
    m_texturePathToIndex.clear();
    m_requiredTextures.clear();
}

bool TextureRegistry::loadMaterialTextures() {
    auto& registry = AssetRegistry::Get();
    const auto& materials = registry.getAllMaterials();
    
    // Collect all unique texture paths
    std::unordered_set<std::string> texturePaths;
    
    for (const auto& mat : materials) {
        if (mat.textures.albedo.has_value()) {
            texturePaths.insert(mat.textures.albedo.value());
        }
        if (mat.textures.normal.has_value()) {
            texturePaths.insert(mat.textures.normal.value());
        }
        if (mat.textures.roughness.has_value()) {
            texturePaths.insert(mat.textures.roughness.value());
        }
        if (mat.textures.metallic.has_value()) {
            texturePaths.insert(mat.textures.metallic.value());
        }
        if (mat.textures.emissive.has_value()) {
            texturePaths.insert(mat.textures.emissive.value());
        }
    }
    
    // Load each unique texture
    size_t loadedCount = 0;
    {
        ScopeTimer timer("TextureLoading");
        
        for (const auto& path : texturePaths) {
            if (!std::filesystem::exists(path)) {
                std::cerr << "Texture file not found: " << path << std::endl;
                continue;
            }
            
            if (loadTextureInternal(path, true)) {
                loadedCount++;
            } else {
                std::cerr << "Failed to load texture: " << path << std::endl;
            }
        }
        
        std::cout << "Loaded " << loadedCount << " textures" << std::endl;
    }
    
    return loadedCount > 0;
}

TexObj TextureRegistry::loadTexture(const std::string& filepath, bool generateMipmaps) {
    // Check if already loaded
    auto it = m_texturePathToIndex.find(filepath);
    if (it != m_texturePathToIndex.end()) {
        return m_textureObjects[it->second];
    }
    
    // Load the texture
    if (!loadTextureInternal(filepath, generateMipmaps)) {
        return 0;
    }
    
    // Return the newly loaded texture
    it = m_texturePathToIndex.find(filepath);
    if (it != m_texturePathToIndex.end()) {
        return m_textureObjects[it->second];
    }
    
    return 0;
}

bool TextureRegistry::loadTextureInternal(const std::string& filepath, bool generateMipmaps) {
    // Check if already loaded
    if (m_texturePathToIndex.find(filepath) != m_texturePathToIndex.end()) {
        return true;
    }
    
    // Load image from file
    int width, height, channels;
    unsigned char* data = stbi_load(filepath.c_str(), &width, &height, &channels, 0);
    
    if (!data) {
        std::cerr << "Failed to load image: " << filepath << std::endl;
        return false;
    }
    
    // Create texture object
    auto texture = std::make_unique<Texture2D>();
    texture->width = width;
    texture->height = height;
    texture->channel = channels;
    
    // Setup CUDA texture
    bool success = false;
    if (generateMipmaps && (width == height) && ((width & (width - 1)) == 0)) {
        // Power of two texture - generate mipmaps
        success = this->generateMipmaps(*texture, data, width, height, channels);
    } else {
        // Non-power-of-two or non-square - single level texture
        cudaChannelFormatDesc channelDesc;
        if (channels == 1) {
            channelDesc = cudaCreateChannelDesc<unsigned char>();
        } else if (channels == 2) {
            channelDesc = cudaCreateChannelDesc<uchar2>();
        } else if (channels == 3) {
            channelDesc = cudaCreateChannelDesc<uchar3>();
        } else {
            channelDesc = cudaCreateChannelDesc<uchar4>();
        }
        
        cudaArray_t cuArray;
        CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));
        
        size_t pitch = width * channels * sizeof(unsigned char);
        CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, data, pitch, pitch, height, cudaMemcpyHostToDevice));
        
        // Create texture object
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;
        
        cudaTextureDesc texDesc{};
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;
        
        texture->resourceDesc = resDesc;
        texture->texDesc = texDesc;
        texture->format = channelDesc;
        
        success = true;
    }
    
    stbi_image_free(data);
    
    if (!success) {
        return false;
    }
    
    // Store texture first (before creating texture object)
    size_t index = m_textures.size();
    m_textures.push_back(std::move(texture));
    
    // Create texture object from the stored texture
    TexObj texObj = createTextureObject(*m_textures.back());
    if (texObj == 0) {
        // Remove the texture we just added on failure
        m_textures.pop_back();
        return false;
    }
    
    // Debug: Print texture object handle
    std::cout << "Created texture object for " << filepath << ": handle=" << texObj << std::endl;
    
    // Store texture object and path mapping
    m_textureObjects.push_back(texObj);
    m_texturePathToIndex[filepath] = index;
    
    return true;
}

bool TextureRegistry::generateMipmaps(Texture2D& texture, unsigned char* data, int width, int height, int channels) {
    // Calculate number of mip levels
    int numLevels = 0;
    int size = width;
    while (size > 0) {
        numLevels++;
        size >>= 1;
    }
    
    // Setup channel format
    // Store original channel count before modification
    int originalChannels = channels;
    
    cudaChannelFormatDesc channelDesc;
    if (channels == 1) {
        channelDesc = cudaCreateChannelDesc<unsigned char>();
    } else if (channels == 2) {
        channelDesc = cudaCreateChannelDesc<uchar2>();
    } else if (channels == 3) {
        // Expand to 4 channels for better GPU alignment
        channelDesc = cudaCreateChannelDesc<uchar4>();
        channels = 4;
    } else {
        channelDesc = cudaCreateChannelDesc<uchar4>();
    }
    
    // Create mipmapped array
    cudaExtent extent = make_cudaExtent(width, height, 0);
    CUDA_CHECK(cudaMallocMipmappedArray(&texture.bufferArray, &channelDesc, extent, numLevels));
    
    // Fill first mip level
    cudaArray_t level0;
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&level0, texture.bufferArray, 0));
    
    // Prepare data for upload (expand to 4 channels if needed)
    std::vector<unsigned char> expandedData;
    unsigned char* uploadData = data;
    
    if (originalChannels == 3 && channels == 4) {  // We expanded from 3 to 4 channels
        expandedData.resize(width * height * 4);
        for (int i = 0; i < width * height; ++i) {
            expandedData[i * 4 + 0] = data[i * 3 + 0];
            expandedData[i * 4 + 1] = data[i * 3 + 1];
            expandedData[i * 4 + 2] = data[i * 3 + 2];
            expandedData[i * 4 + 3] = 255;
        }
        uploadData = expandedData.data();
        
        // Debug: Print first pixel
        if (width > 0 && height > 0) {
            std::cout << "First pixel after expansion: R=" << (int)uploadData[0] 
                      << " G=" << (int)uploadData[1] 
                      << " B=" << (int)uploadData[2] 
                      << " A=" << (int)uploadData[3] << std::endl;
        }
    }
    
    size_t pitch = width * channels * sizeof(unsigned char);
    CUDA_CHECK(cudaMemcpy2DToArray(level0, 0, 0, uploadData, pitch, pitch, height, cudaMemcpyHostToDevice));
    
    // Ensure data is uploaded before continuing
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Generate remaining mip levels (simple box filter)
    int currentWidth = width;
    int currentHeight = height;
    
    for (int level = 1; level < numLevels; ++level) {
        int prevWidth = currentWidth;
        int prevHeight = currentHeight;
        currentWidth >>= 1;
        currentHeight >>= 1;
        
        if (currentWidth == 0) currentWidth = 1;
        if (currentHeight == 0) currentHeight = 1;
        
        // Allocate temporary buffer for this level
        std::vector<unsigned char> mipData(currentWidth * currentHeight * channels);
        
        // Simple box filter downsampling
        for (int y = 0; y < currentHeight; ++y) {
            for (int x = 0; x < currentWidth; ++x) {
                for (int c = 0; c < channels; ++c) {
                    int sum = 0;
                    int count = 0;
                    
                    // Sample 2x2 region from previous level
                    for (int dy = 0; dy < 2; ++dy) {
                        for (int dx = 0; dx < 2; ++dx) {
                            int sx = std::min(x * 2 + dx, prevWidth - 1);
                            int sy = std::min(y * 2 + dy, prevHeight - 1);
                            
                            if (level == 1) {
                                // Sample from original data
                                sum += uploadData[(sy * prevWidth + sx) * channels + c];
                            }
                            count++;
                        }
                    }
                    
                    mipData[(y * currentWidth + x) * channels + c] = sum / count;
                }
            }
        }
        
        // Upload mip level
        cudaArray_t levelArray;
        CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, texture.bufferArray, level));
        
        size_t mipPitch = currentWidth * channels * sizeof(unsigned char);
        CUDA_CHECK(cudaMemcpy2DToArray(levelArray, 0, 0, mipData.data(), mipPitch, mipPitch, currentHeight, cudaMemcpyHostToDevice));
    }
    
    // Setup texture description
    texture.resourceDesc.resType = cudaResourceTypeMipmappedArray;
    texture.resourceDesc.res.mipmap.mipmap = texture.bufferArray;
    
    texture.texDesc.addressMode[0] = cudaAddressModeWrap;
    texture.texDesc.addressMode[1] = cudaAddressModeWrap;
    texture.texDesc.filterMode = cudaFilterModeLinear;
    texture.texDesc.readMode = cudaReadModeNormalizedFloat;
    texture.texDesc.normalizedCoords = 1;
    texture.texDesc.maxMipmapLevelClamp = float(numLevels - 1);
    texture.texDesc.minMipmapLevelClamp = 0;
    texture.texDesc.mipmapFilterMode = cudaFilterModeLinear;
    
    texture.format = channelDesc;
    
    return true;
}

TexObj TextureRegistry::createTextureObject(const Texture2D& texture) {
    TexObj texObj = 0;
    cudaError_t err = cudaCreateTextureObject(&texObj, &texture.resourceDesc, &texture.texDesc, nullptr);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create texture object: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }
    return texObj;
}

TexObj TextureRegistry::getTexture(const std::string& filepath) {
    auto it = m_texturePathToIndex.find(filepath);
    if (it != m_texturePathToIndex.end()) {
        return m_textureObjects[it->second];
    }
    
    // Try to load the texture if not found
    return loadTexture(filepath);
}

bool TextureRegistry::hasTexture(const std::string& filepath) const {
    return m_texturePathToIndex.find(filepath) != m_texturePathToIndex.end();
}

std::vector<std::string> TextureRegistry::getLoadedTextures() const {
    std::vector<std::string> paths;
    paths.reserve(m_texturePathToIndex.size());
    
    for (const auto& [path, index] : m_texturePathToIndex) {
        paths.push_back(path);
    }
    
    return paths;
}

size_t TextureRegistry::getTotalTextureMemory() const {
    size_t totalMemory = 0;
    
    for (const auto& texture : m_textures) {
        if (texture) {
            // Estimate memory usage
            size_t pixelSize = texture->channel * sizeof(unsigned char);
            size_t textureMemory = texture->width * texture->height * pixelSize;
            
            // Account for mipmaps if present
            if (texture->bufferArray) {
                // Mipmaps add approximately 33% more memory
                textureMemory = (textureMemory * 4) / 3;
            }
            
            totalMemory += textureMemory;
        }
    }
    
    return totalMemory;
}

void TextureRegistry::createDefaultTextures() {
    // Create 1x1 default textures
    unsigned char whitePixel[] = {255, 255, 255, 255};
    unsigned char grayPixel[] = {128, 128, 128, 255};
    unsigned char normalPixel[] = {128, 128, 255, 255};  // Default normal (0, 0, 1)
    unsigned char blackPixel[] = {0, 0, 0, 255};
    
    auto createDefaultTexture = [](unsigned char* data, int channels) -> TexObj {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        cudaArray_t cuArray;
        CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, 1, 1));
        CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, data, 4, 4, 1, cudaMemcpyHostToDevice));
        
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;
        
        cudaTextureDesc texDesc{};
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;
        
        TexObj texObj = 0;
        CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
        return texObj;
    };
    
    m_defaultAlbedo = createDefaultTexture(whitePixel, 4);
    m_defaultNormal = createDefaultTexture(normalPixel, 4);
    m_defaultRoughness = createDefaultTexture(grayPixel, 4);
    m_defaultMetallic = createDefaultTexture(blackPixel, 4);
}

} // namespace Assets