#include "util/TextureUtils.h"
#include "util/DebugUtils.h"
#include "util/Timer.h"

#define STB_IMAGE_IMPLEMENTATION
#include "ext/stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ext/stb/stb_image_write.h"

#include "nvtt/nvtt_lowlevel.h"

#include <fstream>
#include <string>
#include <filesystem>
#include "TextureUtils.h"

namespace jazzfusion
{

    const std::vector<std::string> &TextureManager::GetTextureFiles()
    {
        static const std::vector<std::string> textureFiles = {
            "rocky_trail",
            "gray_rocks",
            "seaworn_stone_tiles",
            "beige_wall_001",
            "wood_planks",
            "bark_willow_02",
        };
        return textureFiles;
    }

    __global__ void fillFirstMipmapKernel(
        unsigned char *dMipmap,
        const unsigned char *dSource,
        int nInputBufferChannels,
        int nChannels,
        int currentSize)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= currentSize * currentSize)
            return;

        // Copy the existing channels
        for (int ch = 0; ch < nInputBufferChannels; ++ch)
        {
            dMipmap[i * nChannels + ch] = dSource[i * nInputBufferChannels + ch];
        }

        // Fill in leftover channels with 255
        for (int ch = nInputBufferChannels; ch < nChannels; ++ch)
        {
            dMipmap[i * nChannels + ch] = 255;
        }
    }

    __global__ void fillMipmapKernel(
        unsigned char *dDst,       // Current LOD
        const unsigned char *dSrc, // Previous (larger) LOD
        int nChannels,
        int dstSize) // = currentSize
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= dstSize || y >= dstSize)
            return;

        // We assume that srcSize = 2 * dstSize
        // Index for the downsampled LOD
        int dstIdx = y * dstSize + x;

        int srcSize = dstSize * 2;
        int i00 = (y * 2) * srcSize + (x * 2);
        int i01 = (y * 2) * srcSize + (x * 2 + 1);
        int i10 = (y * 2 + 1) * srcSize + (x * 2);
        int i11 = (y * 2 + 1) * srcSize + (x * 2 + 1);

        for (int ch = 0; ch < nChannels; ++ch)
        {
            float val =
                static_cast<float>(dSrc[i00 * nChannels + ch]) +
                static_cast<float>(dSrc[i01 * nChannels + ch]) +
                static_cast<float>(dSrc[i10 * nChannels + ch]) +
                static_cast<float>(dSrc[i11 * nChannels + ch]);
            val *= 0.25f; // Average
            if (val > 255.0f)
                val = 255.0f;

            dDst[dstIdx * nChannels + ch] = static_cast<unsigned char>(val);
        }
    }

    inline int log2i(int a)
    {
        assert(a != 0);
        int targetlevel = 0;
        while (a >>= 1)
            ++targetlevel;
        return targetlevel;
    }

    inline bool IsPowerOfTwo(int a)
    {
        return (a & (a - 1)) == 0;
    }

    void TextureManager::init()
    {
        std::filesystem::path cwd = std::filesystem::current_path();

        std::vector<std::string> filePaths;
        for (const auto &textureFile : GetTextureFiles())
        {
            filePaths.emplace_back("data/" + textureFile + "_albedo.png");
            filePaths.emplace_back("data/" + textureFile + "_normal.png");
            filePaths.emplace_back("data/" + textureFile + "_rough.png");
        }
        filePaths.emplace_back("data/water1.jpg");

        m_textures.resize(filePaths.size());

        if (!std::filesystem::exists("tex"))
        {
            std::filesystem::create_directory("tex");
        }

        for (int i = 0; i < filePaths.size(); ++i)
        {
            std::string filePath = filePaths[i];
            textureNameIdLookup[filePath] = i;

            // ScopeTimer timer("Generating texture for " + filePath);

            int fileNamePosStart = filePath.find('/') + 1;
            int fileNamePosEnd = filePath.find(".png");

            std::string cacheFileNameBase = "tex/" + filePath.substr(fileNamePosStart, fileNamePosEnd - fileNamePosStart);

            auto &texture = m_textures[i];
            auto &texObj = m_shaderTextures.texObjs[i];

            stbi_uc *buffer = stbi_load(filePath.c_str(), &texture.width, &texture.height, &texture.channel, 0);

            int nChannels = 4;
            int nInputBufferChannels = texture.channel;
            int bytesPerTile = 16;

            if (texture.channel == 3 || texture.channel == 4)
            {
                texture.format = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed7>();
                nChannels = 4;
                bytesPerTile = 16;
            }
            else if (texture.channel == 2)
            {
                texture.format = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed5>();
                nChannels = 2;
                bytesPerTile = 16;
            }
            else if (texture.channel == 1)
            {
                texture.format = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed4>();
                nChannels = 1;
                bytesPerTile = 8;
            }
            else
            {
                assert(0);
            }

            assert(texture.width == texture.height);
            assert(IsPowerOfTwo(texture.width));

            int maxLod = log2i(texture.width) - 2;
            int numLods = maxLod + 1;

            CUDA_CHECK(cudaMallocMipmappedArray(&texture.bufferArray, &texture.format, make_cudaExtent(texture.width, texture.height, 0), numLods, 0));

            texture.resourceDesc = cudaResourceDesc{};
            texture.resourceDesc.resType = cudaResourceTypeMipmappedArray;
            texture.resourceDesc.res.mipmap.mipmap = texture.bufferArray;

            texture.texDesc = cudaTextureDesc{};
            texture.texDesc.addressMode[0] = cudaAddressModeWrap;
            texture.texDesc.addressMode[1] = cudaAddressModeWrap;
            texture.texDesc.addressMode[2] = cudaAddressModeWrap;
            texture.texDesc.borderColor[0] = 0.0f;
            texture.texDesc.borderColor[1] = 0.0f;
            texture.texDesc.borderColor[2] = 0.0f;
            texture.texDesc.borderColor[3] = 0.0f;
            texture.texDesc.disableTrilinearOptimization = 0;
            texture.texDesc.filterMode = cudaFilterModeLinear;
            texture.texDesc.maxAnisotropy = 0;
            texture.texDesc.maxMipmapLevelClamp = (float)maxLod;
            texture.texDesc.minMipmapLevelClamp = 0.0f;
            texture.texDesc.mipmapFilterMode = cudaFilterModeLinear;
            texture.texDesc.mipmapLevelBias = 0;
            texture.texDesc.normalizedCoords = 1;
            texture.texDesc.readMode = cudaReadModeNormalizedFloat;
            texture.texDesc.seamlessCubemap = 0;
            texture.texDesc.sRGB = 0;

            CUDA_CHECK(cudaCreateTextureObject(&texObj, &texture.resourceDesc, &texture.texDesc, nullptr));

            std::vector<std::vector<uint8_t>> mipmapBuffers(numLods);
            int currentSize = texture.width;

            for (int lod = 0; lod < numLods; ++lod)
            {
                // ScopeTimer timer("Generating mip level " + std::to_string(lod));

                cudaArray_t currentMipLevelArray;
                CUDA_CHECK(cudaGetMipmappedArrayLevel(&currentMipLevelArray, texture.bufferArray, lod));

                cudaExtent currentMipLevelSize;
                CUDA_CHECK(cudaArrayGetInfo(NULL, &currentMipLevelSize, NULL, currentMipLevelArray));

                assert(currentMipLevelSize.width == currentSize && currentMipLevelSize.height == currentSize);

                int pitch = currentSize / 4 * bytesPerTile;
                int height = currentSize / 4;

                std::string cacheFileName = cacheFileNameBase + "_lod_" + std::to_string(lod) + ".bin";
                std::string cacheFileNameDebug = cacheFileNameBase + "_lod_" + std::to_string(lod) + ".png";

                int expectedCompressedTextureSize = currentSize * currentSize / 16 * bytesPerTile;

                std::vector<std::byte> compressedImageBuffer(expectedCompressedTextureSize);

                if (std::filesystem::exists(cacheFileName))
                {
                    std::ifstream infile(cacheFileName, std::ifstream::in | std::ifstream::binary);
                    assert(infile.good());
                    infile.read(reinterpret_cast<char *>(compressedImageBuffer.data()), expectedCompressedTextureSize);
                    infile.close();
                }
                else
                {
                    mipmapBuffers[lod].resize(currentSize * currentSize * nChannels);

                    unsigned char *dSource = nullptr;
                    unsigned char *dMipmap = nullptr;

                    if (lod == 0)
                    // Example: create first LOD on GPU
                    {
                        size_t numPixels = currentSize * currentSize;
                        size_t sizeBytes = numPixels * nChannels * sizeof(unsigned char);

                        // Allocate device buffer for the source buffer (optional if you do it once)
                        CUDA_CHECK(cudaMalloc(&dSource, numPixels * nInputBufferChannels * sizeof(unsigned char)));
                        // Copy from host
                        CUDA_CHECK(cudaMemcpy(dSource, buffer, numPixels * nInputBufferChannels * sizeof(unsigned char),
                                              cudaMemcpyHostToDevice));

                        // Allocate device buffer for the first LOD
                        CUDA_CHECK(cudaMalloc(&dMipmap, sizeBytes));

                        // Launch fillFirstMipmapKernel
                        int blockSize = 256;
                        int gridSize = (numPixels + blockSize - 1) / blockSize;
                        fillFirstMipmapKernel<<<gridSize, blockSize>>>(
                            dMipmap,
                            dSource,
                            nInputBufferChannels,
                            nChannels,
                            currentSize);
                        CUDA_CHECK(cudaDeviceSynchronize());

                        // Copy back to host (mipmapBuffers[0]) for CPU-based compression
                        CUDA_CHECK(cudaMemcpy(mipmapBuffers[lod].data(), dMipmap, sizeBytes, cudaMemcpyDeviceToHost));

                        // Now you can compress on the CPU side, store to compressed buffer, etc...

                        CUDA_CHECK(cudaFree(dSource));
                        CUDA_CHECK(cudaFree(dMipmap));
                    }
                    else if (lod < 3)
                    // Next LODs
                    {
                        // Suppose we want to generate mipmapBuffers[lod] from mipmapBuffers[lod - 1]
                        // We can do a similar approach but with fillMipmapKernel.
                        // For each LOD, the "srcSize" = previous LOD dimension, which is 2 * currentSize.
                        // The "dstSize" = currentSize (the new LOD dimension).

                        // 1) Upload mipmapBuffers[lod - 1] to device (dSource).
                        // 2) Allocate dDst for mipmapBuffers[lod].
                        // 3) Launch fillMipmapKernel<<<dimGrid, dimBlock>>>(...).
                        // 4) Copy dDst back to host.

                        int dstSize = currentSize;     // for the new LOD
                        int srcSize = currentSize * 2; // for the old LOD

                        // Example device buffers
                        unsigned char *dSrc = nullptr;
                        unsigned char *dDst = nullptr;

                        size_t srcNumPixels = srcSize * srcSize;
                        size_t dstNumPixels = dstSize * dstSize;

                        CUDA_CHECK(cudaMalloc(&dSrc, srcNumPixels * nChannels * sizeof(unsigned char)));
                        CUDA_CHECK(cudaMalloc(&dDst, dstNumPixels * nChannels * sizeof(unsigned char)));

                        // Copy previous LOD’s data to dSrc
                        CUDA_CHECK(cudaMemcpy(
                            dSrc,
                            mipmapBuffers[lod - 1].data(),
                            srcNumPixels * nChannels * sizeof(unsigned char),
                            cudaMemcpyHostToDevice));

                        // Launch kernel
                        dim3 block(16, 16);
                        dim3 grid(
                            (dstSize + block.x - 1) / block.x,
                            (dstSize + block.y - 1) / block.y);

                        fillMipmapKernel<<<grid, block>>>(
                            dDst,
                            dSrc,
                            nChannels,
                            dstSize);
                        CUDA_CHECK(cudaDeviceSynchronize());

                        // Copy dDst back to host
                        CUDA_CHECK(cudaMemcpy(
                            mipmapBuffers[lod].data(),
                            dDst,
                            dstNumPixels * nChannels * sizeof(unsigned char),
                            cudaMemcpyDeviceToHost));

                        // Free device buffers
                        CUDA_CHECK(cudaFree(dSrc));
                        CUDA_CHECK(cudaFree(dDst));

                        // Continue with compression, etc...
                    }
                    else
                    {
                        for (int x = 0; x < currentSize; ++x)
                        {
                            for (int y = 0; y < currentSize; ++y)
                            {
                                int i = y * currentSize + x;

                                int i00 = (y * 2) * 2 * currentSize + (x * 2);
                                int i01 = (y * 2) * 2 * currentSize + (x * 2 + 1);
                                int i10 = (y * 2 + 1) * 2 * currentSize + (x * 2);
                                int i11 = (y * 2 + 1) * 2 * currentSize + (x * 2 + 1);

                                for (int ch = 0; ch < nChannels; ++ch)
                                {
                                    float val = static_cast<float>(mipmapBuffers[lod - 1][i00 * nChannels + ch]) +
                                                static_cast<float>(mipmapBuffers[lod - 1][i01 * nChannels + ch]) +
                                                static_cast<float>(mipmapBuffers[lod - 1][i10 * nChannels + ch]) +
                                                static_cast<float>(mipmapBuffers[lod - 1][i11 * nChannels + ch]);

                                    val /= 4.0f;
                                    val = std::min(val, 255.0f);

                                    mipmapBuffers[lod][i * nChannels + ch] = static_cast<uint8_t>(val);
                                }
                            }
                        }
                    }

                    stbi_write_png(cacheFileNameDebug.c_str(), currentSize, currentSize, nChannels, mipmapBuffers[lod].data(), currentSize * nChannels);

                    nvtt::RefImage refImage{};
                    refImage.data = mipmapBuffers[lod].data();
                    refImage.width = currentSize;
                    refImage.height = currentSize;
                    refImage.depth = 1;
                    refImage.num_channels = nChannels;

                    nvtt::CPUInputBuffer cpuInputBuffer(&refImage, nvtt::UINT8);

                    int numTiles = cpuInputBuffer.NumTiles();
                    int tileWidth = 0;
                    int tileHeight = 0;
                    cpuInputBuffer.TileSize(tileWidth, tileHeight);

                    int compressedTextureSize = bytesPerTile * numTiles;

                    assert(expectedCompressedTextureSize == compressedTextureSize);
                    assert(tileWidth == 4);
                    assert(tileHeight == 4);

                    if (texture.channel == 4)
                    {
                        nvtt::nvtt_encode_bc7(cpuInputBuffer, true, true, compressedImageBuffer.data(), false, false);
                    }
                    else if (texture.channel == 3)
                    {
                        nvtt::nvtt_encode_bc7(cpuInputBuffer, true, false, compressedImageBuffer.data(), false, false);
                    }
                    else if (texture.channel == 2)
                    {
                        nvtt::nvtt_encode_bc5(cpuInputBuffer, true, compressedImageBuffer.data(), false, false);
                    }
                    else if (texture.channel == 1)
                    {
                        nvtt::nvtt_encode_bc4(cpuInputBuffer, true, compressedImageBuffer.data(), false, false);
                    }
                    else
                    {
                        assert(0);
                    }

                    {
                        std::ofstream myfile(cacheFileName, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
                        assert(myfile.is_open());
                        myfile.write(reinterpret_cast<char *>(compressedImageBuffer.data()), compressedTextureSize);
                        myfile.close();
                        std::cout << "Successfully saved cached compressed texture to file \"" << cacheFileName << "\".\n";
                    }
                }

                CUDA_CHECK(cudaMemcpy2DToArray(currentMipLevelArray, 0, 0, compressedImageBuffer.data(), pitch, pitch, height, cudaMemcpyHostToDevice));
                currentSize >>= 1;
            }

            STBI_FREE(buffer);
        }
    }
}
