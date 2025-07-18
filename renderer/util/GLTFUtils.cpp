#include "GLTFUtils.h"
#include "DebugUtils.h"

#include "tiny_gltf.h"

#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

namespace GLTFUtils {

    bool isGLTFFile(const std::string& filename) {
        if (filename.length() < 5) return false; // Need at least ".gltf" which is 5 characters

        // Find the last dot to get the extension
        size_t lastDot = filename.find_last_of('.');
        if (lastDot == std::string::npos) {
            return false;
        }

        std::string ext = filename.substr(lastDot);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        return (ext == ".gltf" || ext == ".glb");
    }

    bool extractMeshFromGLTF(std::vector<Float3>& vertices,
                             std::vector<unsigned int>& indices,
                             std::vector<Float2>& texcoords,
                             const std::string& filename) {
        tinygltf::Model model;
        tinygltf::TinyGLTF loader;
        std::string err;
        std::string warn;

        bool ret = false;
        if (filename.find(".glb") != std::string::npos) {
            ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
        } else {
            ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
        }

        if (!ret) {
            return false;
        }

        // Process each mesh
        for (const auto& mesh : model.meshes) {
            for (const auto& primitive : mesh.primitives) {
                if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                    continue; // Skip non-triangle primitives
                }

                // Get position attribute
                auto positionIt = primitive.attributes.find("POSITION");
                if (positionIt == primitive.attributes.end()) {
                    continue; // Skip primitives without position data
                }

                const tinygltf::Accessor& positionAccessor = model.accessors[positionIt->second];
                const tinygltf::BufferView& positionBufferView = model.bufferViews[positionAccessor.bufferView];
                const tinygltf::Buffer& positionBuffer = model.buffers[positionBufferView.buffer];

                // Extract vertices
                const float* positionData = reinterpret_cast<const float*>(
                    &positionBuffer.data[positionBufferView.byteOffset + positionAccessor.byteOffset]);

                for (size_t i = 0; i < positionAccessor.count; ++i) {
                    Float3 vertex;
                    vertex.x = positionData[i * 3 + 0];
                    vertex.y = positionData[i * 3 + 1];
                    vertex.z = positionData[i * 3 + 2];
                    vertices.push_back(vertex);
                }

                // Get texture coordinates if available
                auto texcoordIt = primitive.attributes.find("TEXCOORD_0");
                if (texcoordIt != primitive.attributes.end()) {
                    const tinygltf::Accessor& texcoordAccessor = model.accessors[texcoordIt->second];
                    const tinygltf::BufferView& texcoordBufferView = model.bufferViews[texcoordAccessor.bufferView];
                    const tinygltf::Buffer& texcoordBuffer = model.buffers[texcoordBufferView.buffer];

                    const float* texcoordData = reinterpret_cast<const float*>(
                        &texcoordBuffer.data[texcoordBufferView.byteOffset + texcoordAccessor.byteOffset]);

                    for (size_t i = 0; i < texcoordAccessor.count; ++i) {
                        Float2 texcoord;
                        texcoord.x = texcoordData[i * 2 + 0];
                        texcoord.y = texcoordData[i * 2 + 1];
                        texcoords.push_back(texcoord);
                    }
                } else {
                    // Fill with default texture coordinates
                    for (size_t i = 0; i < positionAccessor.count; ++i) {
                        texcoords.push_back(Float2(0.0f, 0.0f));
                    }
                }

                // Get indices
                if (primitive.indices >= 0) {
                    const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
                    const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
                    const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];

                    const unsigned char* indexData = &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset];

                    for (size_t i = 0; i < indexAccessor.count; ++i) {
                        unsigned int index = 0;
                        if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                            index = reinterpret_cast<const unsigned short*>(indexData)[i];
                        } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                            index = reinterpret_cast<const unsigned int*>(indexData)[i];
                        } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                            index = indexData[i];
                        }
                        indices.push_back(index);
                    }
                }
            }
        }

        std::cout << "Loaded GLTF file " << filename
                  << " with " << vertices.size() << " vertices and "
                  << (indices.size() / 3) << " triangles" << std::endl;

        return true;
    }

    bool loadGLTFModel(VertexAttributes** d_attr,
                       unsigned int** d_indices,
                       unsigned int& attrSize,
                       unsigned int& indicesSize,
                       const std::string& filename) {
        // Extract mesh data from GLTF
        std::vector<Float3> vertices;
        std::vector<unsigned int> indices;
        std::vector<Float2> texcoords;

        if (!extractMeshFromGLTF(vertices, indices, texcoords, filename)) {
            attrSize = 0;
            indicesSize = 0;
            *d_attr = nullptr;
            *d_indices = nullptr;
            return false;
        }

        // Build CPU-side vertex attributes array
        std::vector<VertexAttributes> finalVertices(vertices.size());
        for (size_t i = 0; i < vertices.size(); ++i) {
            finalVertices[i].vertex = vertices[i];
            finalVertices[i].texcoord = (i < texcoords.size()) ? texcoords[i] : Float2(0.0f, 0.0f);
        }

        attrSize = static_cast<unsigned int>(finalVertices.size());
        indicesSize = static_cast<unsigned int>(indices.size());

        if (attrSize == 0) {
            *d_attr = nullptr;
            *d_indices = nullptr;
            return false;
        }

        // Allocate device memory
        CUDA_CHECK(cudaMalloc((void**)d_attr, attrSize * sizeof(VertexAttributes)));
        CUDA_CHECK(cudaMalloc((void**)d_indices, indicesSize * sizeof(unsigned int)));

        // Copy CPU data to GPU
        CUDA_CHECK(cudaMemcpy(*d_attr,
                              finalVertices.data(),
                              attrSize * sizeof(VertexAttributes),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(*d_indices,
                              indices.data(),
                              indicesSize * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        return true;
    }

} // namespace GLTFUtils