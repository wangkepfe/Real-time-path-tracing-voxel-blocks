#include "ObjUtils.h"
#include "util/DebugUtils.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace ObjUtils {

    bool extractMeshFromOBJ(std::vector<Float3>& vertices,
                             std::vector<unsigned int>& indices,
                             std::vector<Float2>& texcoords,
                             const std::string& filename) {
        // Temporary storage for reading .obj data
        std::vector<Float3> tempPositions; // from "v x y z"
        std::vector<Float2> tempTexcoords; // from "vt u v"

        // We'll collect face indices for position + texcoord
        std::vector<unsigned int> positionIndices;
        std::vector<unsigned int> texcoordIndices;

        // Read file line-by-line
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open OBJ file: " << filename << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty())
                continue;

            std::stringstream ss(line);
            std::string prefix;
            ss >> prefix;

            // Vertex position: "v x y z"
            if (prefix == "v") {
                Float3 pos;
                ss >> pos.x >> pos.y >> pos.z;
                tempPositions.push_back(pos);
            }
            // Texture coordinate: "vt u v"
            else if (prefix == "vt") {
                Float2 tex;
                ss >> tex.x >> tex.y;
                tempTexcoords.push_back(tex);
            }
            // Face definition: "f v1/t1 v2/t2 v3/t3"
            else if (prefix == "f") {
                // For each of the 3 face vertices
                for (int i = 0; i < 3; i++) {
                    std::string vertexData;
                    ss >> vertexData; // e.g., "4/2" or "4/2/1"

                    if (vertexData.empty())
                        break;

                    // Parse face indices (vIndex, tIndex, [nIndex])
                    // We'll ignore normal indices for brevity.
                    int vIndex = 0, tIndex = 0;
                    char slash;
                    std::stringstream vertexSS(vertexData);

                    vertexSS >> vIndex; // read position index
                    if (vertexSS.fail()) {
                        std::cerr << "Error reading face data in " << filename << std::endl;
                        return false;
                    }

                    // If there's a slash, skip it
                    if (vertexSS.peek() == '/') {
                        vertexSS >> slash;
                    }

                    // Attempt to read texture index
                    if (vertexSS.peek() != '/') {
                        vertexSS >> tIndex;
                        if (vertexSS.fail())
                            tIndex = 0; // might be empty
                    }

                    // Skip any potential normal index
                    if (vertexSS.peek() == '/') {
                        vertexSS >> slash;
                        int nIndex;
                        vertexSS >> nIndex;
                    }

                    // Convert 1-based indices in .obj to 0-based in C++
                    vIndex -= 1;
                    tIndex -= 1;

                    positionIndices.push_back(static_cast<unsigned int>(vIndex < 0 ? 0 : vIndex));
                    texcoordIndices.push_back(static_cast<unsigned int>(tIndex < 0 ? 0 : tIndex));
                }
            }
        }
        file.close();

        // Build final arrays by expanding indices so every face-vertex is unique
        const size_t totalVerts = positionIndices.size(); // # of face-vertices (3 * #triangles)
        vertices.resize(totalVerts);
        texcoords.resize(totalVerts);
        indices.resize(totalVerts);

        for (size_t i = 0; i < totalVerts; ++i) {
            unsigned int posIdx = positionIndices[i];
            unsigned int texIdx = texcoordIndices[i];

            vertices[i] = (posIdx < tempPositions.size()) 
                        ? tempPositions[posIdx] 
                        : Float3{0, 0, 0}; // fallback if out-of-range
                        
            texcoords[i] = (texIdx < tempTexcoords.size()) 
                         ? tempTexcoords[texIdx] 
                         : Float2{0, 0};

            indices[i] = static_cast<unsigned int>(i); // each face-vertex is unique
        }

        std::cout << "Loaded OBJ file " << filename
                  << " with " << vertices.size() << " vertices and "
                  << (indices.size() / 3) << " triangles" << std::endl;

        return true;
    }

    bool loadOBJModel(VertexAttributes** d_attr,
                      unsigned int** d_indices,
                      unsigned int& attrSize,
                      unsigned int& indicesSize,
                      const std::string& filename) {
        // Extract mesh data from OBJ
        std::vector<Float3> vertices;
        std::vector<unsigned int> indices;
        std::vector<Float2> texcoords;

        if (!extractMeshFromOBJ(vertices, indices, texcoords, filename)) {
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
            finalVertices[i].texcoord = (i < texcoords.size()) ? texcoords[i] : Float2{0, 0};
            // Set default joint data for non-animated models
            finalVertices[i].jointIndices = Int4(0, 0, 0, 0);
            finalVertices[i].jointWeights = Float4(1.0f, 0.0f, 0.0f, 0.0f);
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

}