#include "ModelUtils.h"
#include "GLTFUtils.h"

#include "util/DebugUtils.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

/**
 * \brief Loads a model file (.obj or .gltf) into device memory buffers.
 *
 * \param[out] d_attr      Pointer to the device memory for VertexAttributes (allocated inside).
 * \param[out] d_indices   Pointer to the device memory for index data (allocated inside).
 * \param[out] attrSize    Number of VertexAttributes (i.e., how many vertices).
 * \param[out] indicesSize Number of indices (usually 3 * number_of_triangles).
 * \param[in]  filename    Path to the model file (.obj or .gltf).
 *
 * After calling, you get:
 *     - d_attr:    device buffer containing [attrSize] VertexAttributes
 *     - d_indices: device buffer containing [indicesSize] unsigned int indices
 * You must eventually free them with cudaFree(*d_attr), cudaFree(*d_indices).
 */
void loadModel(VertexAttributes **d_attr,
               unsigned int **d_indices,
               unsigned int &attrSize,
               unsigned int &indicesSize,
               const std::string &filename)
{
    // Check if this is a GLTF file and use appropriate loader
    bool isGLTF = GLTFUtils::isGLTFFile(filename);

    if (isGLTF) {
        if (GLTFUtils::loadGLTFModel(d_attr, d_indices, attrSize, indicesSize, filename)) {
            return; // Successfully loaded GLTF file
        } else {
            attrSize = 0;
            indicesSize = 0;
            *d_attr = nullptr;
            *d_indices = nullptr;
            return;
        }
    }

    // Load as OBJ file
    // 1) Temporary storage for reading .obj data
    std::vector<Float3> tempPositions; // from "v x y z"
    std::vector<Float2> tempTexcoords; // from "vt u v"

    // We'll collect face indices for position + texcoord
    std::vector<unsigned int> positionIndices;
    std::vector<unsigned int> texcoordIndices;

    // 2) Read file line-by-line
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        attrSize = 0;
        indicesSize = 0;
        return;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        std::string prefix;
        ss >> prefix;

        // Vertex position: "v x y z"
        if (prefix == "v")
        {
            Float3 pos;
            ss >> pos.x >> pos.y >> pos.z;
            tempPositions.push_back(pos);
        }
        // Texture coordinate: "vt u v"
        else if (prefix == "vt")
        {
            Float2 tex;
            ss >> tex.x >> tex.y;
            tempTexcoords.push_back(tex);
        }
        // Face definition: "f v1/t1 v2/t2 v3/t3"
        else if (prefix == "f")
        {
            // For each of the 3 face vertices
            for (int i = 0; i < 3; i++)
            {
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
                if (vertexSS.fail())
                {
                    std::cerr << "Error reading face data in " << filename << std::endl;
                    return;
                }

                // If there's a slash, skip it
                if (vertexSS.peek() == '/')
                {
                    vertexSS >> slash;
                }

                // Attempt to read texture index
                if (vertexSS.peek() != '/')
                {
                    vertexSS >> tIndex;
                    if (vertexSS.fail())
                        tIndex = 0; // might be empty
                }

                // Skip any potential normal index
                if (vertexSS.peek() == '/')
                {
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

    // 3) Build CPU-side arrays for each face-vertex
    // We'll "expand" indices so every face-vertex is unique.
    // finalIndices will be [0,1,2, 3,4,5, 6,7,8, ...]
    // finalVertices[i] = { positionIndices[i], texcoordIndices[i] }

    const size_t totalVerts = positionIndices.size(); // # of face-vertices (3 * #triangles)
    std::vector<VertexAttributes> finalVertices(totalVerts);
    std::vector<unsigned int> finalIndices(totalVerts);

    for (size_t i = 0; i < totalVerts; ++i)
    {
        unsigned int posIdx = positionIndices[i];
        unsigned int texIdx = texcoordIndices[i];

        VertexAttributes va;
        va.vertex = (posIdx < tempPositions.size())
                        ? tempPositions[posIdx]
                        : Float3{0, 0, 0}; // fallback if out-of-range
        va.texcoord = (texIdx < tempTexcoords.size())
                          ? tempTexcoords[texIdx]
                          : Float2{0, 0};

        finalVertices[i] = va;
        finalIndices[i] = static_cast<unsigned int>(i); // each face-vertex is unique
    }

    attrSize = static_cast<unsigned int>(finalVertices.size());
    indicesSize = static_cast<unsigned int>(finalIndices.size());
    if (attrSize == 0)
    {
        // No data, just return
        *d_attr = nullptr;
        *d_indices = nullptr;
        return;
    }

    // 4) Allocate device memory and copy
    // Because you want the final pointers to be device pointers,
    // we do the cudaMalloc / cudaMemcpy here.

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)d_attr, attrSize * sizeof(VertexAttributes)));
    CUDA_CHECK(cudaMalloc((void **)d_indices, indicesSize * sizeof(unsigned int)));

    // Copy CPU data -> GPU
    CUDA_CHECK(cudaMemcpy(*d_attr,
                          finalVertices.data(),
                          attrSize * sizeof(VertexAttributes),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(*d_indices,
                          finalIndices.data(),
                          indicesSize * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    std::cout << "Loaded " << filename
              << " with " << (attrSize) << " vertices and "
              << (indicesSize / 3) << " triangles.\n";
}
