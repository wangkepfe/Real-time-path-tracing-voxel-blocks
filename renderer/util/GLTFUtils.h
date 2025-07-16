#pragma once

#include "shaders/SystemParameter.h"
#include <string>
#include <vector>

namespace GLTFUtils {
    /**
     * \brief Checks if a filename is a GLTF file based on its extension
     * \param filename The filename to check
     * \return True if the file is a GLTF file (.gltf or .glb), false otherwise
     */
    bool isGLTFFile(const std::string& filename);

    /**
     * \brief Loads a GLTF model file into device memory buffers
     * \param d_attr Output pointer to device memory for VertexAttributes
     * \param d_indices Output pointer to device memory for index data
     * \param attrSize Output size of vertex attributes array
     * \param indicesSize Output size of indices array
     * \param filename Path to the GLTF file
     * \return True if successful, false otherwise
     */
    bool loadGLTFModel(VertexAttributes** d_attr,
                       unsigned int** d_indices,
                       unsigned int& attrSize,
                       unsigned int& indicesSize,
                       const std::string& filename);

    /**
     * \brief Extracts mesh data from a GLTF file
     * \param vertices Output vector of vertex positions
     * \param indices Output vector of triangle indices
     * \param texcoords Output vector of texture coordinates
     * \param filename Path to the GLTF file
     * \return True if successful, false otherwise
     */
    bool extractMeshFromGLTF(std::vector<Float3>& vertices,
                             std::vector<unsigned int>& indices,
                             std::vector<Float2>& texcoords,
                             const std::string& filename);
}