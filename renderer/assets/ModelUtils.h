#pragma once

#include "shaders/SystemParameter.h"
#include <string>

namespace ModelUtils {
    /**
     * \brief High-level model loading function that delegates to appropriate loaders
     * \param d_attr Output pointer to device memory for VertexAttributes
     * \param d_indices Output pointer to device memory for index data
     * \param attrSize Output size of vertex attributes array
     * \param indicesSize Output size of indices array
     * \param filename Path to the model file (.obj or .gltf/.glb)
     * \return True if successful, false otherwise
     */
    bool loadModel(VertexAttributes** d_attr,
                   unsigned int** d_indices,
                   unsigned int& attrSize,
                   unsigned int& indicesSize,
                   const std::string& filename);

    /**
     * \brief Checks if a filename is an OBJ file based on its extension
     * \param filename The filename to check
     * \return True if the file is an OBJ file (.obj), false otherwise
     */
    bool isOBJFile(const std::string& filename);
}