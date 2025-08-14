#pragma once

#include "shaders/SystemParameter.h"
#include <string>
#include <vector>

namespace ObjUtils {
    /**
     * \brief Loads an OBJ model file into CPU vectors
     * \param vertices Output vector of vertex positions
     * \param indices Output vector of triangle indices
     * \param texcoords Output vector of texture coordinates
     * \param filename Path to the OBJ file
     * \return True if successful, false otherwise
     */
    bool extractMeshFromOBJ(std::vector<Float3>& vertices,
                             std::vector<unsigned int>& indices,
                             std::vector<Float2>& texcoords,
                             const std::string& filename);

    /**
     * \brief Loads an OBJ model file into device memory buffers
     * \param d_attr Output pointer to device memory for VertexAttributes
     * \param d_indices Output pointer to device memory for index data
     * \param attrSize Output size of vertex attributes array
     * \param indicesSize Output size of indices array
     * \param filename Path to the OBJ file
     * \return True if successful, false otherwise
     */
    bool loadOBJModel(VertexAttributes** d_attr,
                      unsigned int** d_indices,
                      unsigned int& attrSize,
                      unsigned int& indicesSize,
                      const std::string& filename);
}