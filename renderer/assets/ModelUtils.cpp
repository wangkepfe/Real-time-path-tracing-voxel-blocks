#include "ModelUtils.h"
#include "GLTFUtils.h"
#include "ObjUtils.h"

#include "util/DebugUtils.h"

#include <iostream>
#include <algorithm>
#include <string>

namespace ModelUtils {

    bool isOBJFile(const std::string& filename) {
        if (filename.length() < 4) {
            return false;
        }

        // Find the last dot to get the extension
        size_t lastDot = filename.find_last_of('.');
        if (lastDot == std::string::npos) {
            return false;
        }

        std::string ext = filename.substr(lastDot);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        return (ext == ".obj");
    }

    bool loadModel(VertexAttributes** d_attr,
                   unsigned int** d_indices,
                   unsigned int& attrSize,
                   unsigned int& indicesSize,
                   const std::string& filename) {
        // Determine file type and delegate to appropriate loader
        if (GLTFUtils::isGLTFFile(filename)) {
            return GLTFUtils::loadGLTFModel(d_attr, d_indices, attrSize, indicesSize, filename);
        } else if (isOBJFile(filename)) {
            return ObjUtils::loadOBJModel(d_attr, d_indices, attrSize, indicesSize, filename);
        } else {
            std::cerr << "Unsupported model format: " << filename << std::endl;
            attrSize = 0;
            indicesSize = 0;
            *d_attr = nullptr;
            *d_indices = nullptr;
            return false;
        }
    }

}
