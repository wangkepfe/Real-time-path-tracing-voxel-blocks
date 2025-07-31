#pragma once

#include <string>
#include <vector>

// Forward declarations
namespace tinygltf {
    class Model;
}

struct VertexAttributes;
struct Float3;
struct Float2;
struct Int4;
struct Float4;
struct Skeleton;
struct AnimationClip;

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

    /**
     * \brief Loads a complete animated GLTF model with skinning and animation data
     * \param d_attr Output pointer to device memory for VertexAttributes (with joints/weights)
     * \param d_indices Output pointer to device memory for index data
     * \param attrSize Output size of vertex attributes array
     * \param indicesSize Output size of indices array
     * \param skeleton Output skeleton structure with joints and bind poses
     * \param animationClips Output vector of animation clips
     * \param filename Path to the GLTF file
     * \return True if successful, false otherwise
     */
    bool loadAnimatedGLTFModel(VertexAttributes** d_attr,
                               unsigned int** d_indices,
                               unsigned int& attrSize,
                               unsigned int& indicesSize,
                               Skeleton& skeleton,
                               std::vector<AnimationClip>& animationClips,
                               const std::string& filename);

    /**
     * \brief Extracts complete animated mesh data from a GLTF file
     * \param vertices Output vector of vertex positions
     * \param indices Output vector of triangle indices
     * \param texcoords Output vector of texture coordinates
     * \param jointIndices Output vector of joint indices per vertex
     * \param jointWeights Output vector of joint weights per vertex
     * \param skeleton Output skeleton structure
     * \param animationClips Output vector of animation clips
     * \param filename Path to the GLTF file
     * \return True if successful, false otherwise
     */
    bool extractAnimatedMeshFromGLTF(std::vector<Float3>& vertices,
                                     std::vector<unsigned int>& indices,
                                     std::vector<Float2>& texcoords,
                                     std::vector<Int4>& jointIndices,
                                     std::vector<Float4>& jointWeights,
                                     Skeleton& skeleton,
                                     std::vector<AnimationClip>& animationClips,
                                     const std::string& filename);

    /**
     * \brief Extracts skeleton data from a GLTF model
     * \param skeleton Output skeleton structure
     * \param model The loaded GLTF model
     * \return True if successful, false otherwise
     */
    bool extractSkeletonFromGLTF(Skeleton& skeleton, const tinygltf::Model& model);

    /**
     * \brief Extracts animation clips from a GLTF model
     * \param animationClips Output vector of animation clips
     * \param model The loaded GLTF model
     * \return True if successful, false otherwise
     */
    bool extractAnimationsFromGLTF(std::vector<AnimationClip>& animationClips, const tinygltf::Model& model);
}