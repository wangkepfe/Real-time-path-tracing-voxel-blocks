#include "GLTFUtils.h"
#include "DebugUtils.h"
#include "../animation/Animation.h"
#include "../animation/Skeleton.h"
#include "../shaders/SystemParameter.h"

#include "tiny_gltf.h"

#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <unordered_map>

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

    bool extractAnimatedMeshFromGLTF(std::vector<Float3>& vertices,
                                     std::vector<unsigned int>& indices,
                                     std::vector<Float2>& texcoords,
                                     std::vector<Int4>& jointIndices,
                                     std::vector<Float4>& jointWeights,
                                     Skeleton& skeleton,
                                     std::vector<AnimationClip>& animationClips,
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
            std::cerr << "Failed to load GLTF file: " << filename << std::endl;
            if (!err.empty()) std::cerr << "Error: " << err << std::endl;
            if (!warn.empty()) std::cerr << "Warning: " << warn << std::endl;
            return false;
        }

        std::cout << "Loading animated GLTF file: " << filename << std::endl;
        std::cout << "Model has " << model.meshes.size() << " meshes, "
                  << model.skins.size() << " skins, "
                  << model.animations.size() << " animations" << std::endl;

        // Extract skeleton data first
        // Extract skeleton data first
        if (!extractSkeletonFromGLTF(skeleton, model)) {
            std::cerr << "Failed to extract skeleton from GLTF" << std::endl;
            return false;
        }

        // Extract animation clips
        if (!extractAnimationsFromGLTF(animationClips, model)) {
            std::cerr << "Failed to extract animations from GLTF" << std::endl;
            return false;
        }

        // Process skinned meshes
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

                // Get texture coordinates
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
                    for (size_t i = 0; i < positionAccessor.count; ++i) {
                        texcoords.push_back(Float2(0.0f, 0.0f));
                    }
                }

                // Get joint indices (JOINTS_0)
                auto jointsIt = primitive.attributes.find("JOINTS_0");
                if (jointsIt != primitive.attributes.end()) {
                    const tinygltf::Accessor& jointsAccessor = model.accessors[jointsIt->second];
                    const tinygltf::BufferView& jointsBufferView = model.bufferViews[jointsAccessor.bufferView];
                    const tinygltf::Buffer& jointsBuffer = model.buffers[jointsBufferView.buffer];

                    const unsigned char* jointsData = &jointsBuffer.data[jointsBufferView.byteOffset + jointsAccessor.byteOffset];

                    for (size_t i = 0; i < jointsAccessor.count; ++i) {
                        Int4 joints;
                        if (jointsAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                            const unsigned short* shortData = reinterpret_cast<const unsigned short*>(jointsData);
                            joints.x = shortData[i * 4 + 0];
                            joints.y = shortData[i * 4 + 1];
                            joints.z = shortData[i * 4 + 2];
                            joints.w = shortData[i * 4 + 3];
                        } else if (jointsAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                            joints.x = jointsData[i * 4 + 0];
                            joints.y = jointsData[i * 4 + 1];
                            joints.z = jointsData[i * 4 + 2];
                            joints.w = jointsData[i * 4 + 3];
                        }
                        jointIndices.push_back(joints);
                    }
                } else {
                    // Default to root joint
                    for (size_t i = 0; i < positionAccessor.count; ++i) {
                        jointIndices.push_back(Int4(0, 0, 0, 0));
                    }
                }

                // Get joint weights (WEIGHTS_0)
                auto weightsIt = primitive.attributes.find("WEIGHTS_0");
                if (weightsIt != primitive.attributes.end()) {
                    const tinygltf::Accessor& weightsAccessor = model.accessors[weightsIt->second];
                    const tinygltf::BufferView& weightsBufferView = model.bufferViews[weightsAccessor.bufferView];
                    const tinygltf::Buffer& weightsBuffer = model.buffers[weightsBufferView.buffer];

                    const float* weightsData = reinterpret_cast<const float*>(
                        &weightsBuffer.data[weightsBufferView.byteOffset + weightsAccessor.byteOffset]);

                    for (size_t i = 0; i < weightsAccessor.count; ++i) {
                        Float4 weights;
                        weights.x = weightsData[i * 4 + 0];
                        weights.y = weightsData[i * 4 + 1];
                        weights.z = weightsData[i * 4 + 2];
                        weights.w = weightsData[i * 4 + 3];

                        // Normalize weights to ensure they sum to 1.0
                        float sum = weights.x + weights.y + weights.z + weights.w;
                        if (sum > 0.0f) {
                            weights.x /= sum;
                            weights.y /= sum;
                            weights.z /= sum;
                            weights.w /= sum;
                        }

                        jointWeights.push_back(weights);
                    }
                } else {
                    // Default weights - full weight to first joint
                    for (size_t i = 0; i < positionAccessor.count; ++i) {
                        jointWeights.push_back(Float4(1.0f, 0.0f, 0.0f, 0.0f));
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

        std::cout << "Loaded animated GLTF: " << vertices.size() << " vertices, "
                  << (indices.size() / 3) << " triangles, "
                  << skeleton.joints.size() << " joints, "
                  << animationClips.size() << " animations" << std::endl;

        return true;
    }

    bool extractSkeletonFromGLTF(Skeleton& skeleton, const tinygltf::Model& model) {
        if (model.skins.empty()) {
            std::cout << "No skins found in GLTF model" << std::endl;
            return false;
        }

        // Use the first skin (most models have only one)
        const tinygltf::Skin& skin = model.skins[0];

        skeleton.joints.clear();
        skeleton.jointNameToIndex.clear();

        // Create joints from skin
        for (size_t i = 0; i < skin.joints.size(); ++i) {
            if (skin.joints[i] >= model.nodes.size()) {
                std::cerr << "Invalid joint index: " << skin.joints[i] << std::endl;
                continue;
            }

            const tinygltf::Node& node = model.nodes[skin.joints[i]];
            Joint joint;

            // Set joint name
            joint.name = node.name.empty() ? ("joint_" + std::to_string(i)) : node.name;

            // Extract local transform and store as both current and bind pose
            if (node.translation.size() == 3) {
                joint.position = joint.bindPosition = Float3(
                    static_cast<float>(node.translation[0]),
                    static_cast<float>(node.translation[1]),
                    static_cast<float>(node.translation[2])
                );
            }

            if (node.rotation.size() == 4) {
                joint.rotation = joint.bindRotation = Float4(
                    static_cast<float>(node.rotation[0]),
                    static_cast<float>(node.rotation[1]),
                    static_cast<float>(node.rotation[2]),
                    static_cast<float>(node.rotation[3])
                );
            }

            if (node.scale.size() == 3) {
                joint.scale = joint.bindScale = Float3(
                    static_cast<float>(node.scale[0]),
                    static_cast<float>(node.scale[1]),
                    static_cast<float>(node.scale[2])
                );
            }

            // Find parent joint
            joint.parentIndex = -1;
            for (size_t j = 0; j < skin.joints.size(); ++j) {
                if (j == i) continue;

                const tinygltf::Node& potentialParent = model.nodes[skin.joints[j]];
                auto it = std::find(potentialParent.children.begin(), potentialParent.children.end(), skin.joints[i]);
                if (it != potentialParent.children.end()) {
                    joint.parentIndex = static_cast<int>(j);
                    break;
                }
            }

            skeleton.joints.push_back(joint);
            skeleton.jointNameToIndex[joint.name] = static_cast<int>(i);
        }

        // Extract inverse bind matrices
        if (skin.inverseBindMatrices >= 0) {
            const tinygltf::Accessor& accessor = model.accessors[skin.inverseBindMatrices];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

            const float* matrixData = reinterpret_cast<const float*>(
                &buffer.data[bufferView.byteOffset + accessor.byteOffset]);

            for (size_t i = 0; i < skeleton.joints.size() && i < accessor.count; ++i) {
                memcpy(&skeleton.joints[i].inverseBindMatrix, &matrixData[i * 16], sizeof(float) * 16);
            }
        } else {
            // Set identity matrices
            for (auto& joint : skeleton.joints) {
                joint.inverseBindMatrix = Mat4(); // Default constructor creates identity
            }
        }

        std::cout << "Extracted skeleton with " << skeleton.joints.size() << " joints" << std::endl;

        return true;
    }

    bool extractAnimationsFromGLTF(std::vector<AnimationClip>& animationClips, const tinygltf::Model& model) {
        animationClips.clear();

        for (const auto& gltfAnimation : model.animations) {
            AnimationClip clip;
            clip.name = gltfAnimation.name.empty() ? "animation" : gltfAnimation.name;
            clip.duration = 0.0f;

            // Extract samplers
            for (const auto& gltfSampler : gltfAnimation.samplers) {
                AnimationSampler sampler;

                // Set interpolation
                if (gltfSampler.interpolation == "STEP") {
                    sampler.interpolation = AnimationSampler::STEP;
                } else if (gltfSampler.interpolation == "CUBICSPLINE") {
                    sampler.interpolation = AnimationSampler::CUBICSPLINE;
                } else {
                    sampler.interpolation = AnimationSampler::LINEAR;
                }

                // Extract input times
                if (gltfSampler.input >= 0) {
                    const tinygltf::Accessor& inputAccessor = model.accessors[gltfSampler.input];
                    const tinygltf::BufferView& inputBufferView = model.bufferViews[inputAccessor.bufferView];
                    const tinygltf::Buffer& inputBuffer = model.buffers[inputBufferView.buffer];

                    const float* timeData = reinterpret_cast<const float*>(
                        &inputBuffer.data[inputBufferView.byteOffset + inputAccessor.byteOffset]);

                    for (size_t i = 0; i < inputAccessor.count; ++i) {
                        sampler.inputTimes.push_back(timeData[i]);
                        clip.duration = std::max(clip.duration, timeData[i]);
                    }
                }

                // Extract output values (we'll determine the type from channels)
                if (gltfSampler.output >= 0) {
                    const tinygltf::Accessor& outputAccessor = model.accessors[gltfSampler.output];
                    const tinygltf::BufferView& outputBufferView = model.bufferViews[outputAccessor.bufferView];
                    const tinygltf::Buffer& outputBuffer = model.buffers[outputBufferView.buffer];

                    const float* outputData = reinterpret_cast<const float*>(
                        &outputBuffer.data[outputBufferView.byteOffset + outputAccessor.byteOffset]);

                    // We'll populate the specific arrays based on the channel type
                    // For now, store the raw data in translations (we'll sort it out in channels)
                    if (outputAccessor.type == TINYGLTF_TYPE_VEC3) {
                        for (size_t i = 0; i < outputAccessor.count; ++i) {
                            Float3 vec3Data(
                                outputData[i * 3 + 0],
                                outputData[i * 3 + 1],
                                outputData[i * 3 + 2]
                            );
                            sampler.outputTranslations.push_back(vec3Data);
                            
                            // Debug output for hip translation data
                            if (clip.name == "Run") {
                                std::cout << "Loading Run animation VEC3 data [" << i << "]: " 
                                          << vec3Data.x << ", " << vec3Data.y << ", " << vec3Data.z << std::endl;
                            }
                        }
                    } else if (outputAccessor.type == TINYGLTF_TYPE_VEC4) {
                        for (size_t i = 0; i < outputAccessor.count; ++i) {
                            sampler.outputRotations.push_back(Float4(
                                outputData[i * 4 + 0],
                                outputData[i * 4 + 1],
                                outputData[i * 4 + 2],
                                outputData[i * 4 + 3]
                            ));
                        }
                    }
                }

                clip.samplers.push_back(sampler);
            }

            // Extract channels
            for (const auto& gltfChannel : gltfAnimation.channels) {
                AnimationChannel channel;
                channel.samplerIndex = gltfChannel.sampler;

                // Determine target joint and property
                if (gltfChannel.target_node >= 0) {
                    // Find the joint index for this node
                    for (size_t i = 0; i < model.skins.size(); ++i) {
                        const auto& skin = model.skins[i];
                        auto it = std::find(skin.joints.begin(), skin.joints.end(), gltfChannel.target_node);
                        if (it != skin.joints.end()) {
                            channel.targetJoint = static_cast<int>(std::distance(skin.joints.begin(), it));
                            break;
                        }
                    }
                }

                // Determine target property
                if (gltfChannel.target_path == "translation") {
                    channel.targetPath = AnimationChannel::TRANSLATION;
                } else if (gltfChannel.target_path == "rotation") {
                    channel.targetPath = AnimationChannel::ROTATION;
                } else if (gltfChannel.target_path == "scale") {
                    channel.targetPath = AnimationChannel::SCALE;
                }

                clip.channels.push_back(channel);
            }

            // Now reorganize sampler data based on channel types
            for (const auto& channel : clip.channels) {
                if (channel.samplerIndex >= 0 && channel.samplerIndex < static_cast<int>(clip.samplers.size())) {
                    AnimationSampler& sampler = clip.samplers[channel.samplerIndex];

                    // Move data to appropriate arrays based on channel type
                    if (channel.targetPath == AnimationChannel::TRANSLATION && !sampler.outputTranslations.empty()) {
                        // Data is already in outputTranslations
                    } else if (channel.targetPath == AnimationChannel::ROTATION && !sampler.outputRotations.empty()) {
                        // Data is already in outputRotations
                    } else if (channel.targetPath == AnimationChannel::SCALE && !sampler.outputTranslations.empty()) {
                        // Move translation data to scale data
                        sampler.outputScales = sampler.outputTranslations;
                        sampler.outputTranslations.clear();
                    }
                }
            }

            animationClips.push_back(clip);
            std::cout << "Extracted animation: " << clip.name << " (duration: " << clip.duration << "s)" << std::endl;
        }

        return true;
    }

    bool loadAnimatedGLTFModel(VertexAttributes** d_attr,
                               unsigned int** d_indices,
                               unsigned int& attrSize,
                               unsigned int& indicesSize,
                               Skeleton& skeleton,
                               std::vector<AnimationClip>& animationClips,
                               const std::string& filename) {
        // Extract animated mesh data from GLTF
        std::vector<Float3> vertices;
        std::vector<unsigned int> indices;
        std::vector<Float2> texcoords;
        std::vector<Int4> jointIndices;
        std::vector<Float4> jointWeights;

        if (!extractAnimatedMeshFromGLTF(vertices, indices, texcoords, jointIndices, jointWeights,
                                         skeleton, animationClips, filename)) {
            attrSize = 0;
            indicesSize = 0;
            *d_attr = nullptr;
            *d_indices = nullptr;
            return false;
        }

        // Build CPU-side vertex attributes array with animation data
        std::vector<VertexAttributes> finalVertices(vertices.size());
        for (size_t i = 0; i < vertices.size(); ++i) {
            finalVertices[i].vertex = vertices[i];
            finalVertices[i].texcoord = (i < texcoords.size()) ? texcoords[i] : Float2(0.0f, 0.0f);
            finalVertices[i].jointIndices = (i < jointIndices.size()) ? jointIndices[i] : Int4(0, 0, 0, 0);
            finalVertices[i].jointWeights = (i < jointWeights.size()) ? jointWeights[i] : Float4(1.0f, 0.0f, 0.0f, 0.0f);
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