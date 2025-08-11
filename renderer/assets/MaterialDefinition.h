#pragma once

#include <string>
#include <vector>
#include <optional>
#include "shaders/LinearMath.h"

namespace Assets {

struct MaterialTextures {
    std::optional<std::string> albedo;
    std::optional<std::string> normal;
    std::optional<std::string> roughness;
    std::optional<std::string> metallic;
    std::optional<std::string> emissive;
};

struct MaterialProperties {
    Float3 albedo = Float3(1.0f, 1.0f, 1.0f);
    float roughness = 0.5f;
    float metallic = 0.0f;
    float uv_scale = 1.0f;
    float translucency = 0.0f;
    bool use_world_grid_uv = false;
    bool is_thinfilm = false;
    bool is_emissive = false;
    Float3 emissive_radiance = Float3(0.0f, 0.0f, 0.0f);
};

struct MaterialDefinition {
    std::string id;
    std::string name;
    MaterialTextures textures;
    MaterialProperties properties;
    
    unsigned int runtimeIndex = 0;  // Runtime index in GPU material array
};

struct ModelDefinition {
    std::string id;
    std::string name;
    std::string file;
    std::string type;  // "instanced" or "entity"
    int block_type = -1;
    int entity_type = -1;
    bool has_animation = false;
    
    void* runtimeGeometry = nullptr;  // Runtime pointer to loaded geometry
};

struct BlockDefinition {
    int id;
    std::string name;
    std::string type;  // BlockType enum name
    std::optional<std::string> material_id;
    std::optional<std::string> model_id;
    bool is_instanced = false;
    bool is_transparent = false;
    bool is_base_light = false;
    bool is_emissive = false;
};

} // namespace Assets