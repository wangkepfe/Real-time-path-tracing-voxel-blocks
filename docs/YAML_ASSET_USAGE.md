# YAML Asset Definition Guide

## Overview

The asset management system supports YAML-based configuration files for defining materials, models, and blocks. If YAML files are not present, the system falls back to hardcoded definitions.

## File Locations

All asset definition files should be placed in: `data/assets/`

- `materials.yaml` - Material definitions
- `models.yaml` - 3D model definitions  
- `blocks.yaml` - Block type definitions

## Material Definition

### Basic Structure
```yaml
materials:
  - id: "unique_material_id"       # Required: Unique identifier
    name: "Human Readable Name"    # Required: Display name
    textures:                       # Optional: Texture paths
      albedo: "path/to/albedo.png"
      normal: "path/to/normal.png"
      roughness: "path/to/rough.png"
      metallic: "path/to/metal.png"
      emissive: "path/to/emissive.png"
    properties:                     # Optional: Material properties
      albedo: [1.0, 1.0, 1.0]      # RGB values (0-1)
      roughness: 0.5                # 0=smooth, 1=rough
      metallic: 0.0                 # 0=dielectric, 1=metal
      uv_scale: 1.0                 # Texture coordinate scale
      translucency: 0.0             # 0=opaque, 1=translucent
      use_world_grid_uv: false     # Use world-space UVs
      is_thinfilm: false            # Thin film interference
      is_emissive: false            # Emissive material
      emissive_radiance: [0, 0, 0] # Emission color/intensity
```

### Material Examples

#### Simple Textured Material
```yaml
- id: "wood"
  name: "Wood Material"
  textures:
    albedo: "textures/wood_albedo.png"
    normal: "textures/wood_normal.png"
    roughness: "textures/wood_rough.png"
  properties:
    uv_scale: 2.0
    roughness: 0.7
```

#### Emissive Material
```yaml
- id: "lamp"
  name: "Lamp Light"
  properties:
    is_emissive: true
    emissive_radiance: [10.0, 8.0, 6.0]  # Warm light
```

#### Metallic Material
```yaml
- id: "gold"
  name: "Gold Material"
  textures:
    albedo: "textures/gold_albedo.png"
    metallic: "textures/gold_metallic.png"
  properties:
    metallic: 1.0
    roughness: 0.3
```

#### Translucent Material
```yaml
- id: "leaves"
  name: "Tree Leaves"
  textures:
    albedo: "textures/leaves_albedo.png"
  properties:
    translucency: 0.5
    is_thinfilm: true
    roughness: 0.5
```

## Model Definition

### Basic Structure
```yaml
models:
  - id: "unique_model_id"          # Required: Unique identifier
    name: "Human Readable Name"    # Required: Display name
    file: "models/model.obj"       # Required: Path to model file
    type: "instanced"               # Required: "instanced" or "entity"
    block_type: 13                  # Optional: Associated block type ID
    entity_type: 0                  # Optional: Associated entity type ID
    has_animation: false            # Optional: Has skeletal animation
```

### Model Examples

#### Static Block Model
```yaml
- id: "cube"
  name: "Basic Cube"
  file: "models/cube.obj"
  type: "instanced"
  block_type: 1
```

#### Animated Entity Model
```yaml
- id: "character"
  name: "Player Character"
  file: "models/character.gltf"
  type: "entity"
  entity_type: 0
  has_animation: true
```

## Block Definition

### Basic Structure
```yaml
blocks:
  - id: 1                           # Required: Block ID (matches enum)
    name: "Block Name"              # Required: Display name
    type: "BlockTypeEnum"           # Required: Block type enum name
    material: "material_id"         # Optional: Material to use
    model: "model_id"               # Optional: Model to use
    is_instanced: false             # Optional: Uses instancing
    is_transparent: false           # Optional: Transparent block
    is_base_light: false            # Optional: Light base block
    is_emissive: false              # Optional: Emits light
```

### Block Examples

#### Simple Solid Block
```yaml
- id: 1
  name: "Stone"
  type: "BlockTypeStone"
  material: "stone"
```

#### Instanced Model Block
```yaml
- id: 14
  name: "Leaves"
  type: "BlockTypeLeaves"
  material: "leaves"
  model: "leaves_cube"
  is_instanced: true
```

#### Emissive Block
```yaml
- id: 16
  name: "Lamp"
  type: "BlockTypeLamp"
  material: "lamp_light"
  model: "lamp_model"
  is_instanced: true
  is_emissive: true
```

## Complete Example Files

### materials.yaml
```yaml
materials:
  # Terrain materials
  - id: "grass"
    name: "Grass Block"
    textures:
      albedo: "textures/grass_albedo.png"
      normal: "textures/grass_normal.png"
    properties:
      uv_scale: 4.0
      use_world_grid_uv: true
      
  - id: "stone"
    name: "Stone Block"
    textures:
      albedo: "textures/stone_albedo.png"
      normal: "textures/stone_normal.png"
      roughness: "textures/stone_rough.png"
    properties:
      uv_scale: 2.0
      use_world_grid_uv: true
      roughness: 0.9
      
  # Special materials
  - id: "water"
    name: "Water"
    properties:
      albedo: [0.2, 0.4, 0.8]
      roughness: 0.0
      translucency: 0.8
      is_thinfilm: true
      
  - id: "torch"
    name: "Torch Light"
    properties:
      is_emissive: true
      emissive_radiance: [15.0, 10.0, 5.0]
```

### models.yaml
```yaml
models:
  - id: "grass_model"
    name: "Grass Block Model"
    file: "models/grass_block.obj"
    type: "instanced"
    block_type: 1
    
  - id: "torch_model"
    name: "Torch Model"
    file: "models/torch.obj"
    type: "instanced"
    block_type: 10
    
  - id: "player_model"
    name: "Player Character"
    file: "models/steve.gltf"
    type: "entity"
    entity_type: 0
    has_animation: true
```

### blocks.yaml
```yaml
blocks:
  - id: 0
    name: "Air"
    type: "BlockTypeEmpty"
    
  - id: 1
    name: "Grass"
    type: "BlockTypeGrass"
    material: "grass"
    model: "grass_model"
    is_instanced: true
    
  - id: 2
    name: "Stone"
    type: "BlockTypeStone"
    material: "stone"
    
  - id: 10
    name: "Torch"
    type: "BlockTypeTorch"
    material: "torch"
    model: "torch_model"
    is_instanced: true
    is_emissive: true
```

## Tips and Best Practices

### 1. Naming Conventions
- Use lowercase with underscores for IDs: `stone_brick`
- Use descriptive names: `mossy_cobblestone` instead of `block_3`
- Keep IDs consistent across files

### 2. Texture Organization
- Store textures in `data/textures/`
- Use consistent naming: `{material}_albedo.png`, `{material}_normal.png`
- Keep texture resolutions power-of-two for mipmapping

### 3. Material Properties
- Start with default values and adjust
- Test roughness values in-game
- Use world_grid_uv for terrain blocks

### 4. Performance
- Reuse materials when possible
- Minimize unique texture count
- Use instancing for repeated objects

### 5. Validation
- Check console output for loading errors
- Verify all referenced files exist
- Test materials in different lighting

## Troubleshooting

### Material Not Loading
- Check file paths are relative to project root
- Verify texture files exist
- Check for typos in material IDs

### Black or Missing Textures
- Ensure texture format is supported (PNG, JPG)
- Check texture file isn't corrupted
- Verify paths don't contain backslashes on Unix

### Incorrect Material Assignment
- Verify block references correct material ID
- Check material ID is unique
- Ensure blocks.yaml references are correct

## Advanced Features

### Dynamic Material Updates
Materials can be updated at runtime using the MaterialManager API:
```cpp
auto& manager = MaterialManager::Get();
manager.updateMaterial("wood");  // Reload from definition
```

### Custom UV Mapping
Use `use_world_grid_uv: true` for terrain that should tile seamlessly across blocks.

### Emissive Materials
Emissive materials contribute to global illumination. Adjust `emissive_radiance` for intensity.

## JSON Alternative

The system also supports JSON format if preferred:
```json
{
  "materials": [
    {
      "id": "stone",
      "name": "Stone Material",
      "textures": {
        "albedo": "textures/stone_albedo.png"
      },
      "properties": {
        "roughness": 0.8
      }
    }
  ]
}
```