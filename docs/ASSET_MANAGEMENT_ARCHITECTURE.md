# Asset Management System Architecture

## Overview

The new asset management system provides a centralized, maintainable architecture for managing materials, textures, and models in the voxel engine. It replaces the previous scattered implementation with a clean, modular design.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Asset Definition Layer                   │
├───────────────────┬─────────────────┬───────────────────────┤
│   materials.yaml  │   models.yaml   │     blocks.yaml       │
│   (or hardcoded)  │  (or hardcoded) │    (or hardcoded)     │
└─────────┬─────────┴────────┬────────┴───────────┬───────────┘
          │                  │                     │
          ▼                  ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      AssetRegistry                           │
│  - Central repository for all asset definitions              │
│  - Loads from YAML or falls back to hardcoded               │
│  - Provides unified access to all asset metadata             │
└─────────┬──────────────────┬──────────────────┬─────────────┘
          │                  │                   │
          ▼                  ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ MaterialManager  │ │  ModelManager    │ │ TextureRegistry  │
├──────────────────┤ ├──────────────────┤ ├──────────────────┤
│ - GPU materials  │ │ - Geometry data  │ │ - Texture objects│
│ - Dynamic create │ │ - OBJ loading    │ │ - Mipmap gen     │
│ - Runtime update │ │ - GLTF loading   │ │ - Default tex    │
└──────────────────┘ └──────────────────┘ └──────────────────┘
          │                  │                   │
          └──────────────────┼───────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      OptixRenderer                           │
│  - Uses managers to get materials, models, textures          │
│  - No direct asset loading or management                     │
└──────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### 1. AssetRegistry (Singleton)
**Location:** `renderer/assets/AssetRegistry.h/cpp`

**Purpose:** Central repository for all asset definitions

**Key Features:**
- Loads asset definitions from YAML files (or hardcoded fallback)
- Maintains mappings between IDs and definitions
- Provides lookup functions for materials, models, and blocks
- Single source of truth for asset metadata

**API:**
```cpp
AssetRegistry& Get();  // Singleton access
bool loadFromYAML(const std::string& assetDirectory);
const MaterialDefinition* getMaterial(const std::string& id);
const ModelDefinition* getModel(const std::string& id);
const BlockDefinition* getBlock(int blockType);
```

### 2. MaterialManager (Singleton)
**Location:** `renderer/assets/MaterialManager.h/cpp`

**Purpose:** Manages GPU materials and dynamic material creation

**Key Features:**
- Creates GPU materials from asset definitions
- Supports dynamic material creation/update/destruction
- Maintains material indices for blocks and entities
- Handles material parameter uploads to GPU

**API:**
```cpp
MaterialManager& Get();
bool initialize();
MaterialParameter* getGPUMaterialsPointer();
unsigned int getMaterialIndexForBlock(int blockType);
unsigned int createDynamicMaterial(const MaterialProperties& props);
bool updateMaterial(unsigned int index);
```

### 3. TextureRegistry (Singleton)
**Location:** `renderer/assets/TextureRegistry.h/cpp`

**Purpose:** Centralized texture loading and management

**Key Features:**
- Loads textures from disk
- Generates mipmaps for power-of-two textures
- Creates default textures (white, normal, etc.)
- Manages CUDA texture objects
- Caches loaded textures to avoid duplicates

**API:**
```cpp
TextureRegistry& Get();
bool initialize();
TexObj getTexture(const std::string& filepath);
TexObj loadTexture(const std::string& filepath, bool generateMipmaps);
```

### 4. ModelManager (Singleton)
**Location:** `renderer/assets/ModelManager.h/cpp`

**Purpose:** Manages 3D model geometry loading

**Key Features:**
- Loads OBJ and GLTF models
- Converts indices to Int3 format for OptiX
- Manages GPU memory for geometry
- Maps models to blocks and entities

**API:**
```cpp
ModelManager& Get();
bool initialize();
const LoadedGeometry* getGeometryForBlock(int blockType);
const LoadedGeometry* getGeometryForEntity(int entityType);
```

## Data Flow

### Initialization Sequence
1. **AssetRegistry::loadFromYAML()** - Load asset definitions
2. **TextureRegistry::initialize()** - Load required textures
3. **MaterialManager::initialize()** - Create GPU materials
4. **ModelManager::initialize()** - Load model geometries
5. **OptixRenderer** uses managers to access assets

### Runtime Material Update
1. Application calls `MaterialManager::updateMaterial()`
2. MaterialManager updates CPU material copy
3. Material is uploaded to GPU
4. Next frame uses updated material

### Dynamic Material Creation
1. Application calls `MaterialManager::createDynamicMaterial()`
2. MaterialManager finds free slot or grows buffer
3. New material created and uploaded to GPU
4. Returns dynamic ID for future reference

## Asset Definition Format

### Materials (materials.yaml/json)
```yaml
materials:
  - id: "sand"
    name: "Sand Material"
    textures:
      albedo: "textures/rocky_trail_albedo.png"
      normal: "textures/rocky_trail_normal.png"
      roughness: "textures/rocky_trail_rough.png"
    properties:
      uv_scale: 2.5
      use_world_grid_uv: true
      roughness: 0.8
      metallic: 0.0
```

### Models (models.yaml/json)
```yaml
models:
  - id: "test_plane"
    name: "Test Plane Model"
    file: "models/test_plane.obj"
    type: "instanced"
    block_type: 13
```

### Blocks (blocks.yaml/json)
```yaml
blocks:
  - id: 1
    name: "Sand"
    type: "BlockTypeSand"
    material: "sand"
    is_instanced: false
```

## Benefits of New Architecture

### 1. **Centralized Management**
- All asset definitions in one place
- Easy to find and modify assets
- Clear ownership and responsibility

### 2. **Maintainability**
- Clean separation of concerns
- Each manager handles one aspect
- Easy to extend with new asset types

### 3. **Dynamic Capabilities**
- Runtime material creation/update
- Hot-reload support (future)
- Memory-efficient slot management

### 4. **Error Prevention**
- Type-safe interfaces
- Clear API boundaries
- Validation at load time

### 5. **Performance**
- Efficient GPU memory management
- Texture deduplication
- Optimized material updates

## Migration Guide

### For OptixRenderer
**Before:**
```cpp
// Materials were created inline
MaterialParameter parameter{};
parameter.textureAlbedo = textureManager.GetTexture("...");
m_materialParameters.push_back(parameter);
```

**After:**
```cpp
// Materials come from MaterialManager
m_systemParameter.materialParameters = MaterialManager::Get().getGPUMaterialsPointer();
unsigned int materialIndex = MaterialManager::Get().getMaterialIndexForBlock(blockType);
```

### For Entity Loading
**Before:**
```cpp
// Direct GLTF loading in Entity
GLTFUtils::loadAnimatedGLTFModel(&m_d_attributes, ...);
```

**After:**
```cpp
// Geometry comes from ModelManager
const LoadedGeometry* geometry = ModelManager::Get().getGeometryForEntity(entityType);
m_d_attributes = geometry->d_attributes;
```

## Future Enhancements

1. **Hot Reload Support**
   - Watch YAML files for changes
   - Reload assets without restart
   
2. **Asset Streaming**
   - Load assets on demand
   - Unload unused assets
   
3. **LOD Support**
   - Multiple detail levels per model
   - Automatic LOD selection
   
4. **Material Templates**
   - Inherit from base materials
   - Override specific properties
   
5. **Asset Validation**
   - Check texture dimensions
   - Validate material properties
   - Report missing assets