# Runtime Material Update API Examples

## Overview

The new asset management system provides APIs for dynamically creating, updating, and destroying materials at runtime. This allows for dynamic visual effects, material property animation, and runtime customization.

## Basic Usage

### 1. Access the Material Manager

```cpp
#include "assets/MaterialManager.h"

auto& materialManager = Assets::MaterialManager::Get();
```

### 2. Update an Existing Material

```cpp
// Update a material's properties (e.g., make it emissive)
MaterialProperties newProps;
newProps.isEmissive = true;
newProps.emissiveRadiance = Float3(10.0f, 5.0f, 2.0f); // Orange glow

// Update the material by name
bool success = materialManager.updateMaterial("torch");
if (success) {
    std::cout << "Torch material now emissive!" << std::endl;
}
```

### 3. Create a Dynamic Material

```cpp
// Create a new material at runtime
MaterialProperties dynamicProps;
dynamicProps.albedo = Float3(0.8f, 0.2f, 0.2f); // Red color
dynamicProps.roughness = 0.5f;
dynamicProps.metallic = 0.8f;

unsigned int dynamicId = materialManager.createDynamicMaterial(dynamicProps);
if (dynamicId != UINT_MAX) {
    std::cout << "Created dynamic material with ID: " << dynamicId << std::endl;
}
```

### 4. Update a Dynamic Material

```cpp
// Animate material properties over time
MaterialProperties animatedProps;
float time = GlobalSettings::GetGameTime();
animatedProps.albedo = Float3(
    0.5f + 0.5f * sin(time),
    0.5f + 0.5f * sin(time + 2.0f),
    0.5f + 0.5f * sin(time + 4.0f)
);
animatedProps.roughness = 0.5f + 0.3f * sin(time * 2.0f);

bool updated = materialManager.updateDynamicMaterial(dynamicId, animatedProps);
```

### 5. Destroy a Dynamic Material

```cpp
// Clean up when no longer needed
materialManager.destroyDynamicMaterial(dynamicId);
```

## Advanced Examples

### Example 1: Temperature-Based Material Animation

```cpp
class LavaMaterialController {
private:
    unsigned int lavaMaterialId;
    Assets::MaterialManager& manager;
    
public:
    LavaMaterialController() : manager(Assets::MaterialManager::Get()) {
        // Create initial lava material
        MaterialProperties lavaProps;
        lavaProps.isEmissive = true;
        lavaProps.emissiveRadiance = Float3(5.0f, 1.0f, 0.0f);
        lavaProps.albedo = Float3(0.8f, 0.2f, 0.0f);
        lavaProps.roughness = 0.9f;
        
        lavaMaterialId = manager.createDynamicMaterial(lavaProps);
    }
    
    void update(float temperature) {
        // Map temperature (0-1) to material properties
        MaterialProperties props;
        
        // Hotter = brighter and more yellow/white
        float intensity = 5.0f + temperature * 20.0f;
        props.emissiveRadiance = Float3(
            intensity,
            intensity * (0.3f + 0.7f * temperature),
            intensity * temperature * 0.5f
        );
        
        // Hotter = less rough (more molten)
        props.roughness = 0.9f - temperature * 0.4f;
        
        // Update color from dark red to bright orange
        props.albedo = Float3(
            0.3f + temperature * 0.7f,
            0.1f + temperature * 0.6f,
            temperature * 0.3f
        );
        
        props.isEmissive = true;
        
        manager.updateDynamicMaterial(lavaMaterialId, props);
    }
};
```

### Example 2: Day/Night Cycle Material Updates

```cpp
class DayNightMaterialSystem {
private:
    Assets::MaterialManager& manager;
    
public:
    DayNightMaterialSystem() : manager(Assets::MaterialManager::Get()) {}
    
    void updateTimeOfDay(float timeNormalized) {
        // timeNormalized: 0 = midnight, 0.5 = noon, 1 = midnight
        
        // Update window materials to be emissive at night
        if (timeNormalized < 0.25f || timeNormalized > 0.75f) {
            // Nighttime - windows glow
            float nightIntensity = 1.0f - abs(timeNormalized - 0.5f) * 2.0f;
            
            MaterialProperties windowProps;
            windowProps.isEmissive = true;
            windowProps.emissiveRadiance = Float3(2.0f, 1.8f, 1.0f) * nightIntensity;
            windowProps.translucency = 0.8f;
            
            manager.updateMaterial("window_glass");
        } else {
            // Daytime - windows are transparent
            MaterialProperties windowProps;
            windowProps.isEmissive = false;
            windowProps.translucency = 0.9f;
            windowProps.roughness = 0.0f;
            
            manager.updateMaterial("window_glass");
        }
        
        // Update lamp materials
        if (timeNormalized < 0.3f || timeNormalized > 0.7f) {
            // Lamps on during dawn/dusk/night
            MaterialProperties lampProps;
            lampProps.isEmissive = true;
            lampProps.emissiveRadiance = Float3(15.0f, 12.0f, 8.0f);
            
            manager.updateMaterial("street_lamp");
        } else {
            // Lamps off during day
            MaterialProperties lampProps;
            lampProps.isEmissive = false;
            lampProps.albedo = Float3(0.7f, 0.7f, 0.6f);
            
            manager.updateMaterial("street_lamp");
        }
    }
};
```

### Example 3: Interactive Material Selection

```cpp
class MaterialPainter {
private:
    Assets::MaterialManager& manager;
    std::vector<unsigned int> customMaterials;
    unsigned int activeBrush;
    
public:
    MaterialPainter() : manager(Assets::MaterialManager::Get()), activeBrush(0) {
        // Create a palette of custom materials
        createPalette();
    }
    
    void createPalette() {
        // Create metallic gold
        MaterialProperties gold;
        gold.albedo = Float3(1.0f, 0.84f, 0.0f);
        gold.metallic = 1.0f;
        gold.roughness = 0.3f;
        customMaterials.push_back(manager.createDynamicMaterial(gold));
        
        // Create glass
        MaterialProperties glass;
        glass.albedo = Float3(0.95f, 0.95f, 0.95f);
        glass.roughness = 0.0f;
        glass.translucency = 0.95f;
        customMaterials.push_back(manager.createDynamicMaterial(glass));
        
        // Create glowing crystal
        MaterialProperties crystal;
        crystal.albedo = Float3(0.3f, 0.8f, 1.0f);
        crystal.isEmissive = true;
        crystal.emissiveRadiance = Float3(2.0f, 5.0f, 8.0f);
        crystal.translucency = 0.5f;
        customMaterials.push_back(manager.createDynamicMaterial(crystal));
    }
    
    void selectBrush(int brushIndex) {
        if (brushIndex < customMaterials.size()) {
            activeBrush = brushIndex;
        }
    }
    
    unsigned int getActiveMaterial() const {
        return customMaterials[activeBrush];
    }
    
    void paintBlock(int blockType) {
        // This would integrate with the block system to apply the material
        // Example: Block::setMaterialOverride(blockType, customMaterials[activeBrush]);
    }
    
    ~MaterialPainter() {
        // Clean up dynamic materials
        for (auto id : customMaterials) {
            manager.destroyDynamicMaterial(id);
        }
    }
};
```

## Integration with Rendering

The material updates are automatically synchronized with the GPU before each frame render. The OptixRenderer uses the material manager's GPU pointer directly:

```cpp
// In OptixRenderer::init()
m_systemParameter.materialParameters = MaterialManager::Get().getGPUMaterialsPointer();

// Materials are automatically updated each frame through the pointer
```

## Performance Considerations

1. **Batch Updates**: When updating multiple materials, batch them together before rendering
2. **Update Frequency**: Avoid updating materials every frame unless necessary
3. **Dynamic Slots**: The system pre-allocates slots for dynamic materials to avoid reallocation
4. **Memory Management**: Remember to destroy dynamic materials when no longer needed

## Thread Safety

The MaterialManager is not thread-safe by default. All material updates should be performed from the main thread or properly synchronized.

## Debugging

Enable verbose logging to track material operations:

```cpp
// In MaterialManager initialization
materialManager.setVerboseLogging(true);
```

This will log all material create/update/destroy operations for debugging.