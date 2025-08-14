#include "BlockManager.h"
#include "util/DebugUtils.h"
#include <iostream>

namespace Assets {

bool BlockManager::initialize() {
    if (m_initialized) {
        return true;
    }

    // Get reference to AssetRegistry
    AssetRegistry& registry = AssetRegistry::Get();
    
    // Find the first instanced block type by looking for the first block with is_instanced = true
    m_firstInstancedBlockId = BlockTypeNum; // Default to end if none found
    for (unsigned int i = 0; i < BlockTypeNum; ++i) {
        const BlockDefinition* blockDef = registry.getBlockById(i);
        if (blockDef && blockDef->is_instanced) {
            m_firstInstancedBlockId = i;
            break;
        }
    }
    
    // Find the base light block
    m_hasBaseLightBlock = false;
    for (unsigned int i = 0; i < BlockTypeNum; ++i) {
        const BlockDefinition* blockDef = registry.getBlockById(i);
        if (blockDef && blockDef->is_base_light) {
            m_baseLightBlockId = i;
            m_hasBaseLightBlock = true;
            break;
        }
    }

    m_initialized = true;
    std::cout << "BlockManager initialized - First instanced block: " << m_firstInstancedBlockId 
              << ", Base light block: " << (m_hasBaseLightBlock ? std::to_string(m_baseLightBlockId) : "none") << std::endl;
    
    return true;
}

void BlockManager::cleanup() {
    m_initialized = false;
}

bool BlockManager::isInstancedBlockType(unsigned int blockId) const {
    return blockId >= m_firstInstancedBlockId;
}

bool BlockManager::isUninstancedBlockType(unsigned int blockId) const {
    return blockId < m_firstInstancedBlockId;
}

bool BlockManager::isTransparentBlockType(unsigned int blockId) const {
    // For now, no blocks are transparent - this could be extended with AssetRegistry data
    return false;
}

bool BlockManager::isBaseLightBlockType(unsigned int blockId) const {
    return m_hasBaseLightBlock && blockId == m_baseLightBlockId;
}

unsigned int BlockManager::getNumInstancedBlockTypes() const {
    return BlockTypeNum - m_firstInstancedBlockId;
}

unsigned int BlockManager::getNumUninstancedBlockTypes() const {
    return m_firstInstancedBlockId;
}

unsigned int BlockManager::getNumBlockTypes() const {
    return BlockTypeNum;
}

unsigned int BlockManager::getUninstancedObjectIdBegin() const {
    return 0;
}

unsigned int BlockManager::getUninstancedObjectIdEnd() const {
    return m_firstInstancedBlockId - 1;
}

unsigned int BlockManager::getInstancedObjectIdBegin() const {
    return m_firstInstancedBlockId - 1;
}

unsigned int BlockManager::getInstancedObjectIdEnd() const {
    return BlockTypeNum - 1;
}

unsigned int BlockManager::objectIdToBlockId(unsigned int objectId) const {
    return objectId + 1;
}

unsigned int BlockManager::blockIdToObjectId(unsigned int blockId) const {
    return blockId - 1;
}

bool BlockManager::hasModel(unsigned int blockId) const {
    const BlockDefinition* blockDef = AssetRegistry::Get().getBlockById(blockId);
    return blockDef && blockDef->model_id.has_value() && !blockDef->model_id->empty() && blockDef->model_id != "null";
}

bool BlockManager::isEmissive(unsigned int blockId) const {
    const BlockDefinition* blockDef = AssetRegistry::Get().getBlockById(blockId);
    if (!blockDef) return false;
    
    // Check if block itself is marked as emissive
    if (blockDef->is_emissive) return true;
    
    // Check if the material is emissive
    const MaterialDefinition* materialDef = AssetRegistry::Get().getMaterialForBlock(blockId);
    return materialDef && materialDef->properties.is_emissive;
}

const std::string* BlockManager::getModelName(unsigned int blockId) const {
    const BlockDefinition* blockDef = AssetRegistry::Get().getBlockById(blockId);
    if (blockDef && blockDef->model_id.has_value() && !blockDef->model_id->empty() && blockDef->model_id != "null") {
        return &(*blockDef->model_id);
    }
    return nullptr;
}

const std::string* BlockManager::getMaterialName(unsigned int blockId) const {
    const BlockDefinition* blockDef = AssetRegistry::Get().getBlockById(blockId);
    if (blockDef && blockDef->material_id.has_value() && !blockDef->material_id->empty() && blockDef->material_id != "null") {
        return &(*blockDef->material_id);
    }
    return nullptr;
}

} // namespace Assets