#pragma once

#include "voxelengine/BlockType.h"
#include "AssetRegistry.h"

namespace Assets {

/**
 * BlockManager provides centralized logic for block type classification and management.
 * This class handles all block-related queries and operations, making it easier to
 * maintain block logic in one place rather than scattered inline functions.
 */
class BlockManager {
public:
    // Singleton pattern
    static BlockManager& Get() {
        static BlockManager instance;
        return instance;
    }
    
    BlockManager(const BlockManager&) = delete;
    void operator=(const BlockManager&) = delete;

    // Initialize the block manager with asset registry data
    bool initialize();
    void cleanup();

    // Block type classification
    bool isInstancedBlockType(unsigned int blockId) const;
    bool isUninstancedBlockType(unsigned int blockId) const;
    bool isTransparentBlockType(unsigned int blockId) const;
    bool isBaseLightBlockType(unsigned int blockId) const;

    // Block type counts
    unsigned int getNumInstancedBlockTypes() const;
    unsigned int getNumUninstancedBlockTypes() const;
    unsigned int getNumBlockTypes() const;

    // Object ID mapping functions
    unsigned int getUninstancedObjectIdBegin() const;
    unsigned int getUninstancedObjectIdEnd() const;
    unsigned int getInstancedObjectIdBegin() const;
    unsigned int getInstancedObjectIdEnd() const;
    
    unsigned int objectIdToBlockId(unsigned int objectId) const;
    unsigned int blockIdToObjectId(unsigned int blockId) const;

    // Advanced block queries using AssetRegistry
    bool hasModel(unsigned int blockId) const;
    bool isEmissive(unsigned int blockId) const;
    const std::string* getModelName(unsigned int blockId) const;
    const std::string* getMaterialName(unsigned int blockId) const;

private:
    BlockManager() = default;
    ~BlockManager() { cleanup(); }

    bool m_initialized = false;
    
    // Cached values for performance
    unsigned int m_firstInstancedBlockId = 0;
    unsigned int m_baseLightBlockId = 0;
    bool m_hasBaseLightBlock = false;
};

} // namespace Assets