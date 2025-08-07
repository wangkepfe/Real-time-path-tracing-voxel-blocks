#pragma once

#include "shaders/SystemParameter.h"
#include <vector>
#include <string>
#include <cuda_runtime.h>

/**
 * ChunkGeometryBuffer - Professional dynamic geometry buffer management
 * Similar to std::vector but for GPU-allocated geometry data
 * Handles dynamic resizing, efficient memory management, and face allocation
 */
class ChunkGeometryBuffer
{
public:
    ChunkGeometryBuffer();
    ~ChunkGeometryBuffer();
    
    // Non-copyable but movable
    ChunkGeometryBuffer(const ChunkGeometryBuffer&) = delete;
    ChunkGeometryBuffer& operator=(const ChunkGeometryBuffer&) = delete;
    ChunkGeometryBuffer(ChunkGeometryBuffer&& other) noexcept;
    ChunkGeometryBuffer& operator=(ChunkGeometryBuffer&& other) noexcept;
    
    // Initialization and management
    void initialize(unsigned int initialCapacity = 64);
    void clear();
    void reset(); // Reset counters but keep capacity
    
    // Capacity management
    void reserve(unsigned int newCapacity);
    bool ensureCapacity(unsigned int requiredFaces);
    
    // Face allocation and management
    unsigned int allocateFace();
    void deallocateFace(unsigned int faceIndex);
    bool hasFreeSlots() const { return !m_freeSlots.empty() || m_currentFaceCount < m_capacity; }
    
    // Getters
    VertexAttributes* getVertexBuffer() const { return m_d_vertices; }
    unsigned int* getIndexBuffer() const { return m_d_indices; }
    unsigned int getCurrentFaceCount() const { return m_currentFaceCount; }
    unsigned int getCapacity() const { return m_capacity; }
    unsigned int getVertexCount() const { return m_currentFaceCount * 4; }
    unsigned int getIndexCount() const { return m_currentFaceCount * 6; }
    size_t getMemoryUsage() const;
    
    // Statistics and debugging
    void printStats(const std::string& name = "") const;
    bool isValid() const { return m_d_vertices != nullptr && m_d_indices != nullptr; }

private:
    // GPU buffers
    VertexAttributes* m_d_vertices;
    unsigned int* m_d_indices;
    
    // Capacity and allocation tracking
    unsigned int m_capacity;           // Total allocated capacity (in faces)
    unsigned int m_currentFaceCount;   // Current number of faces in use
    
    // Free slot management (similar to memory pool)
    std::vector<unsigned int> m_freeSlots;
    
    // Growth parameters
    static constexpr unsigned int MIN_CAPACITY = 32;
    static constexpr unsigned int MAX_CAPACITY = 1000000; // 1M faces max
    static constexpr float GROWTH_FACTOR = 1.5f;
    
    // Internal helpers
    void reallocate(unsigned int newCapacity);
    void deallocateBuffers();
    unsigned int calculateGrowthCapacity(unsigned int requiredCapacity) const;
};

/**
 * ChunkGeometryManager - Manages geometry buffers for all chunks and objects
 * Provides high-level interface for the VoxelEngine
 */
class ChunkGeometryManager
{
public:
    ChunkGeometryManager();
    ~ChunkGeometryManager();
    
    // Initialization
    void initialize(unsigned int numChunks, unsigned int numObjectTypes);
    void shutdown();
    
    // Buffer access
    ChunkGeometryBuffer& getBuffer(unsigned int chunkIndex, unsigned int objectId);
    const ChunkGeometryBuffer& getBuffer(unsigned int chunkIndex, unsigned int objectId) const;
    
    // Convenience methods
    VertexAttributes* getVertices(unsigned int chunkIndex, unsigned int objectId);
    unsigned int* getIndices(unsigned int chunkIndex, unsigned int objectId);
    unsigned int getCurrentFaceCount(unsigned int chunkIndex, unsigned int objectId) const;
    unsigned int getCapacity(unsigned int chunkIndex, unsigned int objectId) const;
    
    // Face allocation
    unsigned int allocateFace(unsigned int chunkIndex, unsigned int objectId);
    void deallocateFace(unsigned int chunkIndex, unsigned int objectId, unsigned int faceIndex);
    bool ensureCapacity(unsigned int chunkIndex, unsigned int objectId, unsigned int requiredFaces);
    
    // Statistics
    size_t getTotalMemoryUsage() const;
    void printAllStats() const;

private:
    std::vector<std::vector<ChunkGeometryBuffer>> m_buffers; // [chunkIndex][objectId]
    unsigned int m_numChunks;
    unsigned int m_numObjectTypes;
    bool m_initialized;
};