#include "ChunkGeometryBuffer.h"
#include <iostream>
#include <algorithm>
#include <cassert>
#include <string>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)

//=============================================================================
// ChunkGeometryBuffer Implementation
//=============================================================================

ChunkGeometryBuffer::ChunkGeometryBuffer()
    : m_d_vertices(nullptr)
    , m_d_indices(nullptr)
    , m_capacity(0)
    , m_currentFaceCount(0)
{
}

ChunkGeometryBuffer::~ChunkGeometryBuffer()
{
    deallocateBuffers();
}

ChunkGeometryBuffer::ChunkGeometryBuffer(ChunkGeometryBuffer&& other) noexcept
    : m_d_vertices(other.m_d_vertices)
    , m_d_indices(other.m_d_indices)
    , m_capacity(other.m_capacity)
    , m_currentFaceCount(other.m_currentFaceCount)
    , m_freeSlots(std::move(other.m_freeSlots))
{
    // Reset the moved-from object
    other.m_d_vertices = nullptr;
    other.m_d_indices = nullptr;
    other.m_capacity = 0;
    other.m_currentFaceCount = 0;
}

ChunkGeometryBuffer& ChunkGeometryBuffer::operator=(ChunkGeometryBuffer&& other) noexcept
{
    if (this != &other) {
        // Clean up existing resources
        deallocateBuffers();
        
        // Move resources
        m_d_vertices = other.m_d_vertices;
        m_d_indices = other.m_d_indices;
        m_capacity = other.m_capacity;
        m_currentFaceCount = other.m_currentFaceCount;
        m_freeSlots = std::move(other.m_freeSlots);
        
        // Reset the moved-from object
        other.m_d_vertices = nullptr;
        other.m_d_indices = nullptr;
        other.m_capacity = 0;
        other.m_currentFaceCount = 0;
    }
    return *this;
}

void ChunkGeometryBuffer::initialize(unsigned int initialCapacity)
{
    // Ensure minimum capacity
    initialCapacity = std::max(initialCapacity, MIN_CAPACITY);
    
    deallocateBuffers(); // Clean up any existing buffers
    
    m_capacity = initialCapacity;
    m_currentFaceCount = 0;
    m_freeSlots.clear();
    
    // Allocate GPU buffers
    CUDA_CHECK(cudaMalloc(&m_d_vertices, m_capacity * 4 * sizeof(VertexAttributes)));
    CUDA_CHECK(cudaMalloc(&m_d_indices, m_capacity * 6 * sizeof(unsigned int)));
    
    printf("GEOMETRY BUFFER: Initialized with capacity %u faces (%zu bytes)\n", 
           m_capacity, getMemoryUsage());
}

void ChunkGeometryBuffer::clear()
{
    m_currentFaceCount = 0;
    m_freeSlots.clear();
}

void ChunkGeometryBuffer::reset()
{
    clear();
    // Keep capacity and buffers allocated
}

void ChunkGeometryBuffer::reserve(unsigned int newCapacity)
{
    if (newCapacity > m_capacity) {
        reallocate(newCapacity);
    }
}

bool ChunkGeometryBuffer::ensureCapacity(unsigned int requiredFaces)
{
    unsigned int availableSlots = m_freeSlots.size() + (m_capacity - m_currentFaceCount);
    
    if (requiredFaces <= availableSlots) {
        return true; // Already have enough capacity
    }
    
    // Need to grow
    unsigned int newCapacity = calculateGrowthCapacity(m_currentFaceCount + requiredFaces);
    
    if (newCapacity > MAX_CAPACITY) {
        printf("GEOMETRY BUFFER WARNING: Requested capacity %u exceeds maximum %u\n", 
               newCapacity, MAX_CAPACITY);
        return false;
    }
    
    reallocate(newCapacity);
    return true;
}

unsigned int ChunkGeometryBuffer::allocateFace()
{
    // Try to reuse a free slot first
    if (!m_freeSlots.empty()) {
        unsigned int faceIndex = m_freeSlots.back();
        m_freeSlots.pop_back();
        return faceIndex;
    }
    
    // Check if we need to grow
    if (m_currentFaceCount >= m_capacity) {
        if (!ensureCapacity(1)) {
            printf("GEOMETRY BUFFER ERROR: Failed to allocate face - out of capacity\n");
            return UINT_MAX; // Invalid index
        }
    }
    
    // Allocate new face
    unsigned int faceIndex = m_currentFaceCount;
    m_currentFaceCount++;
    
    return faceIndex;
}

void ChunkGeometryBuffer::deallocateFace(unsigned int faceIndex)
{
    if (faceIndex >= m_capacity) {
        printf("GEOMETRY BUFFER ERROR: Invalid face index %u for deallocation\n", faceIndex);
        return;
    }
    
    // Add to free slots for reuse
    m_freeSlots.push_back(faceIndex);
}

size_t ChunkGeometryBuffer::getMemoryUsage() const
{
    return m_capacity * (4 * sizeof(VertexAttributes) + 6 * sizeof(unsigned int));
}

void ChunkGeometryBuffer::printStats(const std::string& name) const
{
    printf("GEOMETRY BUFFER %s: capacity=%u, used=%u, free_slots=%zu, memory=%.2f KB\n",
           name.c_str(), m_capacity, m_currentFaceCount, m_freeSlots.size(),
           getMemoryUsage() / 1024.0f);
}

void ChunkGeometryBuffer::deallocateBuffers()
{
    if (m_d_vertices) {
        cudaError_t result = cudaFree(m_d_vertices);
        if (result != cudaSuccess) {
            // Don't use CUDA_CHECK macro during shutdown to avoid crashes
            printf("WARNING: ChunkGeometryBuffer cleanup - failed to free vertices: %s\n", 
                   cudaGetErrorString(result));
        }
        m_d_vertices = nullptr;
    }
    if (m_d_indices) {
        cudaError_t result = cudaFree(m_d_indices);
        if (result != cudaSuccess) {
            // Don't use CUDA_CHECK macro during shutdown to avoid crashes
            printf("WARNING: ChunkGeometryBuffer cleanup - failed to free indices: %s\n", 
                   cudaGetErrorString(result));
        }
        m_d_indices = nullptr;
    }
    m_capacity = 0;
    m_currentFaceCount = 0;
    m_freeSlots.clear();
}

void ChunkGeometryBuffer::reallocate(unsigned int newCapacity)
{
    if (newCapacity <= m_capacity) {
        return; // No need to reallocate
    }
    
    printf("GEOMETRY BUFFER: Growing from %u to %u faces\n", m_capacity, newCapacity);
    
    // Allocate new buffers
    VertexAttributes* new_d_vertices;
    unsigned int* new_d_indices;
    
    CUDA_CHECK(cudaMalloc(&new_d_vertices, newCapacity * 4 * sizeof(VertexAttributes)));
    CUDA_CHECK(cudaMalloc(&new_d_indices, newCapacity * 6 * sizeof(unsigned int)));
    
    // Copy existing data if we have any
    if (m_d_vertices && m_currentFaceCount > 0) {
        CUDA_CHECK(cudaMemcpy(new_d_vertices, m_d_vertices, 
                             m_currentFaceCount * 4 * sizeof(VertexAttributes), 
                             cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(new_d_indices, m_d_indices,
                             m_currentFaceCount * 6 * sizeof(unsigned int),
                             cudaMemcpyDeviceToDevice));
    }
    
    // Free old buffers
    deallocateBuffers();
    
    // Update to new buffers
    m_d_vertices = new_d_vertices;
    m_d_indices = new_d_indices;
    m_capacity = newCapacity;
}

unsigned int ChunkGeometryBuffer::calculateGrowthCapacity(unsigned int requiredCapacity) const
{
    // Similar to std::vector growth strategy
    unsigned int growthCapacity = static_cast<unsigned int>(m_capacity * GROWTH_FACTOR);
    return std::max(requiredCapacity, growthCapacity);
}

//=============================================================================
// ChunkGeometryManager Implementation  
//=============================================================================

ChunkGeometryManager::ChunkGeometryManager()
    : m_numChunks(0)
    , m_numObjectTypes(0)
    , m_initialized(false)
{
}

ChunkGeometryManager::~ChunkGeometryManager()
{
    shutdown();
}

void ChunkGeometryManager::initialize(unsigned int numChunks, unsigned int numObjectTypes)
{
    shutdown(); // Clean up existing state
    
    m_numChunks = numChunks;
    m_numObjectTypes = numObjectTypes;
    
    // Resize the 2D buffer array
    m_buffers.resize(numChunks);
    for (unsigned int chunkIndex = 0; chunkIndex < numChunks; ++chunkIndex) {
        m_buffers[chunkIndex].resize(numObjectTypes);
        
        // Initialize each buffer with a reasonable starting capacity
        for (unsigned int objectId = 0; objectId < numObjectTypes; ++objectId) {
            m_buffers[chunkIndex][objectId].initialize(64); // Start with 64 faces capacity
        }
    }
    
    m_initialized = true;
    printf("GEOMETRY MANAGER: Initialized %u chunks x %u objects = %u buffers\n",
           numChunks, numObjectTypes, numChunks * numObjectTypes);
}

void ChunkGeometryManager::shutdown()
{
    m_buffers.clear();
    m_numChunks = 0;
    m_numObjectTypes = 0;
    m_initialized = false;
}

ChunkGeometryBuffer& ChunkGeometryManager::getBuffer(unsigned int chunkIndex, unsigned int objectId)
{
    assert(m_initialized && "GeometryManager not initialized");
    assert(chunkIndex < m_numChunks && "Invalid chunk index");
    assert(objectId < m_numObjectTypes && "Invalid object ID");
    
    return m_buffers[chunkIndex][objectId];
}

const ChunkGeometryBuffer& ChunkGeometryManager::getBuffer(unsigned int chunkIndex, unsigned int objectId) const
{
    assert(m_initialized && "GeometryManager not initialized");
    assert(chunkIndex < m_numChunks && "Invalid chunk index");
    assert(objectId < m_numObjectTypes && "Invalid object ID");
    
    return m_buffers[chunkIndex][objectId];
}

VertexAttributes* ChunkGeometryManager::getVertices(unsigned int chunkIndex, unsigned int objectId)
{
    return getBuffer(chunkIndex, objectId).getVertexBuffer();
}

unsigned int* ChunkGeometryManager::getIndices(unsigned int chunkIndex, unsigned int objectId)
{
    return getBuffer(chunkIndex, objectId).getIndexBuffer();
}

unsigned int ChunkGeometryManager::getCurrentFaceCount(unsigned int chunkIndex, unsigned int objectId) const
{
    return getBuffer(chunkIndex, objectId).getCurrentFaceCount();
}

unsigned int ChunkGeometryManager::getCapacity(unsigned int chunkIndex, unsigned int objectId) const
{
    return getBuffer(chunkIndex, objectId).getCapacity();
}

unsigned int ChunkGeometryManager::allocateFace(unsigned int chunkIndex, unsigned int objectId)
{
    return getBuffer(chunkIndex, objectId).allocateFace();
}

void ChunkGeometryManager::deallocateFace(unsigned int chunkIndex, unsigned int objectId, unsigned int faceIndex)
{
    getBuffer(chunkIndex, objectId).deallocateFace(faceIndex);
}

bool ChunkGeometryManager::ensureCapacity(unsigned int chunkIndex, unsigned int objectId, unsigned int requiredFaces)
{
    return getBuffer(chunkIndex, objectId).ensureCapacity(requiredFaces);
}

size_t ChunkGeometryManager::getTotalMemoryUsage() const
{
    size_t total = 0;
    for (unsigned int chunkIndex = 0; chunkIndex < m_numChunks; ++chunkIndex) {
        for (unsigned int objectId = 0; objectId < m_numObjectTypes; ++objectId) {
            total += m_buffers[chunkIndex][objectId].getMemoryUsage();
        }
    }
    return total;
}

void ChunkGeometryManager::printAllStats() const
{
    printf("=== GEOMETRY MANAGER STATISTICS ===\n");
    printf("Total chunks: %u, Object types: %u\n", m_numChunks, m_numObjectTypes);
    printf("Total memory usage: %.2f MB\n", getTotalMemoryUsage() / (1024.0f * 1024.0f));
    
    for (unsigned int chunkIndex = 0; chunkIndex < m_numChunks; ++chunkIndex) {
        for (unsigned int objectId = 0; objectId < m_numObjectTypes; ++objectId) {
            const auto& buffer = m_buffers[chunkIndex][objectId];
            if (buffer.getCurrentFaceCount() > 0 || buffer.getCapacity() > 64) {
                std::string name = "chunk" + std::to_string(chunkIndex) + "_obj" + std::to_string(objectId);
                buffer.printStats(name);
            }
        }
    }
    printf("=====================================\n");
}