#pragma once

#include "VoxelSceneGen.h"
#include "VoxelChunk.h"

#include "shaders/SystemParameter.h"
#include <cassert>
#include <cstring>
#include <functional>

namespace vox
{

    class VoxelEngine
    {
    public:
        static VoxelEngine &Get()
        {
            static VoxelEngine instance;
            return instance;
        }
        VoxelEngine(VoxelEngine const &) = delete;
        void operator=(VoxelEngine const &) = delete;
        ~VoxelEngine();

        void init();
        void update();
        void generateVoxels();

        VoxelChunk voxelChunk;
        bool leftMouseButtonClicked = false;

        unsigned int currentFaceCount = 0;
        unsigned int maxFaceCount = 0;
        std::vector<unsigned int> freeFaces;
        std::vector<unsigned int> faceLocation;

        // static std::function<void()> UpdateFunc;

    private:
        VoxelEngine()
        {
        }
    };

    static inline void UpdateFunc()
    {
        VoxelEngine::Get().update();
    }

}