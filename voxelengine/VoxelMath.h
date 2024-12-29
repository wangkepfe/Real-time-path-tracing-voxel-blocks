#pragma once

#include "shaders/LinearMath.h"

#include <limits>

namespace vox
{

     struct Ray
     {
         jazzfusion::Float3 orig;
         jazzfusion::Float3 dir;
     };

    // template <typename Lambda>
    // inline void RayVoxelGridTraversal(Ray &ray, Lambda &&func, int maxIteration = 1000,
    //                                   int *hitAxisOut = nullptr, int *stepXOut = nullptr, int *stepYOut = nullptr, int *stepZOut = nullptr)
    // {
    //     using namespace jazzfusion;

    //     // Normalize the ray direction if needed
    //     Float3 dir = ray.dir;
    //     {
    //         float len = std::sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    //         if (len > 1e-8f)
    //         {
    //             dir.x /= len;
    //             dir.y /= len;
    //             dir.z /= len;
    //         }
    //         else
    //         {
    //             // Degenerate direction vector: no traversal possible
    //             return;
    //         }
    //     }

    //     // Find the starting voxel indices
    //     int x = static_cast<int>(std::floor(ray.orig.x));
    //     int y = static_cast<int>(std::floor(ray.orig.y));
    //     int z = static_cast<int>(std::floor(ray.orig.z));

    //     // Determine step direction along each axis
    //     int stepX = (dir.x > 0.0f) ? 1 : -1;
    //     int stepY = (dir.y > 0.0f) ? 1 : -1;
    //     int stepZ = (dir.z > 0.0f) ? 1 : -1;

    //     // Compute tDeltaX, tDeltaY, tDeltaZ
    //     float tDeltaX = (std::fabs(dir.x) < 1e-8f) ? FLT_MAX : (1.0f / std::fabs(dir.x));
    //     float tDeltaY = (std::fabs(dir.y) < 1e-8f) ? FLT_MAX : (1.0f / std::fabs(dir.y));
    //     float tDeltaZ = (std::fabs(dir.z) < 1e-8f) ? FLT_MAX : (1.0f / std::fabs(dir.z));

    //     float nextBoundaryX = (stepX > 0) ? (float)(x + 1) : (float)(x);
    //     float nextBoundaryY = (stepY > 0) ? (float)(y + 1) : (float)(y);
    //     float nextBoundaryZ = (stepZ > 0) ? (float)(z + 1) : (float)(z);

    //     float tMaxX = (std::fabs(dir.x) < 1e-8f) ? FLT_MAX : (nextBoundaryX - ray.orig.x) / dir.x;
    //     float tMaxY = (std::fabs(dir.y) < 1e-8f) ? FLT_MAX : (nextBoundaryY - ray.orig.y) / dir.y;
    //     float tMaxZ = (std::fabs(dir.z) < 1e-8f) ? FLT_MAX : (nextBoundaryZ - ray.orig.z) / dir.z;

    //     int iterationCount = 0;

    //     // This will track the axis along which we moved last:
    //     // 0 = x-axis, 1 = y-axis, 2 = z-axis.
    //     // We initialize it to -1 meaning we haven't moved yet.
    //     int lastAxis = -1;

    //     while (iterationCount++ < maxIteration)
    //     {
    //         // Check the current voxel
    //         if (!func(x, y, z))
    //         {
    //             // We found a solid voxel and want to stop. We break here.
    //             // lastAxis indicates along which axis we stepped to get here.
    //             break;
    //         }

    //         // Move to the next voxel:
    //         // Choose the axis with the smallest tMax
    //         if (tMaxX < tMaxY)
    //         {
    //             if (tMaxX < tMaxZ)
    //             {
    //                 x += stepX;
    //                 tMaxX += tDeltaX;
    //                 lastAxis = 0; // moved along x-axis
    //             }
    //             else
    //             {
    //                 z += stepZ;
    //                 tMaxZ += tDeltaZ;
    //                 lastAxis = 2; // moved along z-axis
    //             }
    //         }
    //         else
    //         {
    //             if (tMaxY < tMaxZ)
    //             {
    //                 y += stepY;
    //                 tMaxY += tDeltaY;
    //                 lastAxis = 1; // moved along y-axis
    //             }
    //             else
    //             {
    //                 z += stepZ;
    //                 tMaxZ += tDeltaZ;
    //                 lastAxis = 2; // moved along z-axis
    //             }
    //         }
    //     }

    //     if (hitAxisOut)
    //         *hitAxisOut = lastAxis;
    //     if (stepXOut)
    //         *stepXOut = stepX;
    //     if (stepYOut)
    //         *stepYOut = stepY;
    //     if (stepZOut)
    //         *stepZOut = stepZ;
    // }

    INL_HOST_DEVICE unsigned int GetLinearId(unsigned int x, unsigned int y, unsigned int z, unsigned int width)
    {
        x = x < width - 1 ? x : width - 1;
        y = y < width - 1 ? y : width - 1;
        z = z < width - 1 ? z : width - 1;

        return x + width * (z + width * y);
    }

}