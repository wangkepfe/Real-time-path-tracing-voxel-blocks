#pragma once

#include "shaders/LinearMath.h"

namespace vox
{

struct Ray
{
    jazzfusion::Float3 orig;
    jazzfusion::Float3 dir;
};

template<typename Lambda>
inline void RayVoxelGridTraversal(Ray& ray, Lambda&& func)
{
    using namespace jazzfusion;

    int x = static_cast<int>(ray.orig.x);
    int y = static_cast<int>(ray.orig.y);
    int z = static_cast<int>(ray.orig.z);

    bool needToContinue = true;

    while (needToContinue)
    {
        Float3 aabbMin = Float3(x, y, z);
        Float3 aabbMax = Float3(x + 1, y + 1, z + 1);

        int axis;
        float t = RayVoxelGridIntersect(ray.orig, ray.dir, aabbMin, aabbMax, axis);

        const float epsilon = 0.001f;

        ray.orig += ray.dir * (t + epsilon);

        if (axis == 0)
        {
            x += (ray.dir.x > 0) ? 1 : -1;
        }
        else if (axis == 1)
        {
            y += (ray.dir.y > 0) ? 1 : -1;
        }
        else if (axis == 2)
        {
            z += (ray.dir.z > 0) ? 1 : -1;
        }

        needToContinue = func(x, y, z);
    }
}

}