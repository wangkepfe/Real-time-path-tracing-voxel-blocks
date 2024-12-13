#pragma once

// Always include this before any OptiX headers.
#include <cuda_runtime.h>

namespace jazzfusion
{

struct __align__(16) Camera
{
    Float2 inversedResolution;
    Float2 resolution;

    Float2 fov;
    Float2 tanHalfFov;

    Float3 adjustedLeft;
    float unused2;

    Float3 adjustedUp;
    float unused3;

    Float3 adjustedFront;
    float unused4;

    Float3 apertureLeft;
    float unused5;

    Float3 apertureUp;
    float unused6;

    Float3 pos;
    float  pitch;

    Float3 dir;
    float  focal;

    Float3 left;
    float  aperture;

    Float3 up;
    float  yaw;

    void init(int width, int height)
    {
        pos = Float3(5.0f, 5.0f, 0.0f);
        Float2 yawPitch = DirToYawPitch(-pos);
        yaw = yawPitch.x;
        pitch = yawPitch.y;
        up = Float3{ 0.0f, 1.0f, 0.0f };
        focal = 5.0f;
        aperture = 0.0f;
        resolution = Float2{ (float)width, (float)height };
        fov.x = 90.0f * Pi_over_180;
    }

    void update()
    {
        dir = YawPitchToDir(yaw, pitch);

        inversedResolution = 1.0f / resolution;
        fov.y = fov.x / resolution.x * resolution.y;
        tanHalfFov = Float2(tanf(fov.x / 2), tanf(fov.y / 2));

        up = Float3(0, 1, 0);
        left = normalize(cross(up, dir));
        up = normalize(cross(dir, left));

        adjustedFront = dir * focal;
        adjustedLeft = left * tanHalfFov.x * focal;
        adjustedUp = up * tanHalfFov.y * focal;

        apertureLeft = left * aperture;
        apertureUp = up * aperture;
    }
};

struct __align__(16) HistoryCamera
{
    inline __host__ void Setup(const Camera & cam)
    {
        invCamMat = Mat3(-cam.left, cam.up, cam.dir);  // build view matrix
        invCamMat.transpose();                      // orthogonal matrix, inverse is transpose
        pos = cam.pos;
    }

    inline __device__ Float2 WorldToScreenSpace(Float3 dir, Float2 tanHalfFov)
    {
        Float3 viewSpacePos = invCamMat * dir;            // transform world pos to view space
        Float2 screenPlanePos = viewSpacePos.xy / viewSpacePos.z;        // projection onto plane
        Float2 ndcSpacePos = screenPlanePos / tanHalfFov;             // [-1, 1]
        Float2 screenSpacePos = Float2(0.5) + ndcSpacePos * Float2(0.5); // [0, 1]
        return screenSpacePos;
    }

    Mat3 invCamMat;
    Float3 pos;
};

}