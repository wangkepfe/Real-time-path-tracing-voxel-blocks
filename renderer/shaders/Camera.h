#pragma once

// Always include this before any OptiX headers.
#include <cuda_runtime.h>

namespace jazzfusion
{
    struct __align__(16) Camera
    {
        Float2 resolution;
        Float2 inversedResolution;

        Float2 tanHalfFov;

        Float3 pos;
        Float3 dir;

        float yaw;
        float pitch;

        Mat3 uvToWorld;
        Mat3 worldToUv;

        Mat3 uvToView;
        Mat3 viewToUv;

        void init(int width, int height)
        {
            pos = Float3(16.0f, 17.0f, 16.0f);
            dir = normalize(Float3(-1.0f, -1.0f, -1.0f)); // Example default direction

            resolution = Float2((float)width, (float)height);
            inversedResolution = 1.0f / resolution;

            float fovX = 90.0f * Pi_over_180;
            float fovY = fovX * (resolution.y / resolution.x);
            tanHalfFov = Float2(tanf(fovX * 0.5f), tanf(fovY * 0.5f));
        }

        void update()
        {
            dir = YawPitchToDir(yaw, pitch);

            // Fixed vertical direction for camera setup
            Float3 worldUp = Float3(0.0f, 1.0f, 0.0f);

            // Compute local camera basis
            Float3 left = normalize(cross(worldUp, dir));
            Float3 up = normalize(cross(dir, left));

            Mat3 uvToNdc(
                Float3(2.0f, 0.0f, 0.0f),
                Float3(0.0f, 2.0f, 0.0f),
                Float3(-1.0f, -1.0f, 1.0f));

            Mat3 ndcToView;
            ndcToView.m00 = tanHalfFov.x;
            ndcToView.m11 = tanHalfFov.y;
            ndcToView.m22 = 1.0f;

            Mat3 viewToWorld(-left, up, dir);

            uvToWorld = viewToWorld * ndcToView * uvToNdc;
            uvToView = viewToWorld * ndcToView;

            Mat3 ndcToUv(
                Float3(0.5f, 0.0f, 0.0f),
                Float3(0.0f, 0.5f, 0.0f),
                Float3(0.5f, 0.5f, 1.0f));

            Mat3 worldToView = viewToWorld;
            worldToView.transpose();

            Mat3 viewToNdc;
            viewToNdc.m00 = 1.0f / tanHalfFov.x;
            viewToNdc.m11 = 1.0f / tanHalfFov.y;
            viewToNdc.m22 = 1.0f;

            worldToUv = ndcToUv * viewToNdc * worldToView;
            viewToUv = viewToNdc * worldToView;
        }

        __device__ Float3 uvToWorldDirection(const Float2 &uv) const
        {
            Float3 uv_h(uv.x, uv.y, 1.0f);
            Float3 worldDir = uvToWorld * uv_h;
            return normalize(worldDir);
        }

        __device__ Float2 worldDirectionToUV(const Float3 &worldDir) const
        {
            Float3 ndc_h = worldToUv * worldDir;
            return Float2(ndc_h.x, ndc_h.y) / ndc_h.z;
        }

        __device__ Float3 uvToViewDirection(const Float2 &uv) const
        {
            Float3 uv_h(uv.x, uv.y, 1.0f);
            Float3 viewDir = uvToView * uv_h;
            return normalize(viewDir);
        }

        __device__ Float2 viewDirectionToUV(const Float3 &viewDir) const
        {
            Float3 ndc_h = viewToUv * viewDir;
            return Float2(ndc_h.x, ndc_h.y) / ndc_h.z;
        }

        __device__ float getPixelWorldSizeScaleToDepth() const
        {
            return tanHalfFov.x / (resolution.x / 2);
        }

        __device__ float getRayConeWidth(Int2 idx) const
        {
            Float2 pixelCenter = (Float2(idx.x, idx.y) + 0.5f) - Float2(resolution.x, resolution.y) / 2;
            Float2 pixelOffset = copysignf2(Float2(0.5f), pixelCenter);

            Float2 uvNear = (pixelCenter - pixelOffset) * inversedResolution * 2; // [-1, 1]
            Float2 uvFar = (pixelCenter + pixelOffset) * inversedResolution * 2;

            Float2 pointOnPlaneNear = uvNear * tanHalfFov;
            Float2 pointOnPlaneFar = uvFar * tanHalfFov;

            float angleNear = atanf(pointOnPlaneNear.length());
            float angleFar = atanf(pointOnPlaneFar.length());
            float pixelAngleWidth = angleFar - angleNear;

            return pixelAngleWidth;
        }
    };
}