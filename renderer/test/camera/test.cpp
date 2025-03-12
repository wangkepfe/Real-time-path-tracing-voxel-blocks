#include "../../shaders/LinearMath.h"
#include <cassert>
#include <cmath>
#include <iostream>

// Camera implementation (modified for unit testing)
struct Camera
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

    Camera() {}

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

        // Construct the UV-to-NDC matrix.
        // Here we assume the three provided Float3 values form the rows of the matrix.
        Mat3 uvToNdc(
            Float3(2.0f, 0.0f, 0.0f),
            Float3(0.0f, 2.0f, 0.0f),
            Float3(-1.0f, -1.0f, 1.0f));

        Mat3 ndcToView;
        ndcToView.m00 = tanHalfFov.x;
        ndcToView.m11 = tanHalfFov.y;
        ndcToView.m22 = 1.0f;

        // Construct the view-to-world matrix.
        // We assume that the constructor Mat3(a, b, c) creates a matrix whose columns are a, b, and c.
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

    Float3 uvToWorldDirection(const Float2 &uv) const
    {
        Float3 uv_h(uv.x, uv.y, 1.0f);
        Float3 worldDir = uvToWorld * uv_h;
        return normalize(worldDir);
    }

    Float2 worldDirectionToUV(const Float3 &worldDir) const
    {
        Float3 ndc_h = worldToUv * worldDir;
        return Float2(ndc_h.x, ndc_h.y) / ndc_h.z;
    }

    Float3 uvToViewDirection(const Float2 &uv) const
    {
        Float3 uv_h(uv.x, uv.y, 1.0f);
        Float3 viewDir = uvToView * uv_h;
        return normalize(viewDir);
    }

    Float2 viewDirectionToUV(const Float3 &viewDir) const
    {
        Float3 ndc_h = viewToUv * viewDir;
        return Float2(ndc_h.x, ndc_h.y) / ndc_h.z;
    }

    float getPixelWorldSizeScaleToDepth() const
    {
        return tanHalfFov.x / (resolution.x / 2);
    }
};

// Helper epsilon and comparison functions
constexpr float EPSILON = 1e-4f;
bool nearlyEqual(float a, float b, float epsilon = EPSILON)
{
    return std::fabs(a - b) < epsilon;
}
bool nearlyEqual(const Float2 &a, const Float2 &b, float epsilon = EPSILON)
{
    return nearlyEqual(a.x, b.x, epsilon) && nearlyEqual(a.y, b.y, epsilon);
}
bool nearlyEqual(const Float3 &a, const Float3 &b, float epsilon = EPSILON)
{
    return nearlyEqual(a.x, b.x, epsilon) && nearlyEqual(a.y, b.y, epsilon) && nearlyEqual(a.z, b.z, epsilon);
}

// For printing (assume operator<< is defined for Float3; otherwise you can print members directly)
std::ostream &operator<<(std::ostream &os, const Float3 &v)
{
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}

int main()
{
    // --- Basic Initialization and Update Tests ---
    Camera cam;
    int width = 800, height = 600;
    cam.init(width, height);

    // Check resolution and inverse resolution.
    assert(nearlyEqual(cam.resolution, Float2(800.0f, 600.0f)));
    Float2 expectedInvRes = Float2(1.0f / 800.0f, 1.0f / 600.0f);
    assert(nearlyEqual(cam.inversedResolution, expectedInvRes));

    // fovX is 90° so tan(45°)=1.0.
    assert(nearlyEqual(cam.tanHalfFov.x, 1.0f, 1e-3f));
    float fovY = 90.0f * (600.0f / 800.0f);
    float expectedTanHalfFovY = std::tanf((fovY * (M_PI / 180.0f)) * 0.5f);
    assert(nearlyEqual(cam.tanHalfFov.y, expectedTanHalfFovY, 1e-3f));

    // Set yaw and pitch so that the camera looks straight ahead.
    // (Assume YawPitchToDir(0,0) yields (0,0,1))
    cam.yaw = 0.0f;
    cam.pitch = 0.0f;
    cam.update();

    // Verify that the camera direction is normalized.
    float dirLen = length(cam.dir);
    assert(nearlyEqual(dirLen, 1.0f));

    // Verify that the center UV (0.5, 0.5) maps to the camera's forward direction.
    Float2 centerUV(0.5f, 0.5f);
    Float3 centerWorldDir = cam.uvToWorldDirection(centerUV);
    float dp = dot(centerWorldDir, cam.dir);
    assert(nearlyEqual(dp, 1.0f, 1e-3f));

    // Verify round-trip conversion.
    Float2 testUV(0.3f, 0.7f);
    Float3 testWorldDir = cam.uvToWorldDirection(testUV);
    Float2 recoveredUV = cam.worldDirectionToUV(testWorldDir);
    assert(nearlyEqual(testUV, recoveredUV, 1e-3f));

    // --- Near Gimbal Lock Scenario ---
    cam.yaw = 0.0f;
    cam.pitch = 89.0f * (M_PI / 180.0f);
    cam.update();
    float newDirLen = length(cam.dir);
    assert(nearlyEqual(newDirLen, 1.0f));
    Float3 nearGimbalDir = cam.uvToWorldDirection(centerUV);
    float nearGimbalLen = length(nearGimbalDir);
    assert(nearlyEqual(nearGimbalLen, 1.0f));

    // --- Concrete Test Cases for uvToWorldDirection and worldDirectionToUV ---
    // For yaw=0 and pitch=0, we assume the following expected world directions:
    // Given:
    //   tanHalfFov.x = 1.0,
    //   tanHalfFov.y ≈ 0.66818,
    // and viewToWorld = (-left, up, dir) with left=(1,0,0), up=(0,1,0), dir=(0,0,1),
    // then we expect:
    //   For uv = (0.5, 0.5): worldDir = (0, 0, 1).
    //   For uv = (0, 0): worldDir ≈ normalize( ( 1, -0.66818, 1) ) ≈ (0.639, -0.427, 0.639 ).
    //   For uv = (1, 1): worldDir ≈ normalize( (-1, 0.66818, 1) ) ≈ (-0.639, 0.428, 0.639 ).
    //   For uv = (0, 1): worldDir ≈ normalize( ( 1, 0.66818, 1) ) ≈ (0.639, 0.428, 0.639 ).
    //   For uv = (1, 0): worldDir ≈ normalize( (-1, -0.66818, 1) ) ≈ (-0.639, -0.427, 0.639 ).

    cam.yaw = 0.0f;
    cam.pitch = 0.0f;
    cam.update();

    // Precompute expected normalized vectors.
    Float3 expectedCenter = Float3(0.0f, 0.0f, 1.0f);
    Float3 expected00 = normalize(Float3(1.0f, -0.66818f, 1.0f));
    Float3 expected11 = normalize(Float3(-1.0f, 0.66818f, 1.0f));
    Float3 expected01 = normalize(Float3(1.0f, 0.66818f, 1.0f));
    Float3 expected10 = normalize(Float3(-1.0f, -0.66818f, 1.0f));

    {
        // Test for uv = (0.5, 0.5)
        Float2 uv(0.5f, 0.5f);
        Float3 worldDir = cam.uvToWorldDirection(uv);
        std::cout << "UV (0.5, 0.5) worldDir: " << worldDir << std::endl;
        assert(nearlyEqual(worldDir, expectedCenter, 1e-3f));
    }
    {
        // Test for uv = (0, 0)
        Float2 uv(0.0f, 0.0f);
        Float3 worldDir = cam.uvToWorldDirection(uv);
        std::cout << "UV (0, 0) worldDir: " << worldDir << std::endl;
        assert(nearlyEqual(worldDir, expected00, 1e-3f));
    }
    {
        // Test for uv = (1, 1)
        Float2 uv(1.0f, 1.0f);
        Float3 worldDir = cam.uvToWorldDirection(uv);
        std::cout << "UV (1, 1) worldDir: " << worldDir << std::endl;
        assert(nearlyEqual(worldDir, expected11, 1e-3f));
    }
    {
        // Test for uv = (0, 1)
        Float2 uv(0.0f, 1.0f);
        Float3 worldDir = cam.uvToWorldDirection(uv);
        std::cout << "UV (0, 1) worldDir: " << worldDir << std::endl;
        assert(nearlyEqual(worldDir, expected01, 1e-3f));
    }
    {
        // Test for uv = (1, 0)
        Float2 uv(1.0f, 0.0f);
        Float3 worldDir = cam.uvToWorldDirection(uv);
        std::cout << "UV (1, 0) worldDir: " << worldDir << std::endl;
        assert(nearlyEqual(worldDir, expected10, 1e-3f));
    }

    // Also verify that converting back recovers the original UV.
    {
        Float2 testUVs[] = {Float2(0.0f, 0.0f), Float2(0.5f, 0.5f), Float2(1.0f, 1.0f),
                            Float2(0.0f, 1.0f), Float2(1.0f, 0.0f), Float2(0.25f, 0.75f)};
        for (const auto &uv : testUVs)
        {
            Float3 wDir = cam.uvToWorldDirection(uv);
            Float2 uvRecovered = cam.worldDirectionToUV(wDir);
            std::cout << "Original UV: (" << uv.x << ", " << uv.y << ") -> Recovered UV: ("
                      << uvRecovered.x << ", " << uvRecovered.y << ")" << std::endl;
            assert(nearlyEqual(uv, uvRecovered, 1e-3f));
        }
    }

    std::cout << "All camera tests passed!" << std::endl;
    return 0;
}
