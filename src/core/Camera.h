#pragma once

// Always include this before any OptiX headers.
#include <cuda_runtime.h>

namespace jazzfusion
{

class Camera
{
public:
    Camera();
    ~Camera();

    void setViewport(int w, int h);
    void setBaseCoordinates(int x, int y);
    void setSpeedRatio(float f);
    void setFocusDistance(float f);

    void orbit(int x, int y);
    void pan(int x, int y);
    void dolly(int x, int y);
    void focus(int x, int y);
    void zoom(float x);

    bool  getFrustum(float3& pos, float3& u, float3& v, float3& w);
    float getAspectRatio() const;

public: // public just to be able to load and save them easily.
    float3 m_center;   // Center of interest point, around which is orbited (and the sharp plane of a depth of field camera).
    float  m_distance; // Distance of the camera from the center of interest.
    float  m_phi;      // Range [0.0f, 1.0f] from positive x-axis 360 degrees around the latitudes.
    float  m_theta;    // Range [0.0f, 1.0f] from negative to positive y-axis.
    float  m_fov;      // In degrees. Default is 60.0f

private:
    bool setDelta(int x, int y);

private:
    int   m_width;    // Viewport width.
    int   m_height;   // Viewport height.
    float m_aspect;   // m_width / m_height
    int   m_baseX;
    int   m_baseY;
    float m_speedRatio;

    // Derived values:
    int  m_dx;
    int  m_dy;
    bool m_changed;

    float3 m_cameraPosition;
    float3 m_cameraU;
    float3 m_cameraV;
    float3 m_cameraW;
};

}