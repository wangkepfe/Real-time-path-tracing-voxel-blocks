

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "util/Picture.h"
#include <string>
#include <vector>

#ifndef SurfObj
#define SurfObj cudaSurfaceObject_t
#endif
#ifndef TexObj
#define TexObj cudaTextureObject_t
#endif

namespace jazzfusion
{

// Bitfield encoding of the texture channels.
// These are used to remap user format and user data to the internal format.
// Each four bits hold the channel index of red, green, blue, alpha, and luminance.
// (encoding >> ENC_*_SHIFT) & ENC_MASK gives the channel index if the result is less than 4.
// That encoding allows to automatically swap red and blue, map luminance to RGB (not the other way round though!),
// fill in alpha with input data or force it to one if required.
// 49 remapper functions take care to convert the data types including fixed-point adjustments.

#define ENC_MASK 0xF

#define ENC_RED_SHIFT 0
#define ENC_RED_0 (0 << ENC_RED_SHIFT)
#define ENC_RED_1 (1 << ENC_RED_SHIFT)
#define ENC_RED_2 (2 << ENC_RED_SHIFT)
#define ENC_RED_3 (3 << ENC_RED_SHIFT)
#define ENC_RED_NONE (15 << ENC_RED_SHIFT)

#define ENC_GREEN_SHIFT 4
#define ENC_GREEN_0 (0 << ENC_GREEN_SHIFT)
#define ENC_GREEN_1 (1 << ENC_GREEN_SHIFT)
#define ENC_GREEN_2 (2 << ENC_GREEN_SHIFT)
#define ENC_GREEN_3 (3 << ENC_GREEN_SHIFT)
#define ENC_GREEN_NONE (15 << ENC_GREEN_SHIFT)

#define ENC_BLUE_SHIFT 8
#define ENC_BLUE_0 (0 << ENC_BLUE_SHIFT)
#define ENC_BLUE_1 (1 << ENC_BLUE_SHIFT)
#define ENC_BLUE_2 (2 << ENC_BLUE_SHIFT)
#define ENC_BLUE_3 (3 << ENC_BLUE_SHIFT)
#define ENC_BLUE_NONE (15 << ENC_BLUE_SHIFT)

#define ENC_ALPHA_SHIFT 12
#define ENC_ALPHA_0 (0 << ENC_ALPHA_SHIFT)
#define ENC_ALPHA_1 (1 << ENC_ALPHA_SHIFT)
#define ENC_ALPHA_2 (2 << ENC_ALPHA_SHIFT)
#define ENC_ALPHA_3 (3 << ENC_ALPHA_SHIFT)
#define ENC_ALPHA_NONE (15 << ENC_ALPHA_SHIFT)

#define ENC_LUM_SHIFT 16
#define ENC_LUM_0 (0 << ENC_LUM_SHIFT)
#define ENC_LUM_1 (1 << ENC_LUM_SHIFT)
#define ENC_LUM_2 (2 << ENC_LUM_SHIFT)
#define ENC_LUM_3 (3 << ENC_LUM_SHIFT)
#define ENC_LUM_NONE (15 << ENC_LUM_SHIFT)

#define ENC_CHANNELS_SHIFT 20
#define ENC_CHANNELS_1 (1 << ENC_CHANNELS_SHIFT)
#define ENC_CHANNELS_2 (2 << ENC_CHANNELS_SHIFT)
#define ENC_CHANNELS_3 (3 << ENC_CHANNELS_SHIFT)
#define ENC_CHANNELS_4 (4 << ENC_CHANNELS_SHIFT)

#define ENC_TYPE_SHIFT 24
// These are indices into the remapper table.
#define ENC_TYPE_CHAR (0 << ENC_TYPE_SHIFT)
#define ENC_TYPE_UNSIGNED_CHAR (1 << ENC_TYPE_SHIFT)
#define ENC_TYPE_SHORT (2 << ENC_TYPE_SHIFT)
#define ENC_TYPE_UNSIGNED_SHORT (3 << ENC_TYPE_SHIFT)
#define ENC_TYPE_INT (4 << ENC_TYPE_SHIFT)
#define ENC_TYPE_UNSIGNED_INT (5 << ENC_TYPE_SHIFT)
#define ENC_TYPE_FLOAT (6 << ENC_TYPE_SHIFT)
#define ENC_TYPE_UNDEFINED (15 << ENC_TYPE_SHIFT)

// Flags to indicate that special handling is required.
#define ENC_MISC_SHIFT 28
#define ENC_FIXED_POINT (1 << ENC_MISC_SHIFT)
#define ENC_ALPHA_ONE (2 << ENC_MISC_SHIFT)
// Highest bit set means invalid encoding.
#define ENC_INVALID (8 << ENC_MISC_SHIFT)

class Texture
{
public:
    Texture();
    ~Texture();

    void setTextureDescription(cudaTextureDesc const& descr);

    void setAddressMode(cudaTextureAddressMode s, cudaTextureAddressMode t, cudaTextureAddressMode r);
    void setFilterMode(cudaTextureFilterMode filter, cudaTextureFilterMode filterMipmap);
    void setReadMode(cudaTextureReadMode mode);
    void setSRGB(bool srgb);
    void setBorderColor(float r, float g, float b, float a);
    void setNormalizedCoords(bool normalized);
    void setMaxAnisotropy(unsigned int aniso);
    void setMipmapLevelBiasMinMax(float bias, float minimum, float maximum);

    bool create(const Picture* picture, const unsigned int flags);

    unsigned int getWidth() const;
    unsigned int getHeight() const;
    unsigned int getDepth() const;

    TexObj getTextureObject() const;

    // Specific to spherical environment map.

    // Create cumulative distribution function for importance sampling of spherical environment lights. Call last.
    void calculateSphericalCDF(const float* rgba);

    CUdeviceptr getCDF_U() const;
    CUdeviceptr getCDF_V() const;
    float getIntegral() const;

private:
    bool create1D(const Picture* picture, const unsigned int flags);
    bool create2D(const Picture* picture, const unsigned int flags);
    bool create3D(const Picture* picture, const unsigned int flags);
    bool createCube(const Picture* picture, const unsigned int flags);
    bool createEnv(const Picture* picture, const unsigned int flags);

private:
    unsigned int m_width;
    unsigned int m_height;
    unsigned int m_depth;

    unsigned int m_hostEncoding;
    unsigned int m_deviceEncoding;
    cudaChannelFormatDesc m_channelFormat;
    size_t m_sizeBytesPerElement;

    cudaTextureDesc m_textureDescription; // This contains all texture parameters which can be set individually or as a whole.
    TexObj m_textureObject;

    cudaArray_t m_d_array;
    cudaMipmappedArray_t m_d_mipmappedArray;

    // Specific to spherical environment map.
    CUdeviceptr m_d_envCDF_U;
    CUdeviceptr m_d_envCDF_V;
    float m_integral;
};

}