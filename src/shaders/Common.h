#pragma once

#ifndef INL_HOST_DEVICE
#define INL_HOST_DEVICE __forceinline__ __host__ __device__
#endif

#ifndef INL_DEVICE
#define INL_DEVICE __forceinline__ __device__
#endif

#ifndef SurfObj
#define SurfObj cudaSurfaceObject_t
#endif

#ifndef TexObj
#define TexObj cudaTextureObject_t
#endif