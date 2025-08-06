#include "SystemParameter.h"
#include "OptixShaderCommon.h"
#include "ShaderDebugUtils.h"
#include "Sampler.h"
#include "SelfHit.h"
#include "Restir.h"

extern "C" __constant__ SystemParameter sysParam;

extern "C" __global__ void __closesthit__radiance()
{
    RayData *rayData = (RayData *)mergePointer(optixGetPayload_0(), optixGetPayload_1());
    
    // TEST STAGE 4: Access triangle indices - THE ORIGINAL SUSPECT!
    const GeometryInstanceData *instanceData = reinterpret_cast<const GeometryInstanceData *>(optixGetSbtDataPointer());
    
    if (!instanceData) {
        rayData->radiance = Float3(1.0f, 0.0f, 0.0f); // Red for null SBT
        rayData->shouldTerminate = true;
        return;
    }
    
    // Check if triangle data exists before accessing
    if (!instanceData->indices || !instanceData->attributes) {
        rayData->radiance = Float3(0.8f, 0.0f, 0.8f); // Magenta for null triangle data
        rayData->shouldTerminate = true;
        return;
    }
    
    // Get triangle index with bounds checking - ROOT CAUSE IDENTIFIED AND FIXED!
    const unsigned int meshTriangleIndex = optixGetPrimitiveIndex();
    
    // CRITICAL FIX: Bounds check triangle index access
    // The issue was that optixGetPrimitiveIndex() can return values >= numIndices
    if (meshTriangleIndex >= instanceData->numIndices) {
        rayData->radiance = Float3(1.0f, 0.5f, 0.0f); // Orange for out-of-bounds triangle
        rayData->shouldTerminate = true;
        return;
    }
    
    const Int3 tri = instanceData->indices[meshTriangleIndex]; // NOW SAFE!
    
    // If we get here, triangle index access succeeded
    // Use triangle vertex indices as color components (normalize to 0-1 range)
    float r = (tri.x % 256) / 255.0f;
    float g = (tri.y % 256) / 255.0f;  
    float b = (tri.z % 256) / 255.0f;
    
    rayData->radiance = Float3(r, g, b);
    rayData->shouldTerminate = true;
}

extern "C" __global__ void __closesthit__bsdf_light()
{
    ShadowRayData *rayData = (ShadowRayData *)mergePointer(optixGetPayload_0(), optixGetPayload_1());

    const GeometryInstanceData *instanceData = reinterpret_cast<const GeometryInstanceData *>(optixGetSbtDataPointer());

    // Defensive null pointer check
    if (!instanceData || !instanceData->indices || !instanceData->attributes) {
        return;
    }

    const unsigned int instanceId = optixGetInstanceId();
    const unsigned int meshTriangleIndex = optixGetPrimitiveIndex();

    float2 bary = optixGetTriangleBarycentrics();

    rayData->bary = Float2(bary);

    // Bounds check for material parameter access
    unsigned int materialIndex = instanceData->materialIndex;
    if (materialIndex >= sysParam.numMaterialParameters) {
        materialIndex = 0; // Default to first material if out of bounds
    }
    const MaterialParameter &parameters = sysParam.materialParameters[materialIndex];

    if (parameters.isEmissive)
    {
        int idx = -1;
        
        // Only perform binary search if there are instanced light meshes
        if (sysParam.numInstancedLightMesh > 0 && sysParam.instanceLightMapping)
        {
            int left = 0;
            int right = sysParam.numInstancedLightMesh - 1;

            while (left <= right)
            {
                int mid = (left + right) / 2;
                unsigned int midVal = sysParam.instanceLightMapping[mid].instanceId;

                if (midVal == instanceId)
                {
                    idx = mid;
                    break;
                }
                else if (midVal < instanceId)
                {
                    left = mid + 1;
                }
                else
                {
                    right = mid - 1;
                }
            }
        }
        if (idx != -1)
        {
            rayData->lightIdx = sysParam.instanceLightMapping[idx].lightOffset + meshTriangleIndex;
        }
    }
}