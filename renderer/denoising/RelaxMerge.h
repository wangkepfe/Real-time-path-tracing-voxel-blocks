#pragma once

#include "denoising/DenoiserCommon.h"
#include "shaders/Sampler.h"

// Final merge pass for RELAX denoising - combines diffuse and specular components
// Based on NVIDIA NRD RELAX methodology

__global__ void RelaxMerge(
    Int2 screenResolution,
    Float2 invScreenResolution,

    // Input: Separate denoised diffuse and specular channels
    SurfObj diffuseIlluminationBuffer,
    SurfObj specularIlluminationBuffer,
    
    // G-Buffer inputs for proper reconstruction
    SurfObj normalRoughnessBuffer,
    SurfObj materialBuffer,
    SurfObj depthBuffer,
    
    // Output: Final combined illumination
    SurfObj finalIlluminationBuffer,

    Camera camera,

    // Merge parameters
    float diffuseAlbedoBoost,
    float specularEnergyConservation,
    bool enableToneMapping,
    float exposureScale)
{
    Int2 pixelPos;
    pixelPos.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelPos.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelPos.x >= screenResolution.x || pixelPos.y >= screenResolution.y)
        return;

    float centerViewZ = Load2DFloat1(depthBuffer, pixelPos);
    
    // Early out for sky pixels or pixels beyond denoising range
    if (centerViewZ > 500000.0f) // TODO: Use denoisingRange parameter
    {
        Store2DFloat4(Float4(0.0f, 0.0f, 0.0f, 1.0f), finalIlluminationBuffer, pixelPos);
        return;
    }

    // Load denoised illumination components
    Float4 diffuseIllumination = Load2DFloat4(diffuseIlluminationBuffer, pixelPos);
    Float4 specularIllumination = Load2DFloat4(specularIlluminationBuffer, pixelPos);
    
    // Load surface properties
    float materialID = Load2DFloat1(materialBuffer, pixelPos);
    Float4 normalRoughness = Load2DFloat4(normalRoughnessBuffer, pixelPos);
    Float3 normal = normalRoughness.xyz;
    float roughness = normalRoughness.w;
    
    // Get world position for proper lighting reconstruction
    Float3 worldPos = GetCurrentWorldPosFromPixelPos(camera, pixelPos, centerViewZ);
    Float3 viewDir = normalize3f(camera.pos - worldPos);
    
    // Apply diffuse albedo boost for artistic control
    Float3 finalDiffuse = diffuseIllumination.xyz * diffuseAlbedoBoost;
    
    // Energy conservation between diffuse and specular
    // Based on Fresnel terms and material properties
    float F0 = 0.04f; // Default dielectric F0
    float cosTheta = max(0.0f, dot(viewDir, normal));
    float fresnel = F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
    
    // Metallic surfaces have different energy distribution
    float metallicFactor = saturate((materialID - 100.0f) / 100.0f); // Assuming material ID encoding
    fresnel = lerp(fresnel, 1.0f, metallicFactor);
    
    // Apply energy conservation
    Float3 finalSpecular = specularIllumination.xyz * specularEnergyConservation;
    Float3 energyConservedDiffuse = finalDiffuse * (1.0f - fresnel);
    
    // Combine diffuse and specular with proper energy conservation
    Float3 finalColor = energyConservedDiffuse + finalSpecular;
    
    // Optional tone mapping for HDR content
    if (enableToneMapping)
    {
        // Simple exposure and Reinhard tone mapping
        finalColor *= exposureScale;
        finalColor = finalColor / (Float3(1.0f) + finalColor);
        
        // Gamma correction (assuming sRGB output)
        finalColor = pow(max(finalColor, Float3(0.0f)), Float3(1.0f / 2.2f));
    }
    
    // Preserve alpha from diffuse channel (typically used for transparency)
    float finalAlpha = diffuseIllumination.w;
    
    // Store final result
    Store2DFloat4(Float4(finalColor, finalAlpha), finalIlluminationBuffer, pixelPos);
}

// Alternative simplified merge for cases where energy conservation is handled elsewhere
__global__ void RelaxMergeSimple(
    Int2 screenResolution,
    Float2 invScreenResolution,

    // Input: Separate denoised diffuse and specular channels
    SurfObj diffuseIlluminationBuffer,
    SurfObj specularIlluminationBuffer,
    
    // Output: Final combined illumination
    SurfObj finalIlluminationBuffer,
    
    // Simple blend weights
    float diffuseWeight,
    float specularWeight)
{
    Int2 pixelPos;
    pixelPos.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelPos.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelPos.x >= screenResolution.x || pixelPos.y >= screenResolution.y)
        return;

    // Load denoised illumination components
    Float4 diffuseIllumination = Load2DFloat4(diffuseIlluminationBuffer, pixelPos);
    Float4 specularIllumination = Load2DFloat4(specularIlluminationBuffer, pixelPos);
    
    // Simple weighted combination
    Float3 finalColor = diffuseIllumination.xyz * diffuseWeight + specularIllumination.xyz * specularWeight;
    float finalAlpha = diffuseIllumination.w;
    
    // Store final result
    Store2DFloat4(Float4(finalColor, finalAlpha), finalIlluminationBuffer, pixelPos);
}

// Advanced merge with material-aware reconstruction
__global__ void RelaxMergeAdvanced(
    Int2 screenResolution,
    Float2 invScreenResolution,

    // Input: Separate denoised diffuse and specular channels
    SurfObj diffuseIlluminationBuffer,
    SurfObj specularIlluminationBuffer,
    
    // G-Buffer inputs for material-aware reconstruction
    SurfObj normalRoughnessBuffer,
    SurfObj materialBuffer,
    SurfObj depthBuffer,
    SurfObj albedoBuffer,        // Base color/albedo
    SurfObj metallicBuffer,      // Metallic factor
    
    // Output: Final combined illumination
    SurfObj finalIlluminationBuffer,

    Camera camera,

    // Advanced merge parameters
    float diffuseBoost,
    float specularBoost,
    float metallicSpecularBoost,
    bool enableAdvancedEnergyConservation,
    bool enableColorCorrection,
    Float3 colorTemperature)
{
    Int2 pixelPos;
    pixelPos.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelPos.y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pixelPos.x >= screenResolution.x || pixelPos.y >= screenResolution.y)
        return;

    float centerViewZ = Load2DFloat1(depthBuffer, pixelPos);
    
    // Early out for sky pixels
    if (centerViewZ > 500000.0f)
    {
        Store2DFloat4(Float4(0.0f, 0.0f, 0.0f, 1.0f), finalIlluminationBuffer, pixelPos);
        return;
    }

    // Load all required data
    Float4 diffuseIllumination = Load2DFloat4(diffuseIlluminationBuffer, pixelPos);
    Float4 specularIllumination = Load2DFloat4(specularIlluminationBuffer, pixelPos);
    
    Float4 normalRoughness = Load2DFloat4(normalRoughnessBuffer, pixelPos);
    Float3 normal = normalRoughness.xyz;
    float roughness = normalRoughness.w;
    
    Float3 albedo = Load2DFloat3(albedoBuffer, pixelPos);
    float metallic = Load2DFloat1(metallicBuffer, pixelPos);
    
    // Calculate view direction for Fresnel computation
    Float3 worldPos = GetCurrentWorldPosFromPixelPos(camera, pixelPos, centerViewZ);
    Float3 viewDir = normalize3f(camera.pos - worldPos);
    float NdotV = max(0.0f, dot(normal, viewDir));
    
    // PBR-style energy conservation
    Float3 F0 = lerp(Float3(0.04f), albedo, metallic);
    Float3 F = F0 + (Float3(1.0f) - F0) * pow(1.0f - NdotV, 5.0f);
    
    // Apply material-aware scaling
    Float3 finalDiffuse = diffuseIllumination.xyz * diffuseBoost;
    Float3 finalSpecular = specularIllumination.xyz * lerp(specularBoost, metallicSpecularBoost, metallic);
    
    if (enableAdvancedEnergyConservation)
    {
        // Proper PBR energy conservation
        Float3 kS = F;  // Specular contribution
        Float3 kD = Float3(1.0f) - kS;  // Diffuse contribution
        kD *= 1.0f - metallic;  // Metallics don't have diffuse
        
        finalDiffuse *= kD;
        finalSpecular *= kS;
    }
    
    // Combine channels
    Float3 finalColor = finalDiffuse + finalSpecular;
    
    // Optional color temperature adjustment
    if (enableColorCorrection)
    {
        // Simple color temperature shift
        finalColor *= colorTemperature;
    }
    
    // Preserve alpha
    float finalAlpha = diffuseIllumination.w;
    
    Store2DFloat4(Float4(finalColor, finalAlpha), finalIlluminationBuffer, pixelPos);
}