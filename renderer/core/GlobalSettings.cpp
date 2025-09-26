#include "GlobalSettings.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#ifndef OFFLINE_MODE
#include <GLFW/glfw3.h>
#endif

// Helper functions for YAML parsing (similar to SceneConfig.cpp)
namespace
{
    std::string trim(const std::string &str)
    {
        size_t start = str.find_first_not_of(" \t\r\n");
        if (start == std::string::npos)
            return "";

        size_t end = str.find_last_not_of(" \t\r\n");
        return str.substr(start, end - start + 1);
    }

    bool parseFloat(const std::string &value, float &result)
    {
        try
        {
            result = std::stof(trim(value));
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    }

    bool parseInt(const std::string &value, int &result)
    {
        try
        {
            result = std::stoi(trim(value));
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    }

    bool parseBool(const std::string &value, bool &result)
    {
        std::string trimmedValue = trim(value);
        std::transform(trimmedValue.begin(), trimmedValue.end(), trimmedValue.begin(), ::tolower);

        if (trimmedValue == "true" || trimmedValue == "1" || trimmedValue == "yes" || trimmedValue == "on")
        {
            result = true;
            return true;
        }
        else if (trimmedValue == "false" || trimmedValue == "0" || trimmedValue == "no" || trimmedValue == "off")
        {
            result = false;
            return true;
        }
        return false;
    }
}

bool GlobalSettings::LoadFromYAML(const std::string &filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Failed to open global settings file: " << filepath << std::endl;
        return false;
    }

    std::string line;
    std::string currentSection;

    while (std::getline(file, line))
    {
        line = trim(line);

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#')
            continue;

        // Check for section headers (e.g., "denoising:")
        if (line.back() == ':')
        {
            currentSection = line.substr(0, line.length() - 1);
            continue;
        }

        // Parse key-value pairs
        size_t colonPos = line.find(':');
        if (colonPos == std::string::npos)
            continue;

        std::string key = trim(line.substr(0, colonPos));
        std::string value = trim(line.substr(colonPos + 1));

        // Parse settings by section
        if (currentSection == "denoising")
        {
            parseDenosingSettings(key, value);
        }
        else if (currentSection == "postprocess")
        {
            parseToneMappingSettings(key, value);
            parsePostProcessingPipelineSettings(key, value);
        }
        else if (currentSection == "sky")
        {
            parseSkySettings(key, value);
        }
        else if (currentSection == "character_movement")
        {
            parseCharacterMovementSettings(key, value);
        }
        else if (currentSection == "character_animation")
        {
            parseCharacterAnimationSettings(key, value);
        }
        else if (currentSection == "camera_movement")
        {
            parseCameraMovementSettings(key, value);
        }
        else if (currentSection == "rendering")
        {
            parseRenderingSettings(key, value);
        }
    }

    std::cout << "Loaded global settings from: " << filepath << std::endl;
    return true;
}

void GlobalSettings::SaveToYAML(const std::string &filepath) const
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Failed to create global settings file: " << filepath << std::endl;
        return;
    }

    file << "# Global Settings Configuration File\n";
    file << "# Generated automatically\n\n";

    // Save Denoising Parameters
    file << "denoising:\n";
    file << "  enableHitDistanceReconstruction: " << (denoisingParams.enableHitDistanceReconstruction ? "true" : "false") << "\n";
    file << "  enablePrePass: " << (denoisingParams.enablePrePass ? "true" : "false") << "\n";
    file << "  enableTemporalAccumulation: " << (denoisingParams.enableTemporalAccumulation ? "true" : "false") << "\n";
    file << "  enableHistoryFix: " << (denoisingParams.enableHistoryFix ? "true" : "false") << "\n";
    file << "  enableHistoryClamping: " << (denoisingParams.enableHistoryClamping ? "true" : "false") << "\n";
    file << "  enableSpatialFiltering: " << (denoisingParams.enableSpatialFiltering ? "true" : "false") << "\n";
    file << "  maxAccumulatedFrameNum: " << denoisingParams.maxAccumulatedFrameNum << "\n";
    file << "  maxFastAccumulatedFrameNum: " << denoisingParams.maxFastAccumulatedFrameNum << "\n";
    file << "  phiLuminance: " << denoisingParams.phiLuminance << "\n";
    file << "  lobeAngleFraction: " << denoisingParams.lobeAngleFraction << "\n";
    file << "  roughnessFraction: " << denoisingParams.roughnessFraction << "\n";
    file << "  depthThreshold: " << denoisingParams.depthThreshold << "\n";
    file << "  atrousIterationNum: " << denoisingParams.atrousIterationNum << "\n";
    file << "  disocclusionThreshold: " << denoisingParams.disocclusionThreshold << "\n";
    file << "  disocclusionThresholdAlternate: " << denoisingParams.disocclusionThresholdAlternate << "\n";
    file << "  denoisingRange: " << denoisingParams.denoisingRange << "\n\n";

    // Save Post Processing Parameters
    // Save Post Processing Parameters
    file << "postprocess:\n";
    // Tone mapping parameters
    file << "  manualExposure: " << toneMappingParams.manualExposure << "\n";
    file << "  toneMappingCurve: " << static_cast<int>(toneMappingParams.curve) << "\n";
    file << "  highlightDesaturation: " << toneMappingParams.highlightDesaturation << "\n";
    file << "  whitePoint: " << toneMappingParams.whitePoint << "\n";
    file << "  contrast: " << toneMappingParams.contrast << "\n";
    file << "  saturation: " << toneMappingParams.saturation << "\n";
    file << "  gain: " << toneMappingParams.gain << "\n";
    file << "  lift: " << toneMappingParams.lift << "\n";
    file << "  enableChromaticAdaptation: " << (toneMappingParams.enableChromaticAdaptation ? "true" : "false") << "\n";
    
    // Post processing pipeline parameters
    // Bloom parameters
    file << "  enableBloom: " << (postProcessingPipelineParams.enableBloom ? "true" : "false") << "\n";
    file << "  bloomThreshold: " << postProcessingPipelineParams.bloomThreshold << "\n";
    file << "  bloomIntensity: " << postProcessingPipelineParams.bloomIntensity << "\n";
    file << "  bloomRadius: " << postProcessingPipelineParams.bloomRadius << "\n";
    
    // Auto-exposure parameters
    file << "  enableAutoExposure: " << (postProcessingPipelineParams.enableAutoExposure ? "true" : "false") << "\n";
    file << "  exposureSpeed: " << postProcessingPipelineParams.exposureSpeed << "\n";
    file << "  exposureMin: " << postProcessingPipelineParams.exposureMin << "\n";
    file << "  exposureMax: " << postProcessingPipelineParams.exposureMax << "\n";
    file << "  exposureCompensation: " << postProcessingPipelineParams.exposureCompensation << "\n";
    file << "  histogramMinPercent: " << postProcessingPipelineParams.histogramMinPercent << "\n";
    file << "  histogramMaxPercent: " << postProcessingPipelineParams.histogramMaxPercent << "\n";
    file << "  targetLuminance: " << postProcessingPipelineParams.targetLuminance << "\n";
    
    // Vignette parameters
    file << "  enableVignette: " << (postProcessingPipelineParams.enableVignette ? "true" : "false") << "\n";
    file << "  vignetteStrength: " << postProcessingPipelineParams.vignetteStrength << "\n";
    file << "  vignetteRadius: " << postProcessingPipelineParams.vignetteRadius << "\n";
    file << "  vignetteSmoothness: " << postProcessingPipelineParams.vignetteSmoothness << "\n";
    
    // Lens Flare parameters
    file << "  enableLensFlare: " << (postProcessingPipelineParams.enableLensFlare ? "true" : "false") << "\n";
    file << "  lensFlareIntensity: " << postProcessingPipelineParams.lensFlareIntensity << "\n";
    file << "  lensFlareThreshold: " << postProcessingPipelineParams.lensFlareThreshold << "\n";
    file << "  lensFlareGhostSpacing: " << postProcessingPipelineParams.lensFlareGhostSpacing << "\n";
    file << "  lensFlareGhostCount: " << postProcessingPipelineParams.lensFlareGhostCount << "\n";
    file << "  lensFlareHaloRadius: " << postProcessingPipelineParams.lensFlareHaloRadius << "\n";
    file << "  lensFlareSunSize: " << postProcessingPipelineParams.lensFlareSunSize << "\n";
    file << "  lensFlareDistortion: " << postProcessingPipelineParams.lensFlareDistortion << "\n";
    file << "  lensFlareHalfRes: " << (postProcessingPipelineParams.lensFlareHalfRes ? "true" : "false") << "\n";
    file << "  lensFlareNeighborFilter: " << (postProcessingPipelineParams.lensFlareNeighborFilter ? "true" : "false") << "\n";
    file << "  lensFlareMaxSpots: " << postProcessingPipelineParams.lensFlareMaxSpots << "\n\n";

    // Save Sky Parameters
    file << "sky:\n";
    file << "  timeOfDay: " << skyParams.timeOfDay << "\n";
    file << "  sunAxisAngle: " << skyParams.sunAxisAngle << "\n";
    file << "  sunAxisRotate: " << skyParams.sunAxisRotate << "\n";
    file << "  skyBrightness: " << skyParams.skyBrightness << "\n\n";

    // Save Character Movement Parameters
    file << "character_movement:\n";
    file << "  walkMoveForce: " << characterMovementParams.walkMoveForce << "\n";
    file << "  runMoveForce: " << characterMovementParams.runMoveForce << "\n";
    file << "  walkMaxSpeed: " << characterMovementParams.walkMaxSpeed << "\n";
    file << "  runMaxSpeed: " << characterMovementParams.runMaxSpeed << "\n";
    file << "  jumpForce: " << characterMovementParams.jumpForce << "\n";
    file << "  gravity: " << characterMovementParams.gravity << "\n";
    file << "  friction: " << characterMovementParams.friction << "\n";
    file << "  rotationSpeed: " << characterMovementParams.rotationSpeed << "\n";
    file << "  radius: " << characterMovementParams.radius << "\n";
    file << "  height: " << characterMovementParams.height << "\n\n";

    // Save Character Animation Parameters
    file << "character_animation:\n";
    file << "  walkSpeedThreshold: " << characterAnimationParams.walkSpeedThreshold << "\n";
    file << "  mediumSpeedThreshold: " << characterAnimationParams.mediumSpeedThreshold << "\n";
    file << "  runSpeedThreshold: " << characterAnimationParams.runSpeedThreshold << "\n";
    file << "  runMediumSpeedThreshold: " << characterAnimationParams.runMediumSpeedThreshold << "\n";
    file << "  animationSpeed: " << characterAnimationParams.animationSpeed << "\n";
    file << "  blendRatio: " << characterAnimationParams.blendRatio << "\n";
    file << "  placeAnimationSpeed: " << characterAnimationParams.placeAnimationSpeed << "\n\n";

    // Save Camera Movement Parameters
    file << "camera_movement:\n";
    file << "  moveSpeed: " << cameraMovementParams.moveSpeed << "\n";
    file << "  cursorMoveSpeed: " << cameraMovementParams.cursorMoveSpeed << "\n";
    file << "  fov: " << cameraMovementParams.fov << "\n";
    file << "  followSpeed: " << cameraMovementParams.followSpeed << "\n";
    file << "  followDistance: " << cameraMovementParams.followDistance << "\n";
    file << "  followHeight: " << cameraMovementParams.followHeight << "\n";
    file << "  gameplayHeight: " << cameraMovementParams.gameplayHeight << "\n\n";

    // Save Rendering Parameters
    file << "rendering:\n";
    file << "  maxFpsAllowed: " << renderingParams.maxFpsAllowed << "\n";
    file << "  targetFPS: " << renderingParams.targetFPS << "\n";
    file << "  dynamicResolution: " << (renderingParams.dynamicResolution ? "true" : "false") << "\n";
    file << "  vsync: " << (renderingParams.vsync ? "true" : "false") << "\n";
    file << "  windowWidth: " << renderingParams.windowWidth << "\n";
    file << "  windowHeight: " << renderingParams.windowHeight << "\n";
    file << "  maxRenderWidth: " << renderingParams.maxRenderWidth << "\n";
    file << "  maxRenderHeight: " << renderingParams.maxRenderHeight << "\n";
    file << "  minRenderWidth: " << renderingParams.minRenderWidth << "\n";
    file << "  minRenderHeight: " << renderingParams.minRenderHeight << "\n";

    std::cout << "Saved global settings to: " << filepath << std::endl;
}

// Private helper methods for parsing specific sections


void GlobalSettings::parseToneMappingSettings(const std::string &key, const std::string &value)
{
    if (key == "manualExposure")
        parseFloat(value, toneMappingParams.manualExposure);
    else if (key == "toneMappingCurve")
    {
        int curveValue;
        if (parseInt(value, curveValue))
            toneMappingParams.curve = static_cast<ToneMappingParams::ToneMappingCurve>(curveValue);
    }
    else if (key == "highlightDesaturation")
        parseFloat(value, toneMappingParams.highlightDesaturation);
    else if (key == "whitePoint")
        parseFloat(value, toneMappingParams.whitePoint);
    else if (key == "contrast")
        parseFloat(value, toneMappingParams.contrast);
    else if (key == "saturation")
        parseFloat(value, toneMappingParams.saturation);
    else if (key == "gain")
        parseFloat(value, toneMappingParams.gain);
    else if (key == "lift")
        parseFloat(value, toneMappingParams.lift);
    else if (key == "enableChromaticAdaptation")
        toneMappingParams.enableChromaticAdaptation = (value == "true");
}

void GlobalSettings::parsePostProcessingPipelineSettings(const std::string &key, const std::string &value)
{
    // Bloom parameters
    if (key == "enableBloom")
        postProcessingPipelineParams.enableBloom = (value == "true");
    else if (key == "bloomThreshold")
        parseFloat(value, postProcessingPipelineParams.bloomThreshold);
    else if (key == "bloomIntensity")
        parseFloat(value, postProcessingPipelineParams.bloomIntensity);
    else if (key == "bloomRadius")
        parseFloat(value, postProcessingPipelineParams.bloomRadius);
    
    // Auto-exposure parameters
    else if (key == "enableAutoExposure")
        postProcessingPipelineParams.enableAutoExposure = (value == "true");
    else if (key == "exposureSpeed")
        parseFloat(value, postProcessingPipelineParams.exposureSpeed);
    else if (key == "exposureMin")
        parseFloat(value, postProcessingPipelineParams.exposureMin);
    else if (key == "exposureMax")
        parseFloat(value, postProcessingPipelineParams.exposureMax);
    else if (key == "exposureCompensation")
        parseFloat(value, postProcessingPipelineParams.exposureCompensation);
    else if (key == "histogramMinPercent")
        parseFloat(value, postProcessingPipelineParams.histogramMinPercent);
    else if (key == "histogramMaxPercent")
        parseFloat(value, postProcessingPipelineParams.histogramMaxPercent);
    else if (key == "targetLuminance")
        parseFloat(value, postProcessingPipelineParams.targetLuminance);
    
    // Vignette parameters
    else if (key == "enableVignette")
        postProcessingPipelineParams.enableVignette = (value == "true");
    else if (key == "vignetteStrength")
        parseFloat(value, postProcessingPipelineParams.vignetteStrength);
    else if (key == "vignetteRadius")
        parseFloat(value, postProcessingPipelineParams.vignetteRadius);
    else if (key == "vignetteSmoothness")
        parseFloat(value, postProcessingPipelineParams.vignetteSmoothness);
    
    // Lens Flare parameters
    else if (key == "enableLensFlare")
        postProcessingPipelineParams.enableLensFlare = (value == "true");
    else if (key == "lensFlareIntensity")
        parseFloat(value, postProcessingPipelineParams.lensFlareIntensity);
    else if (key == "lensFlareThreshold")
        parseFloat(value, postProcessingPipelineParams.lensFlareThreshold);
    else if (key == "lensFlareGhostSpacing")
        parseFloat(value, postProcessingPipelineParams.lensFlareGhostSpacing);
    else if (key == "lensFlareGhostCount")
        parseInt(value, postProcessingPipelineParams.lensFlareGhostCount);
    else if (key == "lensFlareHaloRadius")
        parseFloat(value, postProcessingPipelineParams.lensFlareHaloRadius);
    else if (key == "lensFlareSunSize")
        parseFloat(value, postProcessingPipelineParams.lensFlareSunSize);
    else if (key == "lensFlareDistortion")
        parseFloat(value, postProcessingPipelineParams.lensFlareDistortion);
    else if (key == "lensFlareHalfRes")
        postProcessingPipelineParams.lensFlareHalfRes = (value == "true");
    else if (key == "lensFlareNeighborFilter")
        postProcessingPipelineParams.lensFlareNeighborFilter = (value == "true");
    else if (key == "lensFlareMaxSpots")
        parseInt(value, postProcessingPipelineParams.lensFlareMaxSpots);
}

void GlobalSettings::parseSkySettings(const std::string &key, const std::string &value)
{
    if (key == "timeOfDay")
        parseFloat(value, skyParams.timeOfDay);
    else if (key == "sunAxisAngle")
        parseFloat(value, skyParams.sunAxisAngle);
    else if (key == "sunAxisRotate")
        parseFloat(value, skyParams.sunAxisRotate);
    else if (key == "skyBrightness")
        parseFloat(value, skyParams.skyBrightness);
}

void GlobalSettings::parseCharacterMovementSettings(const std::string &key, const std::string &value)
{
    if (key == "walkMoveForce")
        parseFloat(value, characterMovementParams.walkMoveForce);
    else if (key == "runMoveForce")
        parseFloat(value, characterMovementParams.runMoveForce);
    else if (key == "walkMaxSpeed")
        parseFloat(value, characterMovementParams.walkMaxSpeed);
    else if (key == "runMaxSpeed")
        parseFloat(value, characterMovementParams.runMaxSpeed);
    else if (key == "jumpForce")
        parseFloat(value, characterMovementParams.jumpForce);
    else if (key == "gravity")
        parseFloat(value, characterMovementParams.gravity);
    else if (key == "friction")
        parseFloat(value, characterMovementParams.friction);
    else if (key == "rotationSpeed")
        parseFloat(value, characterMovementParams.rotationSpeed);
    else if (key == "radius")
        parseFloat(value, characterMovementParams.radius);
    else if (key == "height")
        parseFloat(value, characterMovementParams.height);
}

void GlobalSettings::parseCharacterAnimationSettings(const std::string &key, const std::string &value)
{
    if (key == "walkSpeedThreshold")
        parseFloat(value, characterAnimationParams.walkSpeedThreshold);
    else if (key == "mediumSpeedThreshold")
        parseFloat(value, characterAnimationParams.mediumSpeedThreshold);
    else if (key == "runSpeedThreshold")
        parseFloat(value, characterAnimationParams.runSpeedThreshold);
    else if (key == "runMediumSpeedThreshold")
        parseFloat(value, characterAnimationParams.runMediumSpeedThreshold);
    else if (key == "animationSpeed")
        parseFloat(value, characterAnimationParams.animationSpeed);
    else if (key == "blendRatio")
        parseFloat(value, characterAnimationParams.blendRatio);
    else if (key == "placeAnimationSpeed")
        parseFloat(value, characterAnimationParams.placeAnimationSpeed);
}

void GlobalSettings::parseCameraMovementSettings(const std::string &key, const std::string &value)
{
    if (key == "moveSpeed")
        parseFloat(value, cameraMovementParams.moveSpeed);
    else if (key == "cursorMoveSpeed")
        parseFloat(value, cameraMovementParams.cursorMoveSpeed);
    else if (key == "fov")
        parseFloat(value, cameraMovementParams.fov);
    else if (key == "followSpeed")
        parseFloat(value, cameraMovementParams.followSpeed);
    else if (key == "followDistance")
        parseFloat(value, cameraMovementParams.followDistance);
    else if (key == "followHeight")
        parseFloat(value, cameraMovementParams.followHeight);
    else if (key == "gameplayHeight")
        parseFloat(value, cameraMovementParams.gameplayHeight);
}

void GlobalSettings::parseRenderingSettings(const std::string &key, const std::string &value)
{
    if (key == "maxFpsAllowed")
        parseFloat(value, renderingParams.maxFpsAllowed);
    else if (key == "targetFPS")
        parseFloat(value, renderingParams.targetFPS);
    else if (key == "dynamicResolution")
        parseBool(value, renderingParams.dynamicResolution);
    else if (key == "vsync")
        parseBool(value, renderingParams.vsync);
    else if (key == "windowWidth")
        parseInt(value, renderingParams.windowWidth);
    else if (key == "windowHeight")
        parseInt(value, renderingParams.windowHeight);
    else if (key == "maxRenderWidth")
        parseInt(value, renderingParams.maxRenderWidth);
    else if (key == "maxRenderHeight")
        parseInt(value, renderingParams.maxRenderHeight);
    else if (key == "minRenderWidth")
        parseInt(value, renderingParams.minRenderWidth);
    else if (key == "minRenderHeight")
        parseInt(value, renderingParams.minRenderHeight);
}

// Time management is now handled by Backend's Timer.h




void GlobalSettings::parseDenosingSettings(const std::string& key, const std::string& value)
{
    if (key == "enableHitDistanceReconstruction") parseBool(value, denoisingParams.enableHitDistanceReconstruction);
    else if (key == "enablePrePass") parseBool(value, denoisingParams.enablePrePass);
    else if (key == "enableTemporalAccumulation") parseBool(value, denoisingParams.enableTemporalAccumulation);
    else if (key == "enableHistoryFix") parseBool(value, denoisingParams.enableHistoryFix);
    else if (key == "enableHistoryClamping") parseBool(value, denoisingParams.enableHistoryClamping);
    else if (key == "enableSpatialFiltering") parseBool(value, denoisingParams.enableSpatialFiltering);
    else if (key == "maxAccumulatedFrameNum") parseFloat(value, denoisingParams.maxAccumulatedFrameNum);
    else if (key == "maxFastAccumulatedFrameNum") parseFloat(value, denoisingParams.maxFastAccumulatedFrameNum);
    else if (key == "phiLuminance") parseFloat(value, denoisingParams.phiLuminance);
    else if (key == "lobeAngleFraction") parseFloat(value, denoisingParams.lobeAngleFraction);
    else if (key == "roughnessFraction") parseFloat(value, denoisingParams.roughnessFraction);
    else if (key == "depthThreshold") parseFloat(value, denoisingParams.depthThreshold);
    else if (key == "atrousIterationNum") parseInt(value, denoisingParams.atrousIterationNum);
    else if (key == "disocclusionThreshold") parseFloat(value, denoisingParams.disocclusionThreshold);
    else if (key == "disocclusionThresholdAlternate") parseFloat(value, denoisingParams.disocclusionThresholdAlternate);
    else if (key == "denoisingRange") parseFloat(value, denoisingParams.denoisingRange);
}


