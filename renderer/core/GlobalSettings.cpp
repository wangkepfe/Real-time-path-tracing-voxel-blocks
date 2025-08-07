#include "GlobalSettings.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <chrono>
#ifndef OFFLINE_MODE
#include <GLFW/glfw3.h>
#endif

// Helper functions for YAML parsing (similar to SceneConfig.cpp)
namespace 
{
    std::string trim(const std::string& str)
    {
        size_t start = str.find_first_not_of(" \t\r\n");
        if (start == std::string::npos)
            return "";

        size_t end = str.find_last_not_of(" \t\r\n");
        return str.substr(start, end - start + 1);
    }

    bool parseFloat(const std::string& value, float& result)
    {
        try
        {
            result = std::stof(trim(value));
            return true;
        }
        catch (const std::exception&)
        {
            return false;
        }
    }

    bool parseInt(const std::string& value, int& result)
    {
        try
        {
            result = std::stoi(trim(value));
            return true;
        }
        catch (const std::exception&)
        {
            return false;
        }
    }

    bool parseBool(const std::string& value, bool& result)
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

bool GlobalSettings::LoadFromYAML(const std::string& filepath)
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
            parsePostProcessSettings(key, value);
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

void GlobalSettings::SaveToYAML(const std::string& filepath) const
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
    file << "  enableAntiFirefly: " << (denoisingParams.enableAntiFirefly ? "true" : "false") << "\n";
    file << "  enableRoughnessEdgeStopping: " << (denoisingParams.enableRoughnessEdgeStopping ? "true" : "false") << "\n";
    file << "  maxAccumulatedFrameNum: " << denoisingParams.maxAccumulatedFrameNum << "\n";
    file << "  maxFastAccumulatedFrameNum: " << denoisingParams.maxFastAccumulatedFrameNum << "\n";
    file << "  historyFixFrameNum: " << denoisingParams.historyFixFrameNum << "\n";
    file << "  prepassBlurRadius: " << denoisingParams.prepassBlurRadius << "\n";
    file << "  minHitDistanceWeight: " << denoisingParams.minHitDistanceWeight << "\n";
    file << "  phiLuminance: " << denoisingParams.phiLuminance << "\n";
    file << "  lobeAngleFraction: " << denoisingParams.lobeAngleFraction << "\n";
    file << "  roughnessFraction: " << denoisingParams.roughnessFraction << "\n";
    file << "  depthThreshold: " << denoisingParams.depthThreshold << "\n";
    file << "  minLuminanceWeight: " << denoisingParams.minLuminanceWeight << "\n";
    file << "  atrousIterationNum: " << denoisingParams.atrousIterationNum << "\n";
    file << "  luminanceEdgeStoppingRelaxation: " << denoisingParams.luminanceEdgeStoppingRelaxation << "\n";
    file << "  normalEdgeStoppingRelaxation: " << denoisingParams.normalEdgeStoppingRelaxation << "\n";
    file << "  roughnessEdgeStoppingRelaxation: " << denoisingParams.roughnessEdgeStoppingRelaxation << "\n";
    file << "  antilagAccelerationAmount: " << denoisingParams.antilagAccelerationAmount << "\n";
    file << "  antilagSpatialSigmaScale: " << denoisingParams.antilagSpatialSigmaScale << "\n";
    file << "  antilagTemporalSigmaScale: " << denoisingParams.antilagTemporalSigmaScale << "\n";
    file << "  antilagResetAmount: " << denoisingParams.antilagResetAmount << "\n";
    file << "  historyClampingColorBoxSigmaScale: " << denoisingParams.historyClampingColorBoxSigmaScale << "\n";
    file << "  historyFixBasePixelStride: " << denoisingParams.historyFixBasePixelStride << "\n";
    file << "  spatialVarianceEstimationHistoryThreshold: " << denoisingParams.spatialVarianceEstimationHistoryThreshold << "\n";
    file << "  disocclusionThreshold: " << denoisingParams.disocclusionThreshold << "\n";
    file << "  disocclusionThresholdAlternate: " << denoisingParams.disocclusionThresholdAlternate << "\n";
    file << "  confidenceDrivenRelaxationMultiplier: " << denoisingParams.confidenceDrivenRelaxationMultiplier << "\n";
    file << "  confidenceDrivenLuminanceEdgeStoppingRelaxation: " << denoisingParams.confidenceDrivenLuminanceEdgeStoppingRelaxation << "\n";
    file << "  confidenceDrivenNormalEdgeStoppingRelaxation: " << denoisingParams.confidenceDrivenNormalEdgeStoppingRelaxation << "\n";
    file << "  denoisingRange: " << denoisingParams.denoisingRange << "\n\n";

    // Save Post Processing Parameters
    file << "postprocess:\n";
    file << "  postGain: " << postProcessParams.postGain << "\n";
    file << "  gain: " << postProcessParams.gain << "\n";
    file << "  maxWhite: " << postProcessParams.maxWhite << "\n\n";

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
void GlobalSettings::parseDenosingSettings(const std::string& key, const std::string& value)
{
    if (key == "enableHitDistanceReconstruction") parseBool(value, denoisingParams.enableHitDistanceReconstruction);
    else if (key == "enablePrePass") parseBool(value, denoisingParams.enablePrePass);
    else if (key == "enableTemporalAccumulation") parseBool(value, denoisingParams.enableTemporalAccumulation);
    else if (key == "enableHistoryFix") parseBool(value, denoisingParams.enableHistoryFix);
    else if (key == "enableHistoryClamping") parseBool(value, denoisingParams.enableHistoryClamping);
    else if (key == "enableSpatialFiltering") parseBool(value, denoisingParams.enableSpatialFiltering);
    else if (key == "enableAntiFirefly") parseBool(value, denoisingParams.enableAntiFirefly);
    else if (key == "enableRoughnessEdgeStopping") parseBool(value, denoisingParams.enableRoughnessEdgeStopping);
    else if (key == "maxAccumulatedFrameNum") parseFloat(value, denoisingParams.maxAccumulatedFrameNum);
    else if (key == "maxFastAccumulatedFrameNum") parseFloat(value, denoisingParams.maxFastAccumulatedFrameNum);
    else if (key == "historyFixFrameNum") parseFloat(value, denoisingParams.historyFixFrameNum);
    else if (key == "prepassBlurRadius") parseFloat(value, denoisingParams.prepassBlurRadius);
    else if (key == "minHitDistanceWeight") parseFloat(value, denoisingParams.minHitDistanceWeight);
    else if (key == "phiLuminance") parseFloat(value, denoisingParams.phiLuminance);
    else if (key == "lobeAngleFraction") parseFloat(value, denoisingParams.lobeAngleFraction);
    else if (key == "roughnessFraction") parseFloat(value, denoisingParams.roughnessFraction);
    else if (key == "depthThreshold") parseFloat(value, denoisingParams.depthThreshold);
    else if (key == "minLuminanceWeight") parseFloat(value, denoisingParams.minLuminanceWeight);
    else if (key == "atrousIterationNum") parseInt(value, denoisingParams.atrousIterationNum);
    else if (key == "luminanceEdgeStoppingRelaxation") parseFloat(value, denoisingParams.luminanceEdgeStoppingRelaxation);
    else if (key == "normalEdgeStoppingRelaxation") parseFloat(value, denoisingParams.normalEdgeStoppingRelaxation);
    else if (key == "roughnessEdgeStoppingRelaxation") parseFloat(value, denoisingParams.roughnessEdgeStoppingRelaxation);
    else if (key == "antilagAccelerationAmount") parseFloat(value, denoisingParams.antilagAccelerationAmount);
    else if (key == "antilagSpatialSigmaScale") parseFloat(value, denoisingParams.antilagSpatialSigmaScale);
    else if (key == "antilagTemporalSigmaScale") parseFloat(value, denoisingParams.antilagTemporalSigmaScale);
    else if (key == "antilagResetAmount") parseFloat(value, denoisingParams.antilagResetAmount);
    else if (key == "historyClampingColorBoxSigmaScale") parseFloat(value, denoisingParams.historyClampingColorBoxSigmaScale);
    else if (key == "historyFixBasePixelStride") parseInt(value, denoisingParams.historyFixBasePixelStride);
    else if (key == "spatialVarianceEstimationHistoryThreshold") parseInt(value, denoisingParams.spatialVarianceEstimationHistoryThreshold);
    else if (key == "disocclusionThreshold") parseFloat(value, denoisingParams.disocclusionThreshold);
    else if (key == "disocclusionThresholdAlternate") parseFloat(value, denoisingParams.disocclusionThresholdAlternate);
    else if (key == "confidenceDrivenRelaxationMultiplier") parseFloat(value, denoisingParams.confidenceDrivenRelaxationMultiplier);
    else if (key == "confidenceDrivenLuminanceEdgeStoppingRelaxation") parseFloat(value, denoisingParams.confidenceDrivenLuminanceEdgeStoppingRelaxation);
    else if (key == "confidenceDrivenNormalEdgeStoppingRelaxation") parseFloat(value, denoisingParams.confidenceDrivenNormalEdgeStoppingRelaxation);
    else if (key == "denoisingRange") parseFloat(value, denoisingParams.denoisingRange);
}

void GlobalSettings::parsePostProcessSettings(const std::string& key, const std::string& value)
{
    if (key == "postGain") parseFloat(value, postProcessParams.postGain);
    else if (key == "gain") parseFloat(value, postProcessParams.gain);
    else if (key == "maxWhite") parseFloat(value, postProcessParams.maxWhite);
}

void GlobalSettings::parseSkySettings(const std::string& key, const std::string& value)
{
    if (key == "timeOfDay") parseFloat(value, skyParams.timeOfDay);
    else if (key == "sunAxisAngle") parseFloat(value, skyParams.sunAxisAngle);
    else if (key == "sunAxisRotate") parseFloat(value, skyParams.sunAxisRotate);
    else if (key == "skyBrightness") parseFloat(value, skyParams.skyBrightness);
}

void GlobalSettings::parseCharacterMovementSettings(const std::string& key, const std::string& value)
{
    if (key == "walkMoveForce") parseFloat(value, characterMovementParams.walkMoveForce);
    else if (key == "runMoveForce") parseFloat(value, characterMovementParams.runMoveForce);
    else if (key == "walkMaxSpeed") parseFloat(value, characterMovementParams.walkMaxSpeed);
    else if (key == "runMaxSpeed") parseFloat(value, characterMovementParams.runMaxSpeed);
    else if (key == "jumpForce") parseFloat(value, characterMovementParams.jumpForce);
    else if (key == "gravity") parseFloat(value, characterMovementParams.gravity);
    else if (key == "friction") parseFloat(value, characterMovementParams.friction);
    else if (key == "rotationSpeed") parseFloat(value, characterMovementParams.rotationSpeed);
    else if (key == "radius") parseFloat(value, characterMovementParams.radius);
    else if (key == "height") parseFloat(value, characterMovementParams.height);
}

void GlobalSettings::parseCharacterAnimationSettings(const std::string& key, const std::string& value)
{
    if (key == "walkSpeedThreshold") parseFloat(value, characterAnimationParams.walkSpeedThreshold);
    else if (key == "mediumSpeedThreshold") parseFloat(value, characterAnimationParams.mediumSpeedThreshold);
    else if (key == "runSpeedThreshold") parseFloat(value, characterAnimationParams.runSpeedThreshold);
    else if (key == "runMediumSpeedThreshold") parseFloat(value, characterAnimationParams.runMediumSpeedThreshold);
    else if (key == "animationSpeed") parseFloat(value, characterAnimationParams.animationSpeed);
    else if (key == "blendRatio") parseFloat(value, characterAnimationParams.blendRatio);
    else if (key == "placeAnimationSpeed") parseFloat(value, characterAnimationParams.placeAnimationSpeed);
}

void GlobalSettings::parseCameraMovementSettings(const std::string& key, const std::string& value)
{
    if (key == "moveSpeed") parseFloat(value, cameraMovementParams.moveSpeed);
    else if (key == "cursorMoveSpeed") parseFloat(value, cameraMovementParams.cursorMoveSpeed);
    else if (key == "fov") parseFloat(value, cameraMovementParams.fov);
    else if (key == "followSpeed") parseFloat(value, cameraMovementParams.followSpeed);
    else if (key == "followDistance") parseFloat(value, cameraMovementParams.followDistance);
    else if (key == "followHeight") parseFloat(value, cameraMovementParams.followHeight);
    else if (key == "gameplayHeight") parseFloat(value, cameraMovementParams.gameplayHeight);
}

void GlobalSettings::parseRenderingSettings(const std::string& key, const std::string& value)
{
    if (key == "maxFpsAllowed") parseFloat(value, renderingParams.maxFpsAllowed);
    else if (key == "targetFPS") parseFloat(value, renderingParams.targetFPS);
    else if (key == "dynamicResolution") parseBool(value, renderingParams.dynamicResolution);
    else if (key == "vsync") parseBool(value, renderingParams.vsync);
    else if (key == "windowWidth") parseInt(value, renderingParams.windowWidth);
    else if (key == "windowHeight") parseInt(value, renderingParams.windowHeight);
    else if (key == "maxRenderWidth") parseInt(value, renderingParams.maxRenderWidth);
    else if (key == "maxRenderHeight") parseInt(value, renderingParams.maxRenderHeight);
    else if (key == "minRenderWidth") parseInt(value, renderingParams.minRenderWidth);
    else if (key == "minRenderHeight") parseInt(value, renderingParams.minRenderHeight);
}

// Static member definitions for time management
float GlobalSettings::currentTime = 0.0f;
float GlobalSettings::lastTime = -1.0f;
float GlobalSettings::deltaTime = 0.0f;
int GlobalSettings::frameCounter = 0;

// Unified time management implementation
float GlobalSettings::GetGameTime()
{
    if (IsOfflineMode())
    {
        const float targetFPS = 30.0f;
        return frameCounter * (1.0f / targetFPS);
    }
    else
    {
#ifdef OFFLINE_MODE
        // For offline builds, use std::chrono instead of GLFW
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<float>(currentTime - startTime);
        return elapsed.count();
#else
        return static_cast<float>(glfwGetTime());
#endif
    }
}

float GlobalSettings::GetDeltaTime()
{
    return deltaTime;
}

void GlobalSettings::UpdateTime()
{
    if (IsOfflineMode())
    {
        frameCounter++;
        const float targetFPS = 30.0f;
        deltaTime = 1.0f / targetFPS;
        currentTime = frameCounter * deltaTime;
    }
    else
    {
#ifdef OFFLINE_MODE
        // For offline builds, use std::chrono instead of GLFW
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currentTimePoint = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<float>(currentTimePoint - startTime);
        currentTime = elapsed.count();
#else
        currentTime = static_cast<float>(glfwGetTime());
#endif
        
        if (lastTime < 0.0f)
        {
            deltaTime = 1.0f / 60.0f; // Default for first frame
        }
        else
        {
            deltaTime = currentTime - lastTime;
        }
        lastTime = currentTime;
    }
}