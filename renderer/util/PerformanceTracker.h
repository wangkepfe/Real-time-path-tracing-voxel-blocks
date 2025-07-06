#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>

struct PerformanceData
{
    std::string timestamp;
    std::string comment;

    // Timing data in milliseconds
    double wholeFrameTime = 0.0;
    double scenePreparationTime = 0.0;
    double rendererUpdateTime = 0.0;
    double pathTracingTime = 0.0;
    double denoiserTime = 0.0;
    double postProcessingTime = 0.0;

    // Additional metrics
    int frameNumber = 0;
    int width = 0;
    int height = 0;
    double totalPixels = 0.0;
};

class PerformanceTracker
{
public:
    static PerformanceTracker& Get()
    {
        static PerformanceTracker instance;
        return instance;
    }

    void startTiming(const std::string& name)
    {
        timingPoints[name] = std::chrono::high_resolution_clock::now();
    }

    double endTiming(const std::string& name)
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto it = timingPoints.find(name);
        if (it != timingPoints.end())
        {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - it->second);
            double milliseconds = duration.count() / 1000.0;
            timingPoints.erase(it);
            return milliseconds;
        }
        return 0.0;
    }

    void beginFrame(int frameNum, int w, int h, const std::string& comment = "")
    {
        currentFrame.frameNumber = frameNum;
        currentFrame.width = w;
        currentFrame.height = h;
        currentFrame.totalPixels = w * h;
        currentFrame.comment = comment;

        // Generate timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        currentFrame.timestamp = ss.str();

        startTiming("wholeFrame");
    }

    void endFrame()
    {
        currentFrame.wholeFrameTime = endTiming("wholeFrame");
        frameData.push_back(currentFrame);
        currentFrame = PerformanceData(); // Reset for next frame
    }

    void setScenePreparationTime(double time) { currentFrame.scenePreparationTime = time; }
    void setRendererUpdateTime(double time) { currentFrame.rendererUpdateTime = time; }
    void setPathTracingTime(double time) { currentFrame.pathTracingTime = time; }
    void setDenoiserTime(double time) { currentFrame.denoiserTime = time; }
    void setPostProcessingTime(double time) { currentFrame.postProcessingTime = time; }

        void saveReport(const std::string& filename = "data/perf/performance_report.txt", const std::string& runComment = "")
    {
        if (frameData.empty()) return;

        std::ofstream file(filename, std::ios::app); // Append mode
        if (!file.is_open())
        {
            // Try to create directory first
            std::string dir = filename.substr(0, filename.find_last_of("/\\"));
            std::string createDirCmd = "mkdir -p " + dir;
            system(createDirCmd.c_str());

            file.open(filename, std::ios::app);
            if (!file.is_open())
            {
                return; // Failed to create/open file
            }
        }

        // Write header if file is new/empty
        file.seekp(0, std::ios::end);
        if (file.tellp() == 0)
        {
            file << "# Performance Report - Real-time Path Tracing Voxel Renderer (Run Summary)\n";
            file << "# Format: Timestamp         | Frames | Resolution | WholeFrame | StdDev | ScenePrep | RendererUpd | PathTrace | Denoiser | PostProc | Comment\n";
            file << "# ===============================================================================================================================\n";
        }

                // Calculate averages and standard deviation
        double avgWholeFrame = 0.0, avgScenePrep = 0.0, avgRendererUpdate = 0.0;
        double avgPathTracing = 0.0, avgDenoiser = 0.0, avgPostProcess = 0.0;

        for (const auto& data : frameData)
        {
            avgWholeFrame += data.wholeFrameTime;
            avgScenePrep += data.scenePreparationTime;
            avgRendererUpdate += data.rendererUpdateTime;
            avgPathTracing += data.pathTracingTime;
            avgDenoiser += data.denoiserTime;
            avgPostProcess += data.postProcessingTime;
        }

        size_t numFrames = frameData.size();
        avgWholeFrame /= numFrames;
        avgScenePrep /= numFrames;
        avgRendererUpdate /= numFrames;
        avgPathTracing /= numFrames;
        avgDenoiser /= numFrames;
        avgPostProcess /= numFrames;

        // Calculate standard deviation for whole frame time
        double variance = 0.0;
        for (const auto& data : frameData)
        {
            double diff = data.wholeFrameTime - avgWholeFrame;
            variance += diff * diff;
        }
        variance /= numFrames;
        double stdDev = std::sqrt(variance);

        // Write summary line with fixed-width columns
        file << std::fixed << std::setprecision(2);
        file << std::setw(19) << std::left << frameData[0].timestamp << " | ";
        file << std::setw(6) << std::right << numFrames << " | ";
        file << std::setw(10) << std::left << (std::to_string(frameData[0].width) + "x" + std::to_string(frameData[0].height)) << " | ";
        file << std::setw(10) << std::right << avgWholeFrame << " | ";
        file << std::setw(6) << std::right << stdDev << " | ";
        file << std::setw(9) << std::right << avgScenePrep << " | ";
        file << std::setw(11) << std::right << avgRendererUpdate << " | ";
        file << std::setw(9) << std::right << avgPathTracing << " | ";
        file << std::setw(8) << std::right << avgDenoiser << " | ";
        file << std::setw(8) << std::right << avgPostProcess << " | ";
        file << runComment << std::endl;

        file.close();

        // Clear stored data after saving
        frameData.clear();
    }

    void printCurrentFrameStats()
    {
        const auto& data = currentFrame;
        std::cout << "\n=== Frame " << data.frameNumber << " Performance ===" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Resolution: " << data.width << "x" << data.height << std::endl;
        std::cout << "Whole Frame:      " << data.wholeFrameTime << " ms" << std::endl;
        std::cout << "Scene Prep:       " << data.scenePreparationTime << " ms" << std::endl;
        std::cout << "Renderer Update:  " << data.rendererUpdateTime << " ms" << std::endl;
        std::cout << "Path Tracing:     " << data.pathTracingTime << " ms" << std::endl;
        std::cout << "Denoiser:         " << data.denoiserTime << " ms" << std::endl;
        std::cout << "Post Processing:  " << data.postProcessingTime << " ms" << std::endl;
        std::cout << std::endl;
    }

private:
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> timingPoints;
    std::vector<PerformanceData> frameData;
    PerformanceData currentFrame;

    PerformanceTracker() = default;
};