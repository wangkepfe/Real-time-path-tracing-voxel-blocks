#pragma once

#include <string>
#include <vector>
#include <cstdint>

struct ImageDiffResult
{
    // Simple pixel difference metrics
    int differentPixels = 0;
    int totalPixels = 0;
    float pixelDifferenceRatio = 0.0f;

    // RMSE (Root Mean Square Error)
    float rmse = 0.0f;

    // SSIM (Structural Similarity Index)
    float ssim = 0.0f;

    // Overall assessment
    bool isIdentical = false;
    bool isVeryClose = false; // SSIM > 0.99 and RMSE < 1.0
    bool isClose = false;     // SSIM > 0.95 and RMSE < 5.0

    void print() const;
};

struct ImageData
{
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> data;

    bool loadFromFile(const std::string& filename);
    bool saveToFile(const std::string& filename) const;
    bool isValid() const { return width > 0 && height > 0 && !data.empty(); }
};

class ImageDiff
{
public:
    // Main comparison function
    static ImageDiffResult compare(const std::string& imageA, const std::string& imageB);
    static ImageDiffResult compare(const ImageData& imageA, const ImageData& imageB);

    // Generate visual diff image (highlights differences)
    static bool generateDiffImage(const std::string& imageA, const std::string& imageB,
                                  const std::string& outputDiffImage);
    static bool generateDiffImage(const ImageData& imageA, const ImageData& imageB,
                                  const std::string& outputDiffImage);

    // Individual metric calculations
    static int countDifferentPixels(const ImageData& imageA, const ImageData& imageB, float threshold = 0.01f);
    static float calculateRMSE(const ImageData& imageA, const ImageData& imageB);
    static float calculateSSIM(const ImageData& imageA, const ImageData& imageB);

private:
    // Helper functions for SSIM calculation
    static float calculateMean(const std::vector<float>& values);
    static float calculateVariance(const std::vector<float>& values, float mean);
    static float calculateCovariance(const std::vector<float>& valuesA, const std::vector<float>& valuesB,
                                     float meanA, float meanB);
    static std::vector<float> convertToGrayscale(const ImageData& image);
    static std::vector<float> applyGaussianFilter(const std::vector<float>& image, int width, int height);
};