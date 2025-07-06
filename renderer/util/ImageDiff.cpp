#include "util/ImageDiff.h"

#include "ext/stb/stb_image.h"
#include "ext/stb/stb_image_write.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>

void ImageDiffResult::print() const
{
    std::cout << "=== Image Comparison Results ===" << std::endl;
    std::cout << "Different pixels: " << differentPixels << " / " << totalPixels
              << " (" << std::fixed << std::setprecision(2) << (pixelDifferenceRatio * 100.0f) << "%)" << std::endl;
    std::cout << "RMSE: " << std::fixed << std::setprecision(4) << rmse << std::endl;
    std::cout << "SSIM: " << std::fixed << std::setprecision(6) << ssim << std::endl;

    if (isIdentical) {
        std::cout << "Assessment: IDENTICAL" << std::endl;
    } else if (isVeryClose) {
        std::cout << "Assessment: VERY CLOSE (excellent match)" << std::endl;
    } else if (isClose) {
        std::cout << "Assessment: CLOSE (good match)" << std::endl;
    } else {
        std::cout << "Assessment: DIFFERENT (significant differences detected)" << std::endl;
    }
}

bool ImageData::loadFromFile(const std::string& filename)
{
    stbi_set_flip_vertically_on_load(false);
    unsigned char* stb_data = stbi_load(filename.c_str(), &width, &height, &channels, 0);

    if (!stb_data) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return false;
    }

    // Copy data to our vector
    int size = width * height * channels;
    data.resize(size);
    std::memcpy(data.data(), stb_data, size);

    stbi_image_free(stb_data);
    return true;
}

bool ImageData::saveToFile(const std::string& filename) const
{
    if (!isValid()) {
        std::cerr << "Cannot save invalid image data" << std::endl;
        return false;
    }

    // Determine file format from extension
    std::string ext = filename.substr(filename.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    int result = 0;
    if (ext == "png") {
        result = stbi_write_png(filename.c_str(), width, height, channels, data.data(), width * channels);
    } else if (ext == "jpg" || ext == "jpeg") {
        result = stbi_write_jpg(filename.c_str(), width, height, channels, data.data(), 90);
    } else if (ext == "tga") {
        result = stbi_write_tga(filename.c_str(), width, height, channels, data.data());
    } else if (ext == "bmp") {
        result = stbi_write_bmp(filename.c_str(), width, height, channels, data.data());
    } else {
        std::cerr << "Unsupported image format: " << ext << std::endl;
        return false;
    }

    return result != 0;
}

ImageDiffResult ImageDiff::compare(const std::string& imageA, const std::string& imageB)
{
    ImageData imgA, imgB;

    if (!imgA.loadFromFile(imageA)) {
        std::cerr << "Failed to load image A: " << imageA << std::endl;
        return ImageDiffResult{};
    }

    if (!imgB.loadFromFile(imageB)) {
        std::cerr << "Failed to load image B: " << imageB << std::endl;
        return ImageDiffResult{};
    }

    return compare(imgA, imgB);
}

ImageDiffResult ImageDiff::compare(const ImageData& imageA, const ImageData& imageB)
{
    ImageDiffResult result;

    // Check if images have same dimensions
    if (imageA.width != imageB.width || imageA.height != imageB.height) {
        std::cerr << "Images have different dimensions: "
                  << imageA.width << "x" << imageA.height << " vs "
                  << imageB.width << "x" << imageB.height << std::endl;
        return result;
    }

    result.totalPixels = imageA.width * imageA.height;

    // Calculate pixel differences
    result.differentPixels = countDifferentPixels(imageA, imageB);
    result.pixelDifferenceRatio = static_cast<float>(result.differentPixels) / result.totalPixels;

    // Calculate RMSE
    result.rmse = calculateRMSE(imageA, imageB);

    // Calculate SSIM
    result.ssim = calculateSSIM(imageA, imageB);

    // Determine assessment
    result.isIdentical = (result.differentPixels == 0);
    result.isVeryClose = (result.ssim > 0.99f && result.rmse < 1.0f);
    result.isClose = (result.ssim > 0.95f && result.rmse < 5.0f);

    return result;
}

bool ImageDiff::generateDiffImage(const std::string& imageA, const std::string& imageB,
                                  const std::string& outputDiffImage)
{
    ImageData imgA, imgB;

    if (!imgA.loadFromFile(imageA) || !imgB.loadFromFile(imageB)) {
        return false;
    }

    return generateDiffImage(imgA, imgB, outputDiffImage);
}

bool ImageDiff::generateDiffImage(const ImageData& imageA, const ImageData& imageB,
                                  const std::string& outputDiffImage)
{
    if (imageA.width != imageB.width || imageA.height != imageB.height) {
        std::cerr << "Cannot generate diff for images of different sizes" << std::endl;
        return false;
    }

    ImageData diffImage;
    diffImage.width = imageA.width;
    diffImage.height = imageA.height;
    diffImage.channels = 3; // RGB output
    diffImage.data.resize(diffImage.width * diffImage.height * 3);

    int channels = std::min(imageA.channels, imageB.channels);

    for (int y = 0; y < diffImage.height; y++) {
        for (int x = 0; x < diffImage.width; x++) {
            int idx = (y * diffImage.width + x);
            int idxA = idx * imageA.channels;
            int idxB = idx * imageB.channels;
            int idxDiff = idx * 3;

            // Calculate absolute difference for each channel
            float diffR = 0, diffG = 0, diffB = 0;

            if (channels >= 1) {
                diffR = std::abs(static_cast<int>(imageA.data[idxA]) - static_cast<int>(imageB.data[idxB]));
            }
            if (channels >= 2) {
                diffG = std::abs(static_cast<int>(imageA.data[idxA + 1]) - static_cast<int>(imageB.data[idxB + 1]));
            }
            if (channels >= 3) {
                diffB = std::abs(static_cast<int>(imageA.data[idxA + 2]) - static_cast<int>(imageB.data[idxB + 2]));
            } else {
                diffG = diffR; // Grayscale
                diffB = diffR;
            }

            // Enhance differences for visibility (amplify by 3x, cap at 255)
            diffImage.data[idxDiff] = static_cast<uint8_t>(std::min(255.0f, diffR * 3.0f));
            diffImage.data[idxDiff + 1] = static_cast<uint8_t>(std::min(255.0f, diffG * 3.0f));
            diffImage.data[idxDiff + 2] = static_cast<uint8_t>(std::min(255.0f, diffB * 3.0f));
        }
    }

    return diffImage.saveToFile(outputDiffImage);
}

int ImageDiff::countDifferentPixels(const ImageData& imageA, const ImageData& imageB, float threshold)
{
    if (imageA.width != imageB.width || imageA.height != imageB.height) {
        return -1;
    }

    int differentPixels = 0;
    int channels = std::min(imageA.channels, imageB.channels);

    for (int y = 0; y < imageA.height; y++) {
        for (int x = 0; x < imageA.width; x++) {
            int idx = (y * imageA.width + x);
            bool isDifferent = false;

            for (int c = 0; c < channels; c++) {
                int idxA = idx * imageA.channels + c;
                int idxB = idx * imageB.channels + c;

                float diff = std::abs(static_cast<float>(imageA.data[idxA]) - static_cast<float>(imageB.data[idxB])) / 255.0f;
                if (diff > threshold) {
                    isDifferent = true;
                    break;
                }
            }

            if (isDifferent) {
                differentPixels++;
            }
        }
    }

    return differentPixels;
}

float ImageDiff::calculateRMSE(const ImageData& imageA, const ImageData& imageB)
{
    if (imageA.width != imageB.width || imageA.height != imageB.height) {
        return -1.0f;
    }

    double sumSquaredError = 0.0;
    int totalSamples = 0;
    int channels = std::min(imageA.channels, imageB.channels);

    for (int y = 0; y < imageA.height; y++) {
        for (int x = 0; x < imageA.width; x++) {
            int idx = (y * imageA.width + x);

            for (int c = 0; c < channels; c++) {
                int idxA = idx * imageA.channels + c;
                int idxB = idx * imageB.channels + c;

                double diff = static_cast<double>(imageA.data[idxA]) - static_cast<double>(imageB.data[idxB]);
                sumSquaredError += diff * diff;
                totalSamples++;
            }
        }
    }

    return static_cast<float>(std::sqrt(sumSquaredError / totalSamples));
}

float ImageDiff::calculateSSIM(const ImageData& imageA, const ImageData& imageB)
{
    if (imageA.width != imageB.width || imageA.height != imageB.height) {
        return -1.0f;
    }

    // Convert to grayscale for SSIM calculation
    std::vector<float> grayA = convertToGrayscale(imageA);
    std::vector<float> grayB = convertToGrayscale(imageB);

    // Apply Gaussian filter to reduce noise
    grayA = applyGaussianFilter(grayA, imageA.width, imageA.height);
    grayB = applyGaussianFilter(grayB, imageB.width, imageB.height);

    // SSIM parameters
    const float K1 = 0.01f;
    const float K2 = 0.03f;
    const float L = 255.0f; // Dynamic range
    const float C1 = (K1 * L) * (K1 * L);
    const float C2 = (K2 * L) * (K2 * L);

    // Calculate means
    float meanA = calculateMean(grayA);
    float meanB = calculateMean(grayB);

    // Calculate variances and covariance
    float varA = calculateVariance(grayA, meanA);
    float varB = calculateVariance(grayB, meanB);
    float covarAB = calculateCovariance(grayA, grayB, meanA, meanB);

    // Calculate SSIM
    float numerator = (2 * meanA * meanB + C1) * (2 * covarAB + C2);
    float denominator = (meanA * meanA + meanB * meanB + C1) * (varA + varB + C2);

    return numerator / denominator;
}

float ImageDiff::calculateMean(const std::vector<float>& values)
{
    float sum = 0.0f;
    for (float value : values) {
        sum += value;
    }
    return sum / values.size();
}

float ImageDiff::calculateVariance(const std::vector<float>& values, float mean)
{
    float sumSquaredDiff = 0.0f;
    for (float value : values) {
        float diff = value - mean;
        sumSquaredDiff += diff * diff;
    }
    return sumSquaredDiff / (values.size() - 1);
}

float ImageDiff::calculateCovariance(const std::vector<float>& valuesA, const std::vector<float>& valuesB,
                                     float meanA, float meanB)
{
    float sum = 0.0f;
    for (size_t i = 0; i < valuesA.size(); i++) {
        sum += (valuesA[i] - meanA) * (valuesB[i] - meanB);
    }
    return sum / (valuesA.size() - 1);
}

std::vector<float> ImageDiff::convertToGrayscale(const ImageData& image)
{
    std::vector<float> grayscale(image.width * image.height);

    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            int idx = y * image.width + x;
            int pixelIdx = idx * image.channels;

            float gray;
            if (image.channels == 1) {
                gray = image.data[pixelIdx];
            } else if (image.channels >= 3) {
                // Standard RGB to grayscale conversion
                gray = 0.299f * image.data[pixelIdx] +
                       0.587f * image.data[pixelIdx + 1] +
                       0.114f * image.data[pixelIdx + 2];
            } else {
                // Fallback for 2-channel images
                gray = image.data[pixelIdx];
            }

            grayscale[idx] = gray;
        }
    }

    return grayscale;
}

std::vector<float> ImageDiff::applyGaussianFilter(const std::vector<float>& image, int width, int height)
{
    // Simple 3x3 Gaussian kernel
    const float kernel[3][3] = {
        {1.0f/16, 2.0f/16, 1.0f/16},
        {2.0f/16, 4.0f/16, 2.0f/16},
        {1.0f/16, 2.0f/16, 1.0f/16}
    };

    std::vector<float> filtered(width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int ny = std::max(0, std::min(height - 1, y + ky));
                    int nx = std::max(0, std::min(width - 1, x + kx));

                    sum += image[ny * width + nx] * kernel[ky + 1][kx + 1];
                }
            }

            filtered[y * width + x] = sum;
        }
    }

    return filtered;
}