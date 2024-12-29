#include "core/BufferManager.h"
#include "core/GlobalSettings.h"
#include "shaders/Sampler.h"
#include "sky/Sky.h"
#include "sky/SkyData.h"
#include "util/ColorSpace.h"
#include "util/DebugUtils.h"
#include "util/KernelHelper.h"
#include "util/Scan.h"

namespace jazzfusion
{

    __constant__ float cSkyConfigs[90];
    __constant__ float cSkyRadiances[10];
    __constant__ float cSolarDatasets[1800];
    __constant__ float cLimbDarkeningDatasets[60];

    //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
    float SkyModel::getFittingData(const float *elevMatrix, float solarElevation,
                                   int i)
    {
        return (powf(1.0f - solarElevation, 5.0f) * elevMatrix[i] +
                5.0f * powf(1.0f - solarElevation, 4.0f) * solarElevation *
                    elevMatrix[i + 9] +
                10.0f * powf(1.0f - solarElevation, 3.0f) *
                    powf(solarElevation, 2.0f) * elevMatrix[i + 18] +
                10.0f * powf(1.0f - solarElevation, 2.0f) *
                    powf(solarElevation, 3.0f) * elevMatrix[i + 27] +
                5.0f * (1.0f - solarElevation) * powf(solarElevation, 4.0f) *
                    elevMatrix[i + 36] +
                powf(solarElevation, 5.0f) * elevMatrix[i + 45]);
    }

    //(1-t).^3* A1 + 3*(1-t).^2.*t * A2 + 3*(1-t) .* t .^ 2 * A3 + t.^3 * A4;
    float SkyModel::getFittingData2(const float *elevMatrix, float solarElevation)
    {
        return (powf(1.0f - solarElevation, 5.0f) * elevMatrix[0] +
                5.0f * powf(1.0f - solarElevation, 4.0f) * solarElevation *
                    elevMatrix[1] +
                10.0f * powf(1.0f - solarElevation, 3.0f) *
                    powf(solarElevation, 2.0f) * elevMatrix[2] +
                10.0f * powf(1.0f - solarElevation, 2.0f) *
                    powf(solarElevation, 3.0f) * elevMatrix[3] +
                5.0f * (1.0f - solarElevation) * powf(solarElevation, 4.0f) *
                    elevMatrix[4] +
                powf(solarElevation, 5.0f) * elevMatrix[5]);
    }

    void SkyModel::initSkyConstantBuffer()
    {
        CUDA_CHECK(
            cudaMemcpyToSymbol(cSolarDatasets, hSolarDatasets, sizeof(float) * 1800));
        CUDA_CHECK(cudaMemcpyToSymbol(cLimbDarkeningDatasets, hLimbDarkeningDatasets,
                                      sizeof(float) * 60));
    }

    void SkyModel::updateSkyState()
    {
        float hSkyConfigs[90];
        float hSkyRadiances[10];

        float elevation = (M_PI / 2.0f) - acos(sunDir.y);

        float solarElevation = powf(elevation / (M_PI / 2.0f), (1.0f / 3.0f));

        unsigned int channel;
        for (channel = 0; channel < 10; ++channel)
        {
            for (int i = 0; i < 9; ++i)
            {
                hSkyConfigs[channel * 9 + i] =
                    getFittingData(skyDataSets + channel * 54, solarElevation, i);
            }

            hSkyRadiances[channel] =
                getFittingData2(skyDataSetsRad + channel * 6, solarElevation);
        }

        CUDA_CHECK(
            cudaMemcpyToSymbol(cSkyConfigs, hSkyConfigs, sizeof(float) * 10 * 9));
        CUDA_CHECK(
            cudaMemcpyToSymbol(cSkyRadiances, hSkyRadiances, sizeof(float) * 10));
    }

    INL_DEVICE Float3 SpectrumToXyz(int channel)
    {
        static const float spectrumCieX[] = {
            2.372527e-02f,
            1.955480e+00f,
            1.074553e+01f,
            5.056697e+00f,
            4.698190e+00f,
            2.391135e+01f,
            3.798705e+01f,
            1.929414e+01f,
            2.970610e+00f,
            2.092986e-01f,
        };

        static const float spectrumCieY[] = {
            6.813859e-04f,
            6.771017e-02f,
            1.171193e+00f,
            6.997765e+00f,
            2.666710e+01f,
            3.758372e+01f,
            2.503930e+01f,
            8.150395e+00f,
            1.098635e+00f,
            7.563256e-02f,
        };

        static const float spectrumCieZ[] = {
            1.119121e-01f,
            9.441195e+00f,
            5.597921e+01f,
            3.589996e+01f,
            5.070894e+00f,
            3.523189e-01f,
            3.422707e-02f,
            2.539118e-03f,
            7.836666e-06f,
            0.000000e+00f,
        };

        static constexpr float CIE_Y_integral = 106.856895;

        return Float3(spectrumCieX[channel], spectrumCieY[channel],
                      spectrumCieZ[channel]) /
               CIE_Y_integral;
    }

    INL_DEVICE Float3 GetSkyRadiance(const Float3 &raydir, const Float3 &sunDir,
                                     SkyParams &skyParams)
    {
        float theta = acos(raydir.y);
        float gamma = acos(clampf(dot(raydir, sunDir), -1, 1));

        unsigned int channel;

        float spectrum[10];
        Float3 xyzColor = Float3(0);

#pragma unroll
        for (channel = 0; channel < 10; ++channel)
        {
            float *configuration = cSkyConfigs + channel * 9;

            const float expM = exp(configuration[4] * gamma);
            const float rayM = cos(gamma) * cos(gamma);
            const float mieM = (1.0f + cos(gamma) * cos(gamma)) /
                               powf((1.0f + configuration[8] * configuration[8] -
                                     2.0f * configuration[8] * cos(gamma)),
                                    1.5);
            const float zenith = sqrt(cos(theta));

            float radianceInternal =
                (1.0f +
                 configuration[0] * exp(configuration[1] / (cos(theta) + 0.01))) *
                (configuration[2] + configuration[3] * expM + configuration[5] * rayM +
                 configuration[6] * mieM + configuration[7] * zenith);

            float radiance = radianceInternal * cSkyRadiances[channel];

            spectrum[channel] = radiance;

            xyzColor += radiance * SpectrumToXyz(channel);
        }

        Float3 rgbColor = XyzToRgbSrgb(xyzColor);

        return rgbColor;
    }

    INL_DEVICE Float3 GetSunRadiance(const Float3 &raydir, const Float3 &sunDir,
                                     SkyParams &skyParams)
    {
        float theta = acos(raydir.y);
        float gamma = acos(clampf(dot(raydir, sunDir), -1, 1));

        unsigned int channel;

        float elevation = (M_PI / 2.0f) - acos(sunDir.y);

        const float sunAngle = 0.51f;
        const float solarRadius = sunAngle * M_PI / 180.0f / 2.0f;
        const float sunBrightnessScaleFactor =
            1.0f / ((sunAngle / 0.51f) * (sunAngle / 0.51f));

        Float3 xyzColor = Float3(0);

        float sol_rad_sin = sinf(solarRadius);
        float ar2 = 1.0f / (sol_rad_sin * sol_rad_sin);
        float singamma = sinf(gamma);

        float sc2 = 1.0f - ar2 * singamma * singamma;

        if (sc2 < 0.0f)
            sc2 = 0.0f;

        float sampleCosine = sqrtf(sc2);

        if (sampleCosine == 0.0f)
            return Float3(0.0f);

#pragma unroll
        for (channel = 0; channel < 10; ++channel)
        {
            const int pieces = 45;
            const int order = 4;

            int pos = (int)(powf(2.0 * elevation / M_PI, 1.0 / 3.0) * pieces);

            if (pos > 44)
                pos = 44;

            const float break_x =
                powf(((float)pos / (float)pieces), 3.0) * (M_PI * 0.5);

            const float *coefs =
                cSolarDatasets + channel * 180 + (order * (pos + 1) - 1);

            float res = 0.0;
            const float x = elevation - break_x;
            float x_exp = 1.0;

            int i;
#pragma unroll
            for (i = 0; i < order; ++i)
            {
                res += x_exp * *coefs--;
                x_exp *= x;
            }

            float directRadiance = res;

            float ldCoefficient[6];

#pragma unroll
            for (i = 0; i < 6; i++)
                ldCoefficient[i] = cLimbDarkeningDatasets[channel * 6 + i];

            float darkeningFactor = ldCoefficient[0] + ldCoefficient[1] * sampleCosine +
                                    ldCoefficient[2] * powf(sampleCosine, 2.0f) +
                                    ldCoefficient[3] * powf(sampleCosine, 3.0f) +
                                    ldCoefficient[4] * powf(sampleCosine, 4.0f) +
                                    ldCoefficient[5] * powf(sampleCosine, 5.0f);

            directRadiance *= darkeningFactor * sunBrightnessScaleFactor;

            xyzColor += directRadiance * SpectrumToXyz(channel);
        }

        Float3 rgbColor = XyzToRgbSrgb(xyzColor);
        rgbColor = Float3(1.0f);

        return rgbColor;
    }

    __global__ void Sky(SurfObj skyBuffer, float *skyPdf, Int2 size, Float3 sunDir,
                        SkyParams skyParams)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        float u = ((float)x + 0.5f) / size.x;
        float v = ((float)y + 0.5f) / size.y;

        Float3 rayDir = EqualAreaMap(u, v);

        const float skyScalar = 1.0f;
        Float3 color = GetSkyRadiance(rayDir, sunDir, skyParams) * skyScalar;

        color = max3f(color, Float3(0.0f));

        Store2DFloat4(Float4(color, 0), skyBuffer, Int2(x, y));

        // sky cdf
        int i = size.x * y + x;
        skyPdf[i] = dot(color, Float3(0.3f, 0.6f, 0.1f));
    }

    __global__ void SkySun(SurfObj sunBuffer, float *sunPdf, Int2 size,
                           Float3 sunDir, SkyParams skyParams)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        float u = ((float)x + 0.5f) / size.x;
        float v = ((float)y + 0.5f) / size.y;

        const float sunAngle = 0.51f;
        Float3 raydir =
            EqualAreaMapCone(sunDir, u, v, cos(sunAngle * M_PI / 180.0f / 2.0f));

        const float sunScalar = 1.0f;
        Float3 color = GetSunRadiance(raydir, sunDir, skyParams) * sunScalar;

        color = max3f(color, Float3(0.0f));

        Store2DFloat4(Float4(color, 0), sunBuffer, Int2(x, y));

        // sky cdf
        int i = size.x * y + x;
        sunPdf[i] = dot(color, Float3(0.3f, 0.6f, 0.1f));
    }

    void SkyModel::init()
    {
        skySize = skyRes.x * skyRes.y;
        sunSize = sunRes.x * sunRes.y;

        int skyScanBlockCount = skySize / skyScanBlockSize;
        int sunScanBlockCount = sunSize / sunScanBlockSize;

        CUDA_CHECK(cudaMalloc((void **)&skyCdf, skySize * sizeof(float)));
        CUDA_CHECK(cudaMemset(skyCdf, 0, skySize * sizeof(float)));

        CUDA_CHECK(cudaMalloc((void **)&skyPdf, skySize * sizeof(float)));
        CUDA_CHECK(cudaMemset(skyPdf, 0, skySize * sizeof(float)));

        CUDA_CHECK(
            cudaMalloc((void **)&skyCdfScanTmp, skyScanBlockCount * sizeof(float)));
        CUDA_CHECK(cudaMemset(skyCdfScanTmp, 0, skyScanBlockCount * sizeof(float)));

        CUDA_CHECK(cudaMalloc((void **)&sunCdf, sunSize * sizeof(float)));
        CUDA_CHECK(cudaMemset(sunCdf, 0, sunSize * sizeof(float)));

        CUDA_CHECK(cudaMalloc((void **)&sunPdf, sunSize * sizeof(float)));
        CUDA_CHECK(cudaMemset(sunPdf, 0, sunSize * sizeof(float)));

        CUDA_CHECK(
            cudaMalloc((void **)&sunCdfScanTmp, sunScanBlockCount * sizeof(float)));
        CUDA_CHECK(cudaMemset(sunCdfScanTmp, 0, sunScanBlockCount * sizeof(float)));

        initSkyConstantBuffer();
    }

    SkyModel::~SkyModel()
    {
        cudaFree(skyCdf);
        cudaFree(skyPdf);
        cudaFree(skyCdfScanTmp);
        cudaFree(sunCdf);
        cudaFree(sunPdf);
        cudaFree(sunCdfScanTmp);
    }

    void SkyModel::update()
    {
        auto &skyParams = GlobalSettings::GetSkyParams();
        auto &bufferManager = BufferManager::Get();
        if (skyParams.needRegenerate)
        {
            skyParams.needRegenerate = false;

            // Compute sun direction based on the time of day and the sun rotation axis
            const Float3 axis =
                normalize(Float3(0.0f, cos(skyParams.sunAxisAngle * Pi_over_180),
                                 sin(skyParams.sunAxisAngle * Pi_over_180)));
            const float angle = fmodf(skyParams.timeOfDay * M_PI, TWO_PI);
            sunDir = rotate3f(axis, angle, cross(Float3(0, 1, 0), axis)).normalized();

            // TODO: Compute sky resolution based on the camera resolution and FOV to
            // reduce aliasing const auto& renderer = OptixRenderer::Get(); const auto&
            // camera = renderer.getCamera(); Float2 cameraRes = camera.resolution;
            // Float2 fov = camera.fov;

            // Compute sky state on CPU
            updateSkyState();

            // Generate sky on GPU
            Sky KERNEL_ARGS2(GetGridDim(skyRes.x, skyRes.y, BLOCK_DIM_8x8x1),
                             GetBlockDim(BLOCK_DIM_8x8x1))(
                bufferManager.GetBuffer2D(SkyBuffer), skyPdf, skyRes, sunDir,
                skyParams);

            // Scan for CDF
            Scan(skyPdf, skyCdf, skyCdfScanTmp, skySize, skyScanBlockSize, 1);

            // Generate sun on GPU
            SkySun KERNEL_ARGS2(GetGridDim(sunRes.x, sunRes.y, BLOCK_DIM_8x8x1),
                                GetBlockDim(BLOCK_DIM_8x8x1))(
                bufferManager.GetBuffer2D(SunBuffer), sunPdf, sunRes, sunDir,
                skyParams);

            // Scan for CDF
            Scan(sunPdf, sunCdf, sunCdfScanTmp, sunSize, sunScanBlockSize, 1);

            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaPeekAtLastError());
        }
    }

} // namespace jazzfusion