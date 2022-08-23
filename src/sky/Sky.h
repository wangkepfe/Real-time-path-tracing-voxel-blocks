#pragma once

#include "shaders/LinearMath.h"

namespace jazzfusion
{

class SkyModel
{
public:
    static SkyModel& Get()
    {
        static SkyModel instance;
        return instance;
    }
    SkyModel(SkyModel const&) = delete;
    ~SkyModel();
    void operator=(SkyModel const&) = delete;


    void init();
    void update();

    void initSkyConstantBuffer();
    void updateSkyState();

    Int2 getSkyRes() const { return skyRes; }
    Int2 getSunRes() const { return sunRes; }

    float* getSkyCdf() const { return skyCdf; }
    float* getSunCdf() const { return sunCdf; }

    Float3 getSunDir() const { return sunDir; }

private:
    SkyModel() {}

    float getFittingData(const float* elevMatrix, float solarElevation, int i);
    float getFittingData2(const float* elevMatrix, float solarElevation);

    float* skyPdf;
    float* skyCdf;
    float* skyCdfScanTmp;

    float* sunPdf;
    float* sunCdf;
    float* sunCdfScanTmp;

    Int2 skyRes{ 512, 256 };
    Int2 sunRes{ 32, 32 };

    int skySize{};
    int sunSize{};

    int skyScanBlockSize{ 256 };
    int sunScanBlockSize{ 32 };

    Float3 sunDir{};
};

}