#pragma once

#include "shaders/LinearMath.h"
#include "shaders/AliasTable.h"

class SkyModel
{
public:
    static SkyModel &Get()
    {
        static SkyModel instance;
        return instance;
    }
    SkyModel(SkyModel const &) = delete;
    ~SkyModel();
    void operator=(SkyModel const &) = delete;

    void init();
    void update();

    void initSkyConstantBuffer();
    void updateSkyState();

    Int2 getSkyRes() const { return skyRes; }
    Int2 getSunRes() const { return sunRes; }

    AliasTable *getSkyAliasTable() const { return d_skyAliasTable; }
    AliasTable *getSunAliasTable() const { return d_sunAliasTable; }

    float getAccumulatedSkyLuminance() const { return accumulatedSkyLuminance; }
    float getAccumulatedSunLuminance() const { return accumulatedSunLuminance; }

    Float3 getSunDir() const { return sunDir; }

private:
    SkyModel() {}

    float getFittingData(const float *elevMatrix, float solarElevation, int i);
    float getFittingData2(const float *elevMatrix, float solarElevation);

    float *skyPdf;
    float *sunPdf;

    AliasTable skyAliasTable;
    AliasTable sunAliasTable;

    AliasTable *d_skyAliasTable;
    AliasTable *d_sunAliasTable;

    float accumulatedSkyLuminance;
    float accumulatedSunLuminance;

    Int2 skyRes{1024, 256};
    Int2 sunRes{32, 32};

    int skySize{};
    int sunSize{};

    Float3 sunDir{};
};