#pragma once

#include <iostream>

namespace jazzfusion
{

class GlobalSettings
{
public:
    static GlobalSettings& Get()
    {
        static GlobalSettings instance;
        return instance;
    }
    GlobalSettings(GlobalSettings const&) = delete;
    void operator=(GlobalSettings const&) = delete;

    static const std::string& GetCameraSaveFileName() { return Get().cameraSaveFileName; }

private:
    GlobalSettings() {}

    std::string cameraSaveFileName = "mycamera.bin";
};

}