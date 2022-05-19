#pragma once

#include <cuda_runtime.h>

namespace jazzfusion {

class UI
{
public:
    static UI& Get()
    {
        static UI instance;
        return instance;
    }
    UI(UI const&) = delete;
    void operator=(UI const&) = delete;

private:
    UI() {}
};

}