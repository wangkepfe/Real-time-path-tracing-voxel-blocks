#pragma once

namespace jazzfusion
{

class PostProcessor
{
public:
    static PostProcessor& Get()
    {
        static PostProcessor instance;
        return instance;
    }
    PostProcessor(PostProcessor const&) = delete;
    void operator=(PostProcessor const&) = delete;

private:
    PostProcessor() {}

};


}