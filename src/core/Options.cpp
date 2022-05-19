


#include "core/Options.h"

#include <iostream>

// public:

Options::Options()
    : m_widthClient(1920)
    , m_heightClient(1080)
    , m_interop(true)
    , m_light(0)
    , m_miss(2)
    , m_environment("data/NV_Default_HDR_3000x1500.hdr")
{
}

Options::~Options()
{
}

bool Options::parseCommandLine(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);

        if (arg == "?" || arg == "help" || arg == "--help")
        {
            printUsage(std::string(argv[0])); // Application name.
            return false;
        }
        else if (arg == "-w" || arg == "--width")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsage(argv[0]);
                return false;
            }
            m_widthClient = atoi(argv[++i]);
        }
        else if (arg == "-n" || arg == "--nopbo")
        {
            m_interop = false;
        }
        else if (arg == "-h" || arg == "--height")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsage(argv[0]);
                return false;
            }
            m_heightClient = atoi(argv[++i]);
        }
        else if (arg == "-l" || arg == "--light")
        {
            // Prepared for different light setups.
            //if (i == argc - 1)
            //{
            //  std::cerr << "Option '" << arg << "' requires additional argument.\n";
            //  printUsage(argv[0]);
            //  return false;
            //}
            //m_light = atoi(argv[++i]);
            m_light = 1;
        }
        else if (arg == "-m" || arg == "--miss")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsage(argv[0]);
                return false;
            }
            m_miss = atoi(argv[++i]);
        }
        else if (arg == "-e" || arg == "--env")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsage(argv[0]);
                return false;
            }
            m_environment = std::string(argv[++i]);
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsage(argv[0]);
            return false;
        }
    }
    return true;
}

int Options::getClientWidth() const
{
    return m_widthClient;
}

int Options::getClientHeight() const
{
    return m_heightClient;
}

bool Options::getInterop() const
{
    return m_interop;
}

int Options::getLight() const
{
    return m_light;
}

int Options::getMiss() const
{
    return m_miss;
}

std::string Options::getEnvironment() const
{
    return m_environment;
}


// private:

void Options::printUsage(std::string const& argv0)
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr <<
        "App Options:\n"
        "   ? | help | --help    Print this usage message and exit.\n"
        "  -w | --width <int>    Width of the client window  (512)\n"
        "  -h | --height <int>   Height of the client window (512)\n"
        "  -l | --light          Add an area light to the scene.\n"
        "  -m | --miss <0|1|2>   Select the miss shader (0 = black, 1 = white, 2 = HDR texture.\n"
        "  -e | --env <filename> Filename of a spherical HDR texture. Use with --miss 2.\n"
        "App Keystrokes:\n"
        "  SPACE  Toggles ImGui display.\n";
}
