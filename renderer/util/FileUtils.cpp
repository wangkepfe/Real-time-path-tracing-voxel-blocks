#include "FileUtils.h"
#include <fstream>
#include <sstream>
#include <iostream>

std::string ReadFileToString(const std::string &filename)
{
    // std::filesystem::path cwd = std::filesystem::current_path();
    //  std::cout << "The current directory is " << cwd.string() << std::endl;

    std::ifstream inputFile(filename);

    if (!inputFile)
    {
        std::cerr << "ERROR: ReadFileToString() Failed to open file " << filename << '\n';
        return std::string();
    }

    std::stringstream content;

    content << inputFile.rdbuf();

    if (inputFile.fail())
    {
        std::cerr << "ERROR: ReadFileToString() Failed to read file " << filename << '\n';
        return std::string();
    }

    return content.str();
}