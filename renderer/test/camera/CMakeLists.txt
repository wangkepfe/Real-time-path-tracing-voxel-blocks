cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CameraTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(test_camera test.cpp)
target_link_libraries(test_camera ${CUDA_LIBRARIES})
