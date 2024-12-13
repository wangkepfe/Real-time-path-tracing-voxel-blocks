cmake_minimum_required(VERSION 3.20)

project(voxelgame LANGUAGES CUDA CXX)

set(LOCAL_3RDPARTY "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")
set(CMAKE_MODULE_PATH "${LOCAL_3RDPARTY}/CMake")

set(LOCAL_ARCH "amd64")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(OS "win")
add_definitions("-DNOMINMAX")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")

find_package(OptiX81)
set(OPTIX_INCLUDE_DIR "${OPTIX81_INCLUDE_DIR}")

set(JAZZFUSION_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/renderer")
set(VOXEL_ENGINE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/voxelengine")
set(PTX_DIR "${CMAKE_CURRENT_BINARY_DIR}/ptx")

include("nvcuda_compile_ptx")

add_subdirectory(renderer)
add_subdirectory(voxelengine)

set(SOURCES main.cpp)
add_executable(voxelgame ${SOURCES})
target_include_directories( voxelgame PUBLIC ${JAZZFUSION_INCLUDE_DIR} ${VOXEL_ENGINE_INCLUDE_DIR})
target_link_libraries(voxelgame jazzfusion voxelengine)
