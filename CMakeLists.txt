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
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

find_package(OptiX81)
set(OPTIX_INCLUDE_DIR "${OPTIX81_INCLUDE_DIR}")

set(JAZZFUSION_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/renderer")
set(VOXEL_ENGINE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/voxelengine")
set(PTX_DIR "${CMAKE_CURRENT_BINARY_DIR}/ptx")

include("nvcuda_compile_ptx")

find_package(OpenGL REQUIRED)
find_package(GLFW REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDAToolkit 12.6 REQUIRED)
find_package(NVTT REQUIRED)

set( IMGUI
    renderer/ext/imgui/imconfig.h
    renderer/ext/imgui/imgui.h
    renderer/ext/imgui/imgui.cpp
    renderer/ext/imgui/imgui_internal.h
    renderer/ext/imgui/imgui_widgets.cpp
    renderer/ext/imgui/imgui_tables.cpp
    renderer/ext/imgui/imgui_draw.cpp
    renderer/ext/imgui/backends/imgui_impl_glfw.cpp
    renderer/ext/imgui/backends/imgui_impl_glfw.h
    renderer/ext/imgui/backends/imgui_impl_opengl3_loader.h
    renderer/ext/imgui/backends/imgui_impl_opengl3.cpp
    renderer/ext/imgui/backends/imgui_impl_opengl3.h)

set( HEADERS
    renderer/core/Backend.h
    renderer/core/OptixRenderer.h
    renderer/core/Scene.h
    renderer/core/UI.h
    renderer/core/InputHandler.h
    renderer/core/GlobalSettings.h
    renderer/core/BufferManager.h
    renderer/core/RenderCamera.h
    renderer/postprocessing/BicubicFilter.h
    renderer/postprocessing/ScalingFilter.h
    renderer/postprocessing/PostProcessor.h
    renderer/denoising/Denoiser.h
    renderer/util/KernelHelper.h
    renderer/util/DebugUtils.h
    renderer/util/Timer.h
    renderer/util/TextureUtils.h
    renderer/util/RandGenData.h
    renderer/util/RandGenHost.h
    renderer/util/ModelUtils.h
    renderer/shaders/LinearMath.h
    renderer/shaders/Common.h
    renderer/shaders/Camera.h
    renderer/shaders/HalfPrecision.h
    renderer/shaders/Sampler.h
    renderer/shaders/SystemParameter.h
    renderer/shaders/RandGen.h
    renderer/shaders/AliasTable.h
    renderer/shaders/Bsdf.h
    renderer/sky/Sky.h
    renderer/sky/SkyData.h
    voxelengine/VoxelEngine.h
    voxelengine/VoxelSceneGen.h
    voxelengine/Voxel.h
    voxelengine/VoxelChunk.h
    voxelengine/VoxelMath.h
    voxelengine/Noise.h
    voxelengine/ext/PerlinNoise.hpp
)

set( SOURCES
    main.cpp
    renderer/core/Backend.cpp
    renderer/core/OptixRenderer.cpp
    renderer/core/Scene.cpp
    renderer/core/UI.cpp
    renderer/core/InputHandler.cpp
    renderer/core/BufferManager.cpp
    renderer/postprocessing/PostProcessor.cu
    renderer/denoising/Denoiser.cu
    renderer/util/TextureUtils.cu
    renderer/util/RandGenHost.cpp
    renderer/sky/Sky.cu
    renderer/util/BufferUtils.cu
    renderer/util/ModelUtils.cpp
    voxelengine/VoxelEngine.cu
    voxelengine/VoxelSceneGen.cu
    voxelengine/Noise.cpp
    renderer/shaders/AliasTable.cu
)

set( OPTIX_SHADERS
    renderer/shaders/ClosestHit.cu
    renderer/shaders/Miss.cu
    renderer/shaders/RayGen.cu
    renderer/shaders/Bsdf.cu
    renderer/shaders/AliasTable.cu
)

set( SHADERS_HEADERS
    renderer/shaders/OptixShaderCommon.h
    renderer/shaders/SystemParameter.h
    renderer/shaders/LinearMath.h
    renderer/shaders/ShaderDebugUtils.h
    renderer/shaders/Common.h
    renderer/shaders/Camera.h
    renderer/shaders/HalfPrecision.h
    renderer/shaders/Sampler.h
    renderer/shaders/RandGen.h
    renderer/shaders/SelfHit.h
    renderer/shaders/AliasTable.h
    renderer/shaders/Bsdf.h
)

message("PTX_DIR = " "${PTX_DIR}")

NVCUDA_COMPILE_PTX( SOURCES ${OPTIX_SHADERS}
                    DEPENDENCIES ${SHADERS_HEADERS}
                    TARGET_PATH "${PTX_DIR}"
                    GENERATED_FILES PTX_SOURCES
                    NVCC_OPTIONS "-arch=sm_89" "--use_fast_math" "--relocatable-device-code=true" "--generate-line-info" "--extended-lambda" "-Wno-deprecated-gpu-targets" "-I${OPTIX_INCLUDE_DIR}" "-I${CMAKE_CURRENT_SOURCE_DIR}/renderer/shaders"
                  )

source_group("imgui"           FILES ${IMGUI} )
source_group("headers"         FILES ${HEADERS} )
source_group("sources"         FILES ${SOURCES} )
source_group("optix_shaders"   FILES ${OPTIX_SHADERS} )
source_group("shaders_headers" FILES ${SHADERS_HEADERS} )
source_group("ptx"             FILES ${PTX_SOURCES})

add_executable(voxelgame
    ${IMGUI}
    ${SOURCES}
    ${HEADERS}
    ${SHADERS_HEADERS}
    ${PTX_SOURCES}
)

target_include_directories( voxelgame PUBLIC
    "."
    "renderer"
    "renderer/ext/imgui"
    "renderer/ext/imgui/backends"
    "voxelengine"
    ${GLEW_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIR}
    ${OPTIX_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NVTT_INCLUDE_DIR}
)

target_link_libraries( voxelgame
    ${OPENGL_gl_LIBRARY}
    ${GLEW_LIBRARIES}
    ${GLFW_LIBRARIES}
    ${NVTT_LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver
    cudadevrt.lib
)

set_property(TARGET voxelgame PROPERTY CUDA_ARCHITECTURES 89)

# target_compile_options(voxelgame PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -rdc=true>)
# set_property(TARGET voxelgame PROPERTY CUDA_SEPARABLE_COMPILATION ON)