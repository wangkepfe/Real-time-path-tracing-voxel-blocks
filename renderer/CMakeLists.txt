cmake_minimum_required(VERSION 3.17)

project( jazzfusion LANGUAGES CUDA CXX )
message("\nPROJECT_NAME = " "${PROJECT_NAME}")

find_package(OpenGL REQUIRED)
find_package(GLFW REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDAToolkit 12.6 REQUIRED)
find_package(NVTT REQUIRED)

set( IMGUI
    ext/imgui/imconfig.h
    ext/imgui/imgui.h
    ext/imgui/imgui.cpp
    ext/imgui/imgui_internal.h
    ext/imgui/imgui_widgets.cpp
    ext/imgui/imgui_tables.cpp
    ext/imgui/imgui_draw.cpp
    ext/imgui/backends/imgui_impl_glfw.cpp
    ext/imgui/backends/imgui_impl_glfw.h
    ext/imgui/backends/imgui_impl_opengl3_loader.h
    ext/imgui/backends/imgui_impl_opengl3.cpp
    ext/imgui/backends/imgui_impl_opengl3.h)

set( HEADERS
  core/Backend.h
  core/OptixRenderer.h
  core/Scene.h
  core/UI.h
  core/InputHandler.h
  core/GlobalSettings.h
  core/BufferManager.h
  core/RenderCamera.h

  postprocessing/BicubicFilter.h
  postprocessing/ScalingFilter.h
  postprocessing/PostProcessor.h

  denoising/Denoiser.h
  denoising/DenoisingUtils.h
  denoising/NoiseVisualization.h
  denoising/SpatialFilter.h
  denoising/SpatialWideFilter.h
  denoising/TemporalFilter.h
  denoising/BilateralFilter.h

  util/KernelHelper.h
  util/DebugUtils.h
  util/Timer.h
  util/TextureUtils.h
  util/RandGenData.h
  util/RandGenHost.h

  shaders/LinearMath.h
  shaders/Common.h
  shaders/Camera.h
  shaders/HalfPrecision.h
  shaders/Sampler.h
  shaders/SystemParameter.h
  shaders/RandGen.h

  sky/Sky.h
  sky/SkyData.h
)

set( SOURCES
  core/Backend.cpp
  core/OptixRenderer.cpp
  core/Scene.cpp
  core/UI.cpp
  core/InputHandler.cpp
  core/BufferManager.cpp

  postprocessing/PostProcessor.cu

  denoising/Denoiser.cu

  util/TextureUtils.cpp
  util/RandGenHost.cpp

  sky/Sky.cu
)

set( SHADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/ClosestHit.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/Miss.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/RayGen.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/LightSample.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/Bsdf.cu
)

set( SHADERS_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/OptixShaderCommon.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/SystemParameter.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/LinearMath.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/ShaderDebugUtils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/Common.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/Camera.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/HalfPrecision.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/Sampler.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/RandGen.h
)

message("PTX_DIR = " "${PTX_DIR}")

NVCUDA_COMPILE_PTX( SOURCES ${SHADERS}
                    DEPENDENCIES ${SHADERS_HEADERS}
                    TARGET_PATH "${PTX_DIR}"
                    GENERATED_FILES PTX_SOURCES
                    NVCC_OPTIONS "-arch=sm_89" "--use_fast_math" "--relocatable-device-code=true" "--generate-line-info" "-Wno-deprecated-gpu-targets" "-I${OPTIX_INCLUDE_DIR}" "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders"
                  )

source_group( "imgui"           FILES ${IMGUI} )
source_group( "headers"         FILES ${HEADERS} )
source_group( "sources"         FILES ${SOURCES} )
source_group( "shaders"         FILES ${SHADERS} )
source_group( "shaders_headers" FILES ${SHADERS_HEADERS} )
source_group( "ptx"             FILES ${PTX_SOURCES})

add_library( jazzfusion
  ${IMGUI}
  ${SOURCES}
  ${HEADERS}
  ${PTX_SOURCES}
)

target_include_directories( jazzfusion PUBLIC
  "."
  "ext/imgui"
  "ext/imgui/backends"
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${GLEW_INCLUDE_DIRS}
  ${GLFW_INCLUDE_DIR}
  ${OPTIX_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${NVTT_INCLUDE_DIR}
)

set_property(TARGET jazzfusion PROPERTY CUDA_ARCHITECTURES 89)

target_link_libraries( jazzfusion
  ${OPENGL_gl_LIBRARY}
  ${GLEW_LIBRARIES}
  ${GLFW_LIBRARIES}
  CUDA::cudart
  CUDA::cuda_driver
  ${NVTT_LIBRARIES}
)

set_target_properties( jazzfusion PROPERTIES FOLDER "renderer")

