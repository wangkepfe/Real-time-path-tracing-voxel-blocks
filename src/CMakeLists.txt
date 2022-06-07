cmake_minimum_required(VERSION 3.17)

project( jazzfusion LANGUAGES CUDA CXX )
message("\nPROJECT_NAME = " "${PROJECT_NAME}")

find_package(OpenGL REQUIRED)
find_package(GLFW REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDAToolkit 10.0 REQUIRED)
find_package(DevIL_1_8_0 REQUIRED)

set(OPTIX_INCLUDE_DIR "${OPTIX74_INCLUDE_DIR}")

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

  postprocessing/BicubicFilter.h
  postprocessing/ScalingFilter.h
  postprocessing/PostProcessor.h

  util/KernelHelper.h
  util/DebugUtils.h
  util/Picture.h
  util/Texture.h
  util/Timer.h

  shaders/LinearMath.h
  shaders/Common.h
  shaders/Camera.h
)

set( SOURCES
  core/main.cpp
  core/Backend.cpp
  core/OptixRenderer.cpp
  core/Scene.cpp
  core/UI.cpp
  core/InputHandler.cpp

  postprocessing/PostProcessor.cu

  util/Picture.cpp
  util/Texture.cpp
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
)

set(PTX_DIR "${CMAKE_CURRENT_BINARY_DIR}/ptx")

message("PTX_DIR = " "${PTX_DIR}")

NVCUDA_COMPILE_PTX( SOURCES ${SHADERS}
                    DEPENDENCIES ${SHADERS_HEADERS}
                    TARGET_PATH "${PTX_DIR}"
                    GENERATED_FILES PTX_SOURCES
                    NVCC_OPTIONS "--gpu-architecture=compute_50" "--use_fast_math" "--relocatable-device-code=true" "--generate-line-info" "-Wno-deprecated-gpu-targets" "-I${OPTIX_INCLUDE_DIR}" "-I${CMAKE_CURRENT_SOURCE_DIR}/shaders"
                  )

source_group( "imgui"           FILES ${IMGUI} )
source_group( "headers"         FILES ${HEADERS} )
source_group( "sources"         FILES ${SOURCES} )
source_group( "shaders"         FILES ${SHADERS} )
source_group( "shaders_headers" FILES ${SHADERS_HEADERS} )
source_group( "ptx"             FILES ${PTX_SOURCES})

include_directories(
  "."
  "inc"
  "ext/imgui"
  "ext/imgui/backends"
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${GLEW_INCLUDE_DIRS}
  ${GLFW_INCLUDE_DIR}
  ${OPTIX_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${IL_INCLUDE_DIR}
)

add_definitions(
  # Disable warnings for file operations fopen etc.
  "-D_CRT_SECURE_NO_WARNINGS"
)

add_executable( jazzfusion
  ${IMGUI}
  ${SOURCES}
  ${PTX_SOURCES}
)

set_property(TARGET jazzfusion PROPERTY CUDA_ARCHITECTURES 75)

target_link_libraries( jazzfusion
  ${OPENGL_gl_LIBRARY}
  ${GLEW_LIBRARIES}
  ${GLFW_LIBRARIES}
  CUDA::cudart
  CUDA::cuda_driver
  ${IL_LIBRARIES}
  ${ILU_LIBRARIES}
  ${ILUT_LIBRARIES}
)

set_target_properties( jazzfusion PROPERTIES FOLDER "src")

# copy resource
add_custom_target(copyResources)
FILE(TO_NATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}/data" DATA_DESTINATION)
add_custom_command(
    TARGET copyResources POST_BUILD
    COMMAND rd ${DATA_DESTINATION} /s /q
    COMMAND mklink /D ${DATA_DESTINATION} ${DATA_DIR})
add_dependencies(jazzfusion copyResources)

