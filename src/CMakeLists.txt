cmake_minimum_required(VERSION 3.17)

project( jazzfusion )
message("\nPROJECT_NAME = " "${PROJECT_NAME}")

find_package(OpenGL REQUIRED)
find_package(GLFW REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDAToolkit 10.0 REQUIRED)
find_package(DevIL_1_8_0 REQUIRED)

set(OPTIX_INCLUDE_DIR "${OPTIX74_INCLUDE_DIR}")

set( IMGUI
  imgui/imconfig.h
  imgui/imgui.h
  imgui/imgui_impl_glfw_gl3.h
  imgui/imgui_internal.h
  imgui/stb_rect_pack.h
  imgui/stb_textedit.h
  imgui/stb_truetype.h
  imgui/imgui.cpp
  imgui/imgui_demo.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_impl_glfw_gl3.cpp
)

set( HEADERS
  util/DebugUtils.h
  util/Picture.h
  core/Camera.h
  util/Texture.h
  util/Timer.h
  core/Backend.h
  core/OptixRenderer.h
  core/Scene.h
  core/UI.h
)

set( SOURCES
  core/main.cpp
  util/Picture.cpp
  core/Camera.cpp
  util/Texture.cpp
  core/Backend.cpp
  core/OptixRenderer.cpp
  core/Scene.cpp
  core/UI.cpp
)

# Prefix the shaders with the full path name to allow stepping through errors with F8.
set( SHADERS
  # Core shaders.
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/ClosestHit.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/Miss.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/RayGen.cu

  # Direct callables
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/LightSample.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/Bsdf.cu
)

set( SHADERS_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/ShaderCommon.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/RngUtils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/SystemParameter.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/ShaderCommon.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/MathUtils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/ShaderDebugUtils.h
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
  "imgui"
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
  ${HEADERS}
  ${SOURCES}
  ${SHADERS_HEADERS}
  ${SHADERS}
  ${PTX_SOURCES}
)

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

