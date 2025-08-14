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

find_package(OptiX90)
set(OPTIX_INCLUDE_DIR "${OPTIX90_INCLUDE_DIR}")

set(JAZZFUSION_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/renderer")
set(VOXEL_ENGINE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/voxelengine")
set(PTX_DIR "${CMAKE_CURRENT_BINARY_DIR}/ptx")

include("nvcuda_compile_ptx")

find_package(OpenGL REQUIRED)
find_package(GLFW REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDAToolkit 12.6 REQUIRED)
find_package(NVTT REQUIRED)

# tinygltf library
set(TINYGLTF_INCLUDE_DIR "${LOCAL_3RDPARTY}/tinygltf")
add_definitions(-DTINYGLTF_IMPLEMENTATION)
add_definitions(-DTINYGLTF_NO_STB_IMAGE_WRITE)
add_definitions(-DTINYGLTF_NO_STB_IMAGE)

# yaml-cpp library
set(YAMLCPP_INCLUDE_DIR "${LOCAL_3RDPARTY}/yaml-cpp/include")

# Generated header directory
set(GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Auto-generate BlockType.h from blocks.yaml
set(BLOCKS_YAML_FILE "${CMAKE_CURRENT_SOURCE_DIR}/data/assets/blocks.yaml")
set(GENERATED_BLOCKTYPE_HEADER "${GENERATED_INCLUDE_DIR}/voxelengine/BlockType.h")
set(BLOCK_TYPE_GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_block_types.py")

# Add custom command to generate BlockType.h
add_custom_command(
    OUTPUT ${GENERATED_BLOCKTYPE_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GENERATED_INCLUDE_DIR}/voxelengine"
    COMMAND python ${BLOCK_TYPE_GENERATOR} ${BLOCKS_YAML_FILE} ${GENERATED_BLOCKTYPE_HEADER}
    DEPENDS ${BLOCKS_YAML_FILE} ${BLOCK_TYPE_GENERATOR}
    COMMENT "Generating BlockType.h from blocks.yaml"
    VERBATIM
)

# Create a custom target for the generated header
add_custom_target(generate_block_types DEPENDS ${GENERATED_BLOCKTYPE_HEADER})

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
    renderer/core/OfflineBackend.h
    renderer/core/OptixRenderer.h
    renderer/core/Scene.h
    renderer/core/Entity.h
    renderer/core/Character.h
    renderer/core/UI.h
    renderer/core/InputHandler.h
    renderer/core/CameraController.h
    renderer/core/FreeCameraController.h  
    renderer/core/CharacterFollowCameraController.h
    renderer/core/GlobalSettings.h
    renderer/core/BufferManager.h
    renderer/core/RenderCamera.h
    renderer/core/SceneConfig.h
    renderer/assets/AssetRegistry.h
    renderer/assets/MaterialDefinition.h
    renderer/assets/MaterialManager.h
    renderer/assets/ModelManager.h
    renderer/assets/BlockManager.h
    renderer/postprocessing/BicubicFilter.h
    renderer/postprocessing/ScalingFilter.h
    renderer/postprocessing/PostProcessor.h
    renderer/denoising/Denoiser.h
    renderer/util/KernelHelper.h
    renderer/util/DebugUtils.h
    renderer/util/Timer.h
    renderer/assets/TextureManager.h
    renderer/util/RandGenData.h
    renderer/util/RandGenHost.h
    renderer/assets/ModelUtils.h
    renderer/assets/GLTFUtils.h
    renderer/assets/ObjUtils.h
    renderer/util/ImageDiff.h
    renderer/util/FileUtils.h
    renderer/util/PerformanceTracker.h
    renderer/shaders/LinearMath.h
    renderer/shaders/Common.h
    renderer/shaders/Camera.h
    renderer/shaders/HalfPrecision.h
    renderer/shaders/Sampler.h
    renderer/shaders/SystemParameter.h
    renderer/shaders/RandGen.h
    renderer/shaders/AliasTable.h
    renderer/shaders/Bsdf.h
    renderer/shaders/Light.h
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
    renderer/core/Entity.cpp
    renderer/core/Character.cpp
    renderer/core/UI.cpp
    renderer/core/InputHandler.cpp
    renderer/core/CameraController.cpp
    renderer/core/FreeCameraController.cpp
    renderer/core/CharacterFollowCameraController.cpp
    renderer/core/BufferManager.cpp
    renderer/core/SceneConfig.cpp
    renderer/core/GlobalSettings.cpp
    renderer/assets/AssetRegistry.cpp
    renderer/assets/MaterialManager.cpp
    renderer/assets/ModelManager.cpp
    renderer/assets/BlockManager.cpp
    renderer/postprocessing/PostProcessor.cu
    renderer/denoising/Denoiser.cu
    renderer/assets/TextureManager.cu
    renderer/util/RandGenHost.cpp
    renderer/sky/Sky.cu
    renderer/util/BufferUtils.cu
    renderer/assets/ModelUtils.cpp
    renderer/assets/GLTFUtils.cpp
    renderer/assets/ObjUtils.cpp
    renderer/util/ImageDiff.cpp
    renderer/util/FileUtils.cpp
    renderer/animation/Skeleton.cpp
    renderer/animation/AnimationManager.cpp
    renderer/animation/VertexSkinning.cu
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
    renderer/shaders/Light.h
    renderer/shaders/Restir.h
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
    "renderer/ext"
    "renderer/ext/imgui"
    "renderer/ext/imgui/backends"
    "voxelengine"
    ${GENERATED_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIR}
    ${OPTIX_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NVTT_INCLUDE_DIR}
    ${TINYGLTF_INCLUDE_DIR}
    ${YAMLCPP_INCLUDE_DIR}
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

# Make voxelgame depend on generated header
add_dependencies(voxelgame generate_block_types)

# target_compile_options(voxelgame PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -rdc=true>)
# set_property(TARGET voxelgame PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Function to copy data files, PTX files, and DLLs to target directory
function(copy_game_data TARGET_NAME COPY_GLFW_GLEW)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        # Create directories only if they don't exist
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/data/models
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/data/textures
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/data/scene
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/data/settings
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/data/assets
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/ptx
        # Copy directories only if files are different
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${CMAKE_CURRENT_SOURCE_DIR}/data/models $<TARGET_FILE_DIR:${TARGET_NAME}>/data/models
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${CMAKE_CURRENT_SOURCE_DIR}/data/textures $<TARGET_FILE_DIR:${TARGET_NAME}>/data/textures
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${CMAKE_CURRENT_SOURCE_DIR}/data/scene $<TARGET_FILE_DIR:${TARGET_NAME}>/data/scene
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${CMAKE_CURRENT_SOURCE_DIR}/data/settings $<TARGET_FILE_DIR:${TARGET_NAME}>/data/settings
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${CMAKE_CURRENT_SOURCE_DIR}/data/assets $<TARGET_FILE_DIR:${TARGET_NAME}>/data/assets
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${PTX_DIR} $<TARGET_FILE_DIR:${TARGET_NAME}>/ptx
        # Copy NVTT DLL (needed by both targets)
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/nvtt/nvtt30200.dll $<TARGET_FILE_DIR:${TARGET_NAME}>/nvtt30200.dll
    )

    # Copy GLFW and GLEW DLLs only if requested (for main target only)
    if(${COPY_GLFW_GLEW})
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/glfw/lib-vc2022/glfw3.dll $<TARGET_FILE_DIR:${TARGET_NAME}>/glfw3.dll
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/glew/bin/Release/x64/glew32.dll $<TARGET_FILE_DIR:${TARGET_NAME}>/glew32.dll
        )
    endif()
endfunction()

# Call the shared function for main target (with GLFW/GLEW)
copy_game_data(voxelgame TRUE)

# Offline renderer sources (excluding OpenGL/GLFW dependencies)
set( OFFLINE_SOURCES
    mainOffline.cpp
    renderer/core/OfflineBackend.cpp
    renderer/core/SceneConfig.cpp
    renderer/core/OptixRenderer.cpp
    renderer/core/Scene.cpp
    renderer/core/Entity.cpp
    renderer/core/Character.cpp
    renderer/core/CameraController.cpp
    renderer/core/FreeCameraController.cpp
    renderer/core/CharacterFollowCameraController.cpp
    renderer/core/BufferManager.cpp
    renderer/core/GlobalSettings.cpp
    renderer/assets/AssetRegistry.cpp
    renderer/assets/MaterialManager.cpp
    renderer/assets/ModelManager.cpp
    renderer/assets/BlockManager.cpp
    renderer/postprocessing/PostProcessor.cu
    renderer/denoising/Denoiser.cu
    renderer/assets/TextureManager.cu
    renderer/util/RandGenHost.cpp
    renderer/sky/Sky.cu
    renderer/util/BufferUtils.cu
    renderer/assets/ModelUtils.cpp
    renderer/assets/GLTFUtils.cpp
    renderer/assets/ObjUtils.cpp
    renderer/util/ImageDiff.cpp
    renderer/util/FileUtils.cpp
    renderer/animation/Skeleton.cpp
    renderer/animation/AnimationManager.cpp
    renderer/animation/VertexSkinning.cu
    voxelengine/VoxelEngine.cu
    voxelengine/VoxelSceneGen.cu
    voxelengine/Noise.cpp
    renderer/shaders/AliasTable.cu
)

# Create offline renderer executable
add_executable(voxelgame_offline
    ${OFFLINE_SOURCES}
    ${HEADERS}
    ${SHADERS_HEADERS}
    ${PTX_SOURCES}
)

target_include_directories( voxelgame_offline PUBLIC
    "."
    "renderer"
    "voxelengine"
    ${GENERATED_INCLUDE_DIR}
    ${OPTIX_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NVTT_INCLUDE_DIR}
    ${TINYGLTF_INCLUDE_DIR}
    ${YAMLCPP_INCLUDE_DIR}
)

target_compile_definitions( voxelgame_offline PRIVATE
    OFFLINE_MODE
)

target_link_libraries( voxelgame_offline
    ${NVTT_LIBRARIES}
    CUDA::cudart
    CUDA::cuda_driver
    cudadevrt.lib
)

set_property(TARGET voxelgame_offline PROPERTY CUDA_ARCHITECTURES 89)

# Make voxelgame_offline depend on generated header
add_dependencies(voxelgame_offline generate_block_types)

# Call the shared function for offline target (without GLFW/GLEW)
copy_game_data(voxelgame_offline FALSE)

# Separate include for stb_image_write.h since it's in renderer/ext/stb/
target_include_directories( voxelgame_offline PRIVATE
    "renderer/ext"
)