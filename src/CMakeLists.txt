cmake_minimum_required(VERSION 3.20)

project( voxelgame LANGUAGES CUDA CXX )

set(JAZZFUSION_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/renderer")
set(VOXEL_ENGINE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/voxelengine")
set(PTX_DIR "${CMAKE_CURRENT_BINARY_DIR}/ptx")



add_subdirectory( renderer )
add_subdirectory( voxelengine )



set( SOURCES
  main.cpp
)

add_executable( voxelgame
  ${SOURCES}
)

target_include_directories( voxelgame PUBLIC
  ${JAZZFUSION_INCLUDE_DIR}
  ${VOXEL_ENGINE_INCLUDE_DIR}
)

target_link_libraries( voxelgame
  jazzfusion
  voxelengine
)

# copy resource
add_custom_target(copyResources)
FILE(TO_NATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}/data" DATA_DESTINATION)
add_custom_command(
    TARGET copyResources POST_BUILD
    COMMAND rd ${DATA_DESTINATION} /s /q
    COMMAND mklink /D ${DATA_DESTINATION} ${DATA_DIR})
add_dependencies(voxelgame copyResources)