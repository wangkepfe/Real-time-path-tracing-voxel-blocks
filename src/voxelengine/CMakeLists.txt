cmake_minimum_required(VERSION 3.20)
project( voxelengine LANGUAGES CUDA CXX )

set(OPTIX_INCLUDE_DIR "${OPTIX74_INCLUDE_DIR}")

set( SOURCES
  VoxelEngine.cpp
  BlockMesher.cpp
)

add_library( voxelengine
  ${SOURCES}
)

target_include_directories( voxelengine PUBLIC
  ${JAZZFUSION_INCLUDE_DIR}
  ${OPTIX_INCLUDE_DIR}
)

# target_link_libraries( voxelengine
#   jazzfusion
# )