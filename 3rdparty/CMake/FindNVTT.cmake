FIND_PATH(NVTT_INCLUDE_DIR nvtt/nvtt_lowlevel.h
  PATHS
  ${LOCAL_3RDPARTY}/nvtt/include
  PATH_SUFFIXES include
)

FIND_LIBRARY(NVTT_LIBRARIES
  NAMES nvtt nvtt30106
  PATHS
  ${LOCAL_3RDPARTY}/nvtt
  PATH_SUFFIXES lib lib/x64-v142
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVTT NVTT_INCLUDE_DIR NVTT_LIBRARIES)
mark_as_advanced(NVTT_INCLUDE_DIR NVTT_LIBRARIES)