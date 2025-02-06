cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(AliasTableTest LANGUAGES CXX CUDA)

# Set C++ standard (optional, but common)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optionally, add the --extended-lambda flag for CUDA.
# This applies only when compiling CUDA code.
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Add the executable target containing both the implementation and the test.
add_executable(test_alias_table
    ../../shaders/AliasTable.cu
    testAliasTable.cu
)

# (Optional) Link CUDA libraries if necessary.
find_package(CUDA REQUIRED)
target_link_libraries(test_alias_table ${CUDA_LIBRARIES})
