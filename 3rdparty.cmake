cmake_minimum_required(VERSION 3.17)

set(BUILD_ARCH x64)
set(GENERATOR "Visual Studio 17 2022")
set(MSVC_TOOLSET "msvc-14.3")

include("nvcuda_compile_ptx")

message("Creating 3rdparty library folder for ${GENERATOR} ${BUILD_ARCH}")

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty" CACHE PATH "default install path" FORCE)

set(DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/downloads")
set(PATCH_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/patches")

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/temp/sources")
set(BUILD_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/temp/build/${MSVC_TOOLSET}")

message("Install prefix: ${CMAKE_INSTALL_PREFIX} ${ARGC} ${ARGV}")

file(MAKE_DIRECTORY ${SOURCE_DIR})
file(MAKE_DIRECTORY ${BUILD_DIR})

macro(glew_sourceforge)
    message("GLEW")
    set(FILENAME "glew-2.1.0-win32.zip")
    if (NOT EXISTS "${DOWNLOAD_DIR}/${FILENAME}")
        message("  downloading")
        file(DOWNLOAD "https://sourceforge.net/projects/glew/files/glew/2.1.0/${FILENAME}" "${DOWNLOAD_DIR}/${FILENAME}" STATUS downloaded)
    endif()
    if (EXISTS "${CMAKE_INSTALL_PREFIX}/glew")
      message("  removing ${CMAKE_INSTALL_PREFIX}/glew")
      execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_INSTALL_PREFIX}/glew")
    endif()
    message("  extracting")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${DOWNLOAD_DIR}/${FILENAME}" WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}")
    message("  renaming")
    file(RENAME "${CMAKE_INSTALL_PREFIX}/glew-2.1.0" "${CMAKE_INSTALL_PREFIX}/glew")
endmacro()

macro(glfw_sourceforge)
    message("GLFW")
    set(FILENAME "glfw-3.3.7.bin.WIN64.zip")
    if (NOT EXISTS "${DOWNLOAD_DIR}/${FILENAME}")
        message("  downloading")
        file(DOWNLOAD "https://github.com/glfw/glfw/releases/download/3.3.7/glfw-3.3.7.bin.WIN64.zip" "${DOWNLOAD_DIR}/${FILENAME}" STATUS downloaded)
    endif()
    if (EXISTS "${CMAKE_INSTALL_PREFIX}/glfw")
      message("  removing ${CMAKE_INSTALL_PREFIX}/glfw")
      execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_INSTALL_PREFIX}/glfw")
    endif()
    message("  extracting")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${DOWNLOAD_DIR}/${FILENAME}" WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}")
    message("  renaming")
    file(RENAME "${CMAKE_INSTALL_PREFIX}/glfw-3.3.7.bin.WIN64" "${CMAKE_INSTALL_PREFIX}/glfw")
endmacro()

macro(glfw_github)
    message("GLFW")
    set(FILENAME "glfw-master.zip")
    if (NOT EXISTS "${DOWNLOAD_DIR}/${FILENAME}")
        message("  downloading")
        file(DOWNLOAD "https://github.com/glfw/glfw/archive/master.zip" "${DOWNLOAD_DIR}/${FILENAME}" STATUS downloaded)
    endif()
    if (EXISTS "${CMAKE_INSTALL_PREFIX}/glfw")
      message("  removing ${CMAKE_INSTALL_PREFIX}/glfw")
      execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_INSTALL_PREFIX}/glfw")
    endif()
    message("  extracting")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${DOWNLOAD_DIR}/${FILENAME}" WORKING_DIRECTORY "${SOURCE_DIR}")
    if (NOT EXISTS "${BUILD_DIR}/glfw")
      message("  creating ${BUILD_DIR}/glfw")
      file(MAKE_DIRECTORY "${BUILD_DIR}/glfw")
    endif()
    message("  generating")
    execute_process(COMMAND ${CMAKE_COMMAND} "-G${GENERATOR}" "-A${BUILD_ARCH}" "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/glfw" "${SOURCE_DIR}/glfw-master" WORKING_DIRECTORY "${BUILD_DIR}/glfw")
    message("  compiling")
    execute_process(COMMAND devenv.exe "${BUILD_DIR}/glfw/glfw.sln" /Build "Release|${BUILD_ARCH}" WORKING_DIRECTORY "${BUILD_DIR}/glfw")
    message("  installing")
    execute_process(COMMAND devenv.exe "${BUILD_DIR}/glfw/glfw.sln" /Build "Release|${BUILD_ARCH}" /Project INSTALL WORKING_DIRECTORY "${BUILD_DIR}/glfw")
endmacro()

macro(assimp_github)
    message("ASSIMP")
    set(FILENAME "assimp-master.zip")
    if (NOT EXISTS "${DOWNLOAD_DIR}/${FILENAME}")
        message("  downloading... (~46 MB)")
        file(DOWNLOAD "https://github.com/assimp/assimp/archive/master.zip" "${DOWNLOAD_DIR}/${FILENAME}" STATUS downloaded)
    endif()
    if (EXISTS "${CMAKE_INSTALL_PREFIX}/assimp")
      message("  removing ${CMAKE_INSTALL_PREFIX}/assimp")
      execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_INSTALL_PREFIX}/assimp")
    endif()
    message("  extracting")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${DOWNLOAD_DIR}/${FILENAME}" WORKING_DIRECTORY "${SOURCE_DIR}")
    if (NOT EXISTS "${BUILD_DIR}/assimp")
      message("  creating ${BUILD_DIR}/assimp")
      file(MAKE_DIRECTORY "${BUILD_DIR}/assimp")
    endif()
    message("  generating")
    execute_process(COMMAND ${CMAKE_COMMAND} "-G${GENERATOR}" "-A${BUILD_ARCH}" "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/assimp" "${SOURCE_DIR}/assimp-master" WORKING_DIRECTORY "${BUILD_DIR}/assimp")
    message("  compiling")
    execute_process(COMMAND devenv.exe "${BUILD_DIR}/assimp/assimp.sln" /Build "Release|${BUILD_ARCH}" WORKING_DIRECTORY "${BUILD_DIR}/assimp")
    message("  installing")
    execute_process(COMMAND devenv.exe "${BUILD_DIR}/assimp/assimp.sln" /Build "Release|${BUILD_ARCH}" /Project INSTALL WORKING_DIRECTORY "${BUILD_DIR}/assimp")
endmacro()

glew_sourceforge()
glfw_sourceforge()

# If the 3rdparty tools should be updated with additional libraries, commenting out these two lines avoids expensive recompilation of existing tools again.
message("deleting temp folder")
execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_CURRENT_SOURCE_DIR}/temp")

