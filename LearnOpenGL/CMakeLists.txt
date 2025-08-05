cmake_minimum_required(VERSION 3.31)

project(OpenGLRenderer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set variables for source files
set(SOURCE_FILES main.cpp
        Shader.h
        Shader.cpp
        Assets.h)

# set executable name
set(EXE_FILE OpenGLRenderer)

add_executable(${EXE_FILE} ${SOURCE_FILES})

include(FetchContent)

# GLFW
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw
        GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)

# GLAD
add_subdirectory(extern/glad)

# GLM
FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm
        GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

target_link_libraries(${EXE_FILE} PRIVATE glfw glad glm::glm)

# define asset root
target_compile_definitions(${EXE_FILE} PRIVATE ASSET_ROOT=./assets/)

# copy assets to output directory before build
add_custom_command(TARGET ${EXE_FILE} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:${EXE_FILE}>/assets)

# add extern to include dirs so header only libs are easy to find
target_include_directories(${EXE_FILE} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern)