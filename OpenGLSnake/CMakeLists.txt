cmake_minimum_required(VERSION 3.31)

project(OpenGLSnake)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXE_FILE Fusion)

set(SOURCE_FILES main.cpp
                 Game.h
                 Game.cpp
)

add_executable(${EXE_FILE} ${SOURCE_FILES})

# Libs
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