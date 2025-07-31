cmake_minimum_required(VERSION 3.31)

project(OpenGLRenderer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set variables for source files
set(SOURCE_FILES main.cpp)
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
add_subdirectory(glad)

# GLM
FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm
        GIT_TAG 1.0.1
)
FetchContent_MakeAvailable(glm)

target_link_libraries(${EXE_FILE} PRIVATE glfw glad glm::glm)