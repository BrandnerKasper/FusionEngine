cmake_minimum_required(VERSION 3.31)

project(FusionEngine)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXE_FILE Fusion)

# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------
set(SOURCE_FILES
        main.cpp
        Application.cpp              Application.h
        settings.h                   assets.h
        random.h

        Window/IWindow.h             Window/GLFWWindow.cpp   Window/GLFWWindow.h

        Input/IInput.h
        Input/GLFWInput.cpp          Input/GLFWInput.h
        Input/TerminalInput.cpp      Input/TerminalInput.h

        Renderer/IRenderer.h
        Renderer/OpenGLRenderer.cpp  Renderer/OpenGLRenderer.h
        Renderer/ASCIIRenderer.cpp   Renderer/ASCIIRenderer.h
        Renderer/NeuralRenderer.cpp  Renderer/NeuralRenderer.h
        Renderer/Shader.cpp          Renderer/Shader.h
        Renderer/Mesh.cpp            Renderer/Mesh.h
        Renderer/Sprite.cpp          Renderer/Sprite.h
        Renderer/Texture.cpp         Renderer/Texture.h
        Renderer/FrameBuffer.cpp     Renderer/FrameBuffer.h

        Game/Game.cpp                Game/Game.h
        Game/Player.cpp              Game/Player.h
        Game/Tile.h
)

add_executable(${EXE_FILE} ${SOURCE_FILES})


# ------------------------------------------------------------------------------
# Third-party Libs
# ------------------------------------------------------------------------------
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

# TORCH GPU (ROCM)
#set(Torch_DIR "${CMAKE_SOURCE_DIR}/extern/libtorch/share/cmake/Torch")
#set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/extern/libtorch;/opt/rocm-6.3.0")
#find_package(Torch REQUIRED)

# TORCH CPU
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/extern/libtorch")
find_package(Torch REQUIRED)

target_link_libraries(${EXE_FILE} PRIVATE glfw glad glm::glm "${TORCH_LIBRARIES}")


# ------------------------------------------------------------------------------
# Compile definitions for assets/data (multi-config safe)
#   - Release: use relative "./assets/"
#   - Others : use source-tree assets
# ------------------------------------------------------------------------------
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(ASSET_ROOT ./assets/)
else ()
    set(ASSET_ROOT ${CMAKE_SOURCE_DIR}/assets/)
endif ()


# ------------------------------------------------------------------------------
# Define Data ROOT
# ------------------------------------------------------------------------------
set(DATA_ROOT ${CMAKE_SOURCE_DIR}/data/)

target_compile_definitions(${EXE_FILE} PRIVATE ASSET_ROOT="${ASSET_ROOT}" DATA_ROOT="${DATA_ROOT}")