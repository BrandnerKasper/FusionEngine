cmake_minimum_required(VERSION 3.30)
project(FusionEngine)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the path to the LibTorch folder
set(Torch_DIR "${CMAKE_SOURCE_DIR}/libtorch/share/cmake/Torch")
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch;/opt/rocm")

# Get the current source directory (the project root)
set(PROJECT_ROOT_DIR "${CMAKE_SOURCE_DIR}")

# Find PyTorch
find_package(Torch REQUIRED)
# Find OpenCV
find_package(OpenCV REQUIRED)
# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Fetch spdlog
# Include the FetchContent module
include(FetchContent)

# Download spdlog
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.10.0  # Specify the version here
)

FetchContent_MakeAvailable(spdlog)

add_executable(FusionEngine src/main.cpp
        datasets/urban100.cpp
        datasets/urban100.h
        utils/utils.h)
target_link_libraries(FusionEngine PRIVATE "${TORCH_LIBRARIES}" "${OpenCV_LIBS}" spdlog::spdlog)
# Add defines like root
add_compile_definitions(FusionEngine PRIVATE PROJECT_ROOT_DIR="${PROJECT_ROOT_DIR}")
