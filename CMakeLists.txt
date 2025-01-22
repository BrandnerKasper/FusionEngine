cmake_minimum_required(VERSION 3.30)
project(FusionEngine)

enable_testing()  # Enable CTest

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the path to the LibTorch folder
set(Torch_DIR "${CMAKE_SOURCE_DIR}/libtorch/share/cmake/Torch")
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch;/opt/rocm")

# Get the current source directory (the project root)
set(PROJECT_ROOT_DIR "${CMAKE_SOURCE_DIR}")

# Find required libraries
find_package(Torch REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Fetch spdlog
include(FetchContent)
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.10.0
)
FetchContent_MakeAvailable(spdlog)

# ------------------- Main Executable -------------------
add_executable(FusionEngine
        src/main.cpp
        src/datasets/urban100.cpp src/datasets/urban100.h
        src/utils/utils.h
        src/models/sisr.cpp src/models/sisr.h
        src/metrics/metrics.cpp
        src/metrics/metrics.h
)

target_link_libraries(FusionEngine
        PRIVATE "${TORCH_LIBRARIES}" "${OpenCV_LIBS}" spdlog::spdlog
)
# Add defines like root
add_compile_definitions(FusionEngine PRIVATE PROJECT_ROOT_DIR="${PROJECT_ROOT_DIR}")

# ------------------- Test Executable -------------------
add_executable(runTests
        src/utils/utils.h
        src/models/sisr.cpp src/models/sisr.h
        src/datasets/urban100.h src/datasets/urban100.cpp
        tests/test_model.cpp
        tests/test_dataset.cpp
)

target_link_libraries(runTests
        PRIVATE "${TORCH_LIBRARIES}" "${OpenCV_LIBS}" spdlog::spdlog
        GTest::GTest GTest::Main pthread
)

# Register tests with CTest
add_test(NAME FusionTests COMMAND runTests)
# Add defines like root
add_compile_definitions(runTests PRIVATE PROJECT_ROOT_DIR="${PROJECT_ROOT_DIR}")