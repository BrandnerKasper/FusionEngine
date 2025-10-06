cmake_minimum_required(VERSION 3.31)
project(NeuralHook)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXE_FILE Hook)

set(SOURCE_FILES main.cpp)

#set(MODEL_ROOT ${CMAKE_SOURCE_DIR}/pretrained_models/)

add_executable(${EXE_FILE} ${SOURCE_FILES})

# LIBS
# Set the path to the LibTorch folder
#set(Torch_DIR "${CMAKE_SOURCE_DIR}/libtorch/share/cmake/Torch")
#set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch;/opt/rocm-6.3.0")

# TORCH CPU
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/extern/libtorch")
find_package(Torch REQUIRED)

# Get the current source directory (the project root)
set(PROJECT_ROOT_DIR "${CMAKE_SOURCE_DIR}")

target_link_libraries(${EXE_FILE} PRIVATE ${TORCH_LIBRARIES}) #MODEL_ROOT="${MODEL_ROOT}")

