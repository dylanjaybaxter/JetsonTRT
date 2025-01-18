cmake_minimum_required(VERSION 3.10)
project(jetsontrt)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

if(CMAKE_COMPILER_ISGNUXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-std=c++17)
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories("${OpenCV_INCLUDE_DIRS}")

# Find CUDA
find_package(CUDA REQUIRED)

# Find TensorRT libraries
find_library(NVINFER NAMES nvinfer REQUIRED)
find_library(NVONNXPARSER NAMES nvonnxparser REQUIRED)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# Find spdlog
find_package(spdlog REQUIRED)

# Include directories
include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}  
)

# Library for yolov8
add_library(yolov8
    src/yolov8.cpp
    src/generic.cpp
)

target_include_directories(yolov8
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(yolov8
    ${NVINFER}
    ${NVONNXPARSER}
    ${NVINFER_PLUGIN}
    ${CUDA_LIBRARIES}
    spdlog::spdlog  
)

# Demo executable
add_executable(demo 
    demos/demo.cpp
)

target_link_libraries(demo
    yolov8
    ${OpenCV_LIBS}
    ${NVINFER}
    ${NVONNXPARSER}
    ${NVINFER_PLUGIN}
    spdlog::spdlog  
)

target_include_directories(demo
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

set_target_properties(demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../")

# Install configuration
install(
    DIRECTORY include/
    DESTINATION include
)

install(
    TARGETS yolov8
    EXPORT export_yolov8
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)
