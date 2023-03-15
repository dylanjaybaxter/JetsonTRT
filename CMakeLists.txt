cmake_minimum_required(VERSION 3.10)
project(jetsontrt)
set(CMAKE_CXX_STANDARD)
set(CMAKE_BUILD_TYPE Debug)

if(CMAKE_COMPILER_ISGNUXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-std=c++17)
endif()

find_package(OpenCV REQUIRED)
include_directories("${OpenCV_INCLUDE_DIRS}")
find_package(CUDA REQUIRED)
find_library(NVINFER NAMES nvinfer REQUIRED)
find_library(NVONNXPARSER NAMES nvonnxparser REQUIRED)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

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
)

add_executable(demo 
    demos/demo.cpp
)

target_link_libraries(demo
    yolov8
    ${OpenCV_LIBS}
    ${NVINFER}
    ${NVONNXPARSER}
    ${NVINFER_PLUGIN}
)

target_include_directories(demo
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

set_target_properties(demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../")

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