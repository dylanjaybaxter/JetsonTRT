cmake_minimum_required(VERSION 3.10)
project(jetsontrt)
set(CMAKE_CXX_STANDARD)
set(CMAKE_BUILD_TYPE Debug)

if(CMAKE_COMPILER_ISGNUXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-std=c++17)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
include_directories("${OpenCV_INCLUDE_DIRS}")
find_package(CUDA REQUIRED)
find_library(NVINFER NAMES nvinfer REQUIRED)
find_library(NVONNXPARSER NAMES nvonnxparser REQUIRED)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

include_directories(
    include
)

add_library(yolov8
    src/yolov8.cpp
)

ament_target_dependencies(yolov8
    OpenCV
)

target_include_directories(yolov8
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

ament_export_targets(export_yolov8 HAS_LIBRARY_TARGET)

target_link_libraries(
    ${NVINFER}
    ${NVONNXPARSER}
    ${NVINFER_PLUGIN}
)

add_executable(demo demos/demo.cpp)

target_link_libraries(demo
    yolov8
    ${OpenCV_LIBS}
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