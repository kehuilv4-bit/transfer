cmake_minimum_required(VERSION 3.16)

add_library(dart_guide STATIC
    guide_detector.cpp
)

target_include_directories(dart_guide PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(dart_guide ${OpenCV_LIBS} yaml-cpp tools)
