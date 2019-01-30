cmake_minimum_required(VERSION 2.8)
project(facetracker)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

set(SOURCES facetracker.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
add_definitions(-std=c++11)
add_executable(facetracker ${SOURCES} ${HEADERS})
target_link_libraries(facetracker ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS facetracker RUNTIME DESTINATION bin)
