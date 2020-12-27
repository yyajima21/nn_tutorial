cmake_minimum_required(VERSION 3.0.0)
project(basic)

set(CMAKE_CXX_STANDARD 14)  
set(CMAKE_CXX_FLAGS "-Wall -g")
list(APPEND CMAKE_PREFIX_PATH "../../libtorch/share/cmake")
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(basic basic.cpp)
target_link_libraries(basic ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})
set_property(TARGET basic PROPERTY CXX_STANDARD 14)