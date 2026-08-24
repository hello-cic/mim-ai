cmake_minimum_required(VERSION 3.10)
project(Mim-AI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE RNN)

# 支持多IDE
if(APPLE)
    set(CMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH NO)
endif()
