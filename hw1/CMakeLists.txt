cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME hw1)
project(${PROJECT_NAME} CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

## compiler option setting
set(CMAKE_BUILD_TYPE "TEST")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -cxx=clang -Wall -fsanitize=undefined,address -Werror -Wextra -Wno-unused-variable -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_TEST "-cxx=clang++ -Wall -O2 -fsanitize=undefined,address")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
## Use C++14 std
set(CMAKE_CXX_STANDARD 14)
## Use clang instead of gcc
set(CMAKE_CXX_COMPILER "mpicxx")

file(GLOB SOURCES "*.cc")
add_executable(hw1 ${SOURCES})
