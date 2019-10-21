cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME hw1)
project(${PROJECT_NAME} CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

## compiler option setting
#set(CMAKE_BUILD_TYPE "FDOUSE")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -cxx=clang++ -Wall -fsanitize=undefined,address -Werror -Wextra -Wno-unused-variable -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_TEST "-cxx=clang++ -Wall -O3 -fsanitize=undefined,address -DDEBUG")
set(CMAKE_CXX_FLAGS_FDOGEN "-cxx=clang++ -fprofile-instr-generate -O3 -flto -march=native")
set(CMAKE_CXX_FLAGS_FDOUSE "-cxx=clang++ -fprofile-instr-use=code.profdata -O3 -flto -march=native -DPERF")
set(CMAKE_CXX_FLAGS_PROFILE "-cxx=clang++ -O3 -march=native -DPERF")
set(CMAKE_CXX_FLAGS_RELEASE "-cxx=clang++ -O3 -march=native")

## Use C++17 std
set(CMAKE_CXX_STANDARD 17)
## Use clang instead of gcc
set(CMAKE_CXX_COMPILER "mpicxx")

file(GLOB SOURCES "*.cc")
add_executable(hw1 ${SOURCES})
