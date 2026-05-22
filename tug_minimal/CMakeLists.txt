cmake_minimum_required(VERSION 3.24)

project(tug_minimal LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
    Release
    CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake)
CPMAddPackage("gl:libeigen/eigen#3.4.1")
CPMAddPackage("gh:CLIUtils/CLI11@2.6.2")

set(BARITE_200_BENCH "${CMAKE_CURRENT_SOURCE_DIR}/inputs/tug/barite_200")
set(BARITE_LARGE_BENCH "${CMAKE_CURRENT_SOURCE_DIR}/inputs/tug/barite_large")
set(BARITE_HUGE_BENCH "${CMAKE_CURRENT_SOURCE_DIR}/inputs/tug/barite_huge")
set(SURFEX_BENCH "${CMAKE_CURRENT_SOURCE_DIR}/inputs/tug/surfex")

set(GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED_INCLUDE_DIR}")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/bench_defs.hpp.in"
  "${GENERATED_INCLUDE_DIR}/bench_defs.hpp"
  @ONLY
)

set(COMMON_SOURCES
  src/main.cpp
  src/run.cpp
)

set(COMMON_HEADERS
  src/io.hpp
  src/run.hpp
)

function(add_tug_target target_name implementation_dir)
  add_executable(${target_name}
    ${COMMON_SOURCES}
    ${COMMON_HEADERS}
    "${GENERATED_INCLUDE_DIR}/bench_defs.hpp"
  )

  target_include_directories(${target_name}
    PRIVATE
    "${GENERATED_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/${implementation_dir}"
  )

  target_link_libraries(${target_name}
    PRIVATE
    CLI11::CLI11
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
  )
endfunction()

add_tug_target(tug-old old)
add_tug_target(tug-optimized optimized)
