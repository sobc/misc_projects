cmake_minimum_required(VERSION 3.23)

set(CMAKE_CXX_STANDARD 17)

project(matmul_cuda)

enable_language(CUDA C)

add_subdirectory(src)
