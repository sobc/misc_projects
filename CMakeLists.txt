cmake_minimum_required(VERSION 3.23)

project(MiscProjects)

option(BUILD_CUDA_MATMUL "Build the CUDA matrix multiplication project" ON)

if(BUILD_CUDA_MATMUL)
    add_subdirectory(cuda_matmul)
endif()