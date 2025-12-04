cmake_minimum_required(VERSION 3.10)

project(cuda_ubench LANGUAGES CUDA C)

add_library(file_writer OBJECT file_writer.c)

add_executable(read16k managed_ubench.cu)
target_compile_definitions(read16k PRIVATE MEMSIZE=16384 UBENCH_READ_ONLY)
target_link_libraries(read16k PRIVATE file_writer)

add_executable(write16k managed_ubench.cu)
target_compile_definitions(write16k PRIVATE MEMSIZE=16384 UBENCH_WRITE_ONLY)
target_link_libraries(write16k PRIVATE file_writer)