cmake_minimum_required(VERSION 3.10)

project(cuda_ubench LANGUAGES CUDA)

add_executable(ro16k copy_read_only.cu)
target_compile_definitions(ro16k PRIVATE MEMSIZE=16384)

add_executable(ro4k copy_read_only.cu)
target_compile_definitions(ro4k PRIVATE MEMSIZE=4096)

add_executable(ro65k copy_read_only.cu)
target_compile_definitions(ro65k PRIVATE MEMSIZE=65536)

add_executable(ro65p1k copy_read_only.cu)
target_compile_definitions(ro65p1k PRIVATE MEMSIZE=66537)