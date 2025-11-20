find_package(OpenSSL REQUIRED)
find_package(OpenMP)

add_library(utils OBJECT utils.c)
target_link_libraries(utils PUBLIC OpenSSL::Crypto)

add_executable(naive naive.cu)
target_link_libraries(naive PRIVATE utils)

add_executable(naive_gpu_transpose transpose_gpu.cu)
target_link_libraries(naive_gpu_transpose PRIVATE utils)

add_executable(naive_host_transpose transpose_host.cu)
target_link_libraries(naive_host_transpose PRIVATE utils)

if (OpenMP_CXX_FOUND)
  target_link_libraries(naive_host_transpose PRIVATE OpenMP::OpenMP_CXX)
endif()