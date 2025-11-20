find_package(OpenSSL REQUIRED)

add_library(utils OBJECT utils.c)
target_link_libraries(utils PUBLIC OpenSSL::Crypto)

add_executable(naive naive.cu)
target_link_libraries(naive PRIVATE utils)
