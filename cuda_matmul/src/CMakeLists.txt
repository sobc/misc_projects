find_package(OpenSSL REQUIRED)

add_executable(naive naive.cu)
target_link_libraries(naive PRIVATE OpenSSL::Crypto)
