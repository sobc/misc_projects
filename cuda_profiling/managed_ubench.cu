#include "file_writer.h"

#include <cstdio>
#include <cuda_runtime.h>
#include <unistd.h>

#ifndef MEMSIZE
#define MEMSIZE 4096 * 4
#endif

#if defined(UBENCH_READ_ONLY)
#define RELFILE "./read_ubench.bin"
__global__ void read_kernel(const char *x, int size) {
  // The kernel reads from the Unified variable x ...
  volatile __shared__ char tmp;
  for (int i = 0; i < size; i++) {
    tmp = x[i];
  }
}
#elif defined(UBENCH_WRITE_ONLY)
#define RELFILE "./write_ubench.bin"
__global__ void write_kernel(char *x, int size) {
  // The kernel writes to the Unified variable x ...
  for (int i = 0; i < size; i++) {
    x[i] = (char)(i & 0xEA);
  }
}
#else
#error "Either UBENCH_READ_ONLY or UBENCH_WRITE_ONLY must be defined"
#endif

int main(void) {
  char *x;
  char *a;

  printf("Page size is set to: %d bytes\n", getpagesize());

  /* Initialize context */
  cudaMallocManaged(&a, 128);
  cudaDeviceSynchronize();
  cudaFree(a);

  // The microbenchmark allocates a Unified variable x ...
  cudaMallocManaged(&x, MEMSIZE);

  // ... and initializes it on the CPU.
  for (int i = 0; i < MEMSIZE; i++) {
    x[i] = (char)(i & 0xFF);
  }

#if defined(UBENCH_READ_ONLY)
  // Then we developed a kernel which only reads the managed memory in this
  // microbenchmark.
  read_kernel<<<1, 1>>>(x, MEMSIZE);
#elif defined(UBENCH_WRITE_ONLY)
  write_kernel<<<1, 1>>>(x, MEMSIZE);
#endif

  cudaDeviceSynchronize();

  // Next, the host reads the data again after kernel return.
  if (write_to_file(RELFILE, x, MEMSIZE) != 0) {
    fprintf(stderr, "Error writing to file\n");
    cudaFree(x);
    return -1;
  }

  cudaFree(x);

  return 0;
}