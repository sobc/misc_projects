#include <cstdio>
#include <cuda_runtime.h>
#include <unistd.h>

#ifndef MEMSIZE
#define MEMSIZE 4096 * 4
#endif

__global__ void read_only_kernel(const char *x, int size) {
  // The kernel reads from the Unified variable x ...
  volatile __shared__ char tmp;
  for (int i = 0; i < size; i++) {
    tmp = x[i];
  }
}

int write_to_file(const char *filename, const char *data, size_t size) {
  FILE *fp = fopen(filename, "wb");

  if (fp == NULL) {
    return -1; // Error opening file
  }

  fwrite(data, sizeof(char), size, fp);
  fclose(fp);

  return 0; // Success
}

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

  if (write_to_file("/dev/null", x, MEMSIZE) != 0) {
    fprintf(stderr, "Error writing to file\n");
    cudaFree(x);
    return -1;
  }

  // Then we developed a kernel which only reads the managed memory in this
  // microbenchmark.
  read_only_kernel<<<1, 1>>>(x, MEMSIZE);

  cudaDeviceSynchronize();

  // Next, the host reads the data again after kernel return.
  if (write_to_file("/dev/null", x, MEMSIZE) != 0) {
    fprintf(stderr, "Error writing to file\n");
    cudaFree(x);
    return -1;
  }

  cudaFree(x);

  return 0;
}