#include <cstdio>
#include <cuda_runtime.h>
#include <unistd.h>

#define MEM16K 4096 * 4

__global__ void read_only_kernel(const char *x, int size) {
  // The kernel reads from the Unified variable x ...
  volatile __shared__ char tmp;
  for (int i = 0; i < size; i++) {
    tmp = x[i];
  }
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
  cudaMallocManaged(&x, MEM16K);

  // ... and initializes it on the CPU.
  for (int i = 0; i < MEM16K; i++) {
    x[i] = (char)(i & 0xFF);
  }

  FILE *fp = fopen("/dev/null", "rw");

  if (fp == NULL) {
    fprintf(stderr, "Error opening /dev/null\n");
    return 1;
  }

  // print the result to prevent compiler optimization
  for (int i = 0; i < MEM16K; i++) {
    fprintf(fp, "%d\n", x[i]);
  }

  // Then we developed a kernel which only reads the managed memory in this
  // microbenchmark.
  read_only_kernel<<<1, 1>>>(x, MEM16K);

  cudaDeviceSynchronize();

  // Next, the host reads the data again after kernel return.
  for (int i = 0; i < MEM16K; i++) {
    fprintf(fp, "%d\n", x[i]);
  }

  fclose(fp);
  cudaFree(x);

  return 0;
}