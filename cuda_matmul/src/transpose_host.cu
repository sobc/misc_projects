#include <cstdlib>
#include <cuda.h>
#include <openssl/md5.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_utils.h"
#include "utils.h"

__global__ void matrix_mult_kernel(const Matrix A, const Matrix B, Matrix C,
                                   size_t n_rows) {
  float sum = 0;

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  for (size_t i = 0; i < n_rows; i++) {
    sum += A[row * n_rows + i] * B[col * n_rows + i];
  }

  C[row * n_rows + col] = sum;

  return;
}

void matrix_transpose(const Matrix in, Matrix out, size_t N) {
#pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      out[j * N + i] = in[i * N + j];
    }
  }
}

double matrix_mult_gpu(const Matrix A, const Matrix B, const Matrix C,
                       size_t N) {
  timer start, end;

  size_t mem_size = N * N * sizeof(float);
  Matrix Ad, Bd, Cd;

  // To save memory, we will store the result of the transposition in C
  matrix_transpose(B, C, N);

  TIME_GET(start);

  CUDA_ERR_CHECK(cudaMalloc(&Ad, mem_size));
  CUDA_ERR_CHECK(cudaMalloc(&Bd, mem_size));
  CUDA_ERR_CHECK(cudaMalloc(&Cd, mem_size));

  CUDA_ERR_CHECK(cudaMemcpy(Ad, A, mem_size, cudaMemcpyHostToDevice));
  // NOTE: As the transposed matrix is stored in C, we need to copy from C here
  CUDA_ERR_CHECK(cudaMemcpy(Bd, C, mem_size, cudaMemcpyHostToDevice));

  dim3 dGrid(N / 32, N / 32);
  dim3 dBlock(32, 32);
  matrix_mult_kernel<<<dGrid, dBlock>>>(Ad, Bd, Cd, N);
  CUDA_ERR_CHECK(cudaGetLastError());

  CUDA_ERR_CHECK(cudaMemcpy(C, Cd, mem_size, cudaMemcpyDeviceToHost));

  cudaFree(Ad);
  cudaFree(Bd);
  cudaFree(Cd);

  TIME_GET(end);
  return TIME_DIFF(start, end);
}

int main(int argc, char **argv) {
  double result;
  size_t n_rows;
  Matrix A, B, C;

  if (argc != 2) {
    fprintf(stderr, "Exactly one argument is required!\nUsage: %s [exponent]",
            argv[0]);
    return EXIT_FAILURE;
  }

  size_t exp = atoi(argv[1]);

  if (exp < 1 || exp > 15) {
    fprintf(stderr, "Exponent must be in [1,15]!\n");
    return EXIT_FAILURE;
  }

  n_rows = 1 << exp;

  size_t mem_size = n_rows * n_rows * sizeof(float);

  CUDA_ERR_CHECK(cudaMallocHost((void **)&A, mem_size));
  CUDA_ERR_CHECK(cudaMallocHost((void **)&B, mem_size));
  CUDA_ERR_CHECK(cudaMallocHost((void **)&C, mem_size));

  srand(RAND_SEED);

  for (size_t i = 0; i < n_rows * n_rows; i++) {
    A[i] = 100. / (rand() % 10000);
    B[i] = 100. / (rand() % 10000);
  }

  result = matrix_mult_gpu(A, B, C, n_rows);
  fprintf(stderr, "Time: %lf\n", result);
  fprintf(stdout, "Result GPU: %s \n", getMD5DigestStr(C, n_rows));

  FILE *foutput = fopen("transpose_host_result.bin", "w");

  if (foutput == NULL) {
    return EXIT_FAILURE;
  }

  fwrite(C, sizeof(float), n_rows * n_rows, foutput);

  fclose(foutput);
}
