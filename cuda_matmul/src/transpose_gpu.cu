#include <cstdlib>
#include <cuda.h>
#include <openssl/md5.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_utils.h"
#include "utils.h"

#define TILE_DIM 32

__global__ void matrix_transpose(const Matrix in, Matrix out, size_t n_rows) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (size_t i = 0; i < TILE_DIM; i += n_rows)
    tile[threadIdx.y + i][threadIdx.x] = in[(y + i) * width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (size_t i = 0; i < TILE_DIM; i += n_rows)
    out[(y + i) * width + x] = tile[threadIdx.x][threadIdx.y + i];
}

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

double matrix_mult_gpu(const Matrix A, const Matrix B, const Matrix C,
                       size_t N) {
  timer start, end;

  size_t mem_size = N * N * sizeof(float);
  Matrix Ad, Bd, Cd;

  TIME_GET(start);

  CUDA_ERR_CHECK(cudaMalloc(&Ad, mem_size));
  CUDA_ERR_CHECK(cudaMalloc(&Bd, mem_size));
  CUDA_ERR_CHECK(cudaMalloc(&Cd, mem_size));

  CUDA_ERR_CHECK(cudaMemcpy(Ad, A, mem_size, cudaMemcpyHostToDevice));
  CUDA_ERR_CHECK(cudaMemcpy(Cd, B, mem_size, cudaMemcpyHostToDevice));

  dim3 dBlock_transpose(32, 8);
  dim3 dGrid(N / 32, N / 32);

  matrix_transpose<<<dGrid, dBlock_transpose>>>(Cd, Bd, 8);
  CUDA_ERR_CHECK(cudaGetLastError());

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

  srand(0);

  for (size_t i = 0; i < n_rows * n_rows; i++) {
    A[i] = 100. / (rand() % 10000);
    B[i] = 100. / (rand() % 10000);
  }

  result = matrix_mult_gpu(A, B, C, n_rows);
  fprintf(stderr, "Time: %lf\n", result);
  fprintf(stdout, "Result GPU: %s \n", getMD5DigestStr(C, n_rows));

  FILE *foutput = fopen("wo_tiling.bin", "w");

  if (foutput == NULL) {
    return EXIT_FAILURE;
  }

  fwrite(C, sizeof(float), n_rows * n_rows, foutput);
}
