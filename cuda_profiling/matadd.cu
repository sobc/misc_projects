#include <cuda.h>
#include <openssl/md5.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCKX 32
#define BLOCKY 32

#define TIME_GET(timer) clock_gettime(CLOCK_MONOTONIC, &timer)

#define TIME_DIFF(timer1, timer2)                                              \
  ((timer2.tv_sec * 1.0E+9 + timer2.tv_nsec) -                                 \
   (timer1.tv_sec * 1.0E+9 + timer1.tv_nsec)) /                                \
      1.0E+9

typedef int *Matrix;

static const size_t NELEMENTS = 1 << 14; // 16384

#define CUDA_ERR_CHECK(x)                                                      \
  do {                                                                         \
    cudaError_t err = x;                                                       \
    if ((err) != cudaSuccess) {                                                \
      printf("Error \"%s\" at %s :%d \n", cudaGetErrorString(err), __FILE__,   \
             __LINE__);                                                        \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

__global__ void matrix_add_kernel(const Matrix A, const Matrix B,
                                  const Matrix C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  C[row * NELEMENTS + col] =
      A[row * NELEMENTS + col] + B[row * NELEMENTS + col];
  return;
}

char *getMD5DigestStr(Matrix m) {
  MD5_CTX ctx;
  unsigned char sum[MD5_DIGEST_LENGTH];
  char *retval, *ptr;

  MD5_Init(&ctx);
  MD5_Update(&ctx, m, NELEMENTS * NELEMENTS);
  MD5_Final(sum, &ctx);

  retval = (char *)calloc(MD5_DIGEST_LENGTH * 2 + 1, sizeof(*retval));
  ptr = retval;

  for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
    snprintf(ptr, 3, "%02X", sum[i]);
    ptr += 2;
  }
  return retval;
}

double matrix_add_gpu(const Matrix A, const Matrix B, const Matrix C) {
  struct timespec start;
  struct timespec end;
  size_t size = NELEMENTS * NELEMENTS * sizeof(int);
  Matrix Ad, Bd, Cd;

  TIME_GET(start);
  CUDA_ERR_CHECK(cudaMalloc(&Ad, size));
  CUDA_ERR_CHECK(cudaMalloc(&Bd, size));
  CUDA_ERR_CHECK(cudaMalloc(&Cd, size));

  CUDA_ERR_CHECK(cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice));
  CUDA_ERR_CHECK(cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice));

  dim3 dBlock(BLOCKX, BLOCKY);
  dim3 dGrid(NELEMENTS / BLOCKX, NELEMENTS / BLOCKY);
  matrix_add_kernel<<<dGrid, dBlock>>>(Ad, Bd, Cd);
  CUDA_ERR_CHECK(cudaGetLastError());

  CUDA_ERR_CHECK(cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost));
  TIME_GET(end);

  cudaFree(Ad);
  cudaFree(Bd);
  cudaFree(Cd);
  return TIME_DIFF(start, end);
}

int main() {
  double result;
  Matrix A, B, C;
  A = (int *)malloc(NELEMENTS * NELEMENTS * sizeof(int));
  B = (int *)malloc(NELEMENTS * NELEMENTS * sizeof(int));
  C = (int *)malloc(NELEMENTS * NELEMENTS * sizeof(int));

  srand(0);

  for (int i = 0; i < NELEMENTS * NELEMENTS; i++) {
    A[i] = rand();
    B[i] = rand();
  }

  result = matrix_add_gpu(A, B, C);
  printf("Result GPU: %s Time: %lf\n", getMD5DigestStr(C), result);

  free(A);
  free(B);
  free(C);
}
