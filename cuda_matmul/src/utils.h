#ifndef UTILS_H
#define UTILS_H

#include <cuda.h>
#include <openssl/md5.h>
#include <stddef.h>
#include <time.h>

typedef float *Matrix;
typedef struct timespec timer;

#define CUDA_ERR_CHECK(x)                                                      \
  do {                                                                         \
    cudaError_t err = x;                                                       \
    if ((err) != cudaSuccess) {                                                \
      printf("Error \"%s\" at %s :%d \n", cudaGetErrorString(err), __FILE__,   \
             __LINE__);                                                        \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define TIME_GET(timer) clock_gettime(CLOCK_MONOTONIC, &timer)

#define TIME_DIFF(timer1, timer2)                                              \
  ((timer2.tv_sec * 1.0E+9 + timer2.tv_nsec) -                                 \
   (timer1.tv_sec * 1.0E+9 + timer1.tv_nsec)) /                                \
      1.0E+9

char *getMD5DigestStr(Matrix m, size_t size);

#endif // UTILS_H