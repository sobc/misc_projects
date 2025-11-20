#ifndef UTILS_H
#define UTILS_H

#include <openssl/md5.h>
#include <stddef.h>
#include <time.h>

typedef float *Matrix;
typedef struct timespec timer;

#define TIME_GET(timer) clock_gettime(CLOCK_MONOTONIC, &timer)

#define TIME_DIFF(timer1, timer2)                                              \
  ((timer2.tv_sec * 1.0E+9 + timer2.tv_nsec) -                                 \
   (timer1.tv_sec * 1.0E+9 + timer1.tv_nsec)) /                                \
      1.0E+9

#ifdef __cplusplus
extern "C" {
#endif
char *getMD5DigestStr(Matrix m, size_t size);
#ifdef __cplusplus
}
#endif

#endif // UTILS_H