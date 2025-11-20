#include "utils.h"

#include <openssl/md5.h>
#include <stdio.h>

char *getMD5DigestStr(Matrix m, size_t size) {
  MD5_CTX ctx;
  unsigned char sum[MD5_DIGEST_LENGTH];
  char *retval, *ptr;

  MD5_Init(&ctx);
  MD5_Update(&ctx, m, size * size);
  MD5_Final(sum, &ctx);

  retval = (char *)calloc(MD5_DIGEST_LENGTH * 2 + 1, sizeof(*retval));
  ptr = retval;

  for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
    snprintf(ptr, 3, "%02X", sum[i]);
    ptr += 2;
  }
  return retval;
}