#include "file_writer.h"

#include <stdio.h>

int write_to_file(const char *filename, const char *data, size_t size) {
  FILE *fp = fopen(filename, "wb");

  if (fp == NULL) {
    return -1; // Error opening file
  }

  fwrite(data, sizeof(char), size, fp);
  fclose(fp);

  return 0; // Success
}