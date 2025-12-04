#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
int write_to_file(const char *filename, const char *data, size_t size);

#ifdef __cplusplus
}
#endif
