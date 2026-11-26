// Windows/MSVC shims for POSIX-only constructs in ndarray_backend_cpu.cc.
// Uses plain malloc so the source's free() stays valid (MSVC guarantees only
// 16-byte alignment, which is fine for correctness).
#pragma once
#include <cstdlib>
#include <cassert>
#define __restrict__ __restrict
inline int posix_memalign(void** p, size_t, size_t size) {
  *p = std::malloc(size);
  return *p ? 0 : 12;
}
