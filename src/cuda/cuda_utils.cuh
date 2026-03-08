#pragma once
// cuda/cuda_utils.cuh — CUDA error checking and device query utilities

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace xn {

// ─── Error checking ──────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(err));                                    \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

// Check for errors after kernel launches (asynchronous errors).
// In release builds this is a no-op unless XN_CUDA_DEBUG is defined.
#if !defined(NDEBUG) || defined(XN_CUDA_DEBUG)
#define CUDA_SYNC_CHECK()                                                      \
  do {                                                                         \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  } while (0)
#else
#define CUDA_SYNC_CHECK()                                                      \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                       \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA async error at %s:%d — %s\n",                \
                   __FILE__, __LINE__, cudaGetErrorString(err));                \
      std::abort();                                                            \
    }                                                                          \
  } while (0)
#endif

// ─── Device info ─────────────────────────────────────────────────────────────

inline void cuda_print_device_info() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("[CUDA] Device %d: %s\n", device, prop.name);
  std::printf("[CUDA]   Compute capability: %d.%d\n", prop.major, prop.minor);
  std::printf("[CUDA]   SMs: %d, max threads/SM: %d\n",
              prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor);
  std::printf("[CUDA]   Global memory: %zu MB\n",
              prop.totalGlobalMem / (1024 * 1024));
  std::printf("[CUDA]   Shared memory/block: %zu KB\n",
              prop.sharedMemPerBlock / 1024);
  std::printf("[CUDA]   Warp size: %d\n", prop.warpSize);
}

} // namespace xn
