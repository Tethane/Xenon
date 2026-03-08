#pragma once
// cuda/cuda_renderer.cuh — CUDA wavefront renderer declaration

#include "cuda/gpu_types.cuh"
#include "cuda/gpu_scene.cuh"

namespace xn {

// Forward declarations
struct Scene;
class  Camera;
class  TripleSwapchain;

class CudaRenderer {
public:
  CudaRenderer(int width, int height, int max_bounces, int min_bounces);
  ~CudaRenderer();

  CudaRenderer(const CudaRenderer&) = delete;
  CudaRenderer& operator=(const CudaRenderer&) = delete;

  // Upload scene data to GPU (call once after scene build)
  void upload_scene(const Scene& scene);

  // Render one sample and write to swapchain
  void render_frame(const Scene& scene, const Camera& camera, TripleSwapchain& swapchain);

  int spp() const { return spp_; }

private:
  int width_, height_;
  int max_bounces_, min_bounces_;
  int spp_ = 0;
  int num_pixels_;

  // GPU scene
  GpuSceneHost gpu_scene_;

  // Device buffers
  GpuPathState*   d_paths_       = nullptr;
  int*            d_ray_queue_   = nullptr;
  int*            d_ray_next_    = nullptr;
  GpuHitRecord*   d_hits_        = nullptr;
  GpuShadowWork*  d_shadow_work_ = nullptr;
  int*            d_shadow_queue_ = nullptr;
  int*            d_counters_    = nullptr;  // [0]=ray_count, [1]=ray_next, [2]=hit_count, [3]=shadow_count
  float*          d_accum_       = nullptr;  // W*H*3 accumulation buffer
  float*          d_output_      = nullptr;  // W*H*3 output (tonemapped)

  // Host staging
  float*          h_output_      = nullptr;

  void alloc_buffers();
  void free_buffers();
};

} // namespace xn
