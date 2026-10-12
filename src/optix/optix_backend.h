#pragma once

#include <memory>
#include "cuda/gpu_scene.cuh"

namespace xn {
struct Scene;

// OptiX owns only traversal. All queues, materials and transport stay in CUDA.
class OptixBackend {
  public:
	OptixBackend();
	~OptixBackend();
	OptixBackend(const OptixBackend &) = delete;
	OptixBackend &operator=(const OptixBackend &) = delete;
	void build(const Scene &scene, GpuSceneData gpu_scene);
	void intersect(GpuSceneData scene, GpuPathState *paths, const int *queue, int *counters,
				   GpuHitRecord *hits, int count);
	void shadow(GpuSceneData scene, GpuPathState *paths, const GpuShadowWork *work, int *counters,
				int count);
	// Device buffers, for geometry regression tests and non-rendering ray queries.
	void query(GpuSceneData scene, const GpuRay *rays, GpuHitRecord *hits, bool *occluded,
			   int count);
	double build_seconds() const;
	size_t acceleration_bytes() const;

  private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};
} // namespace xn
