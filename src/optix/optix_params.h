#pragma once
#include <optix.h>
#include "cuda/gpu_scene.cuh"

namespace xn {
struct OptixPlaneRef {
	int instance, primitive;
};

struct OptixParams {
	OptixTraversableHandle handle = 0;
	GpuSceneData scene{};
	GpuPathState *paths = nullptr;
	const int *queue = nullptr;
	int *counters = nullptr;
	GpuHitRecord *hits = nullptr;
	const GpuShadowWork *shadows = nullptr;
	const GpuRay *rays = nullptr;
	bool *occluded = nullptr;
	const OptixPlaneRef *planes = nullptr;
	int num_planes = 0;
	int mode = 0; // 0 closest wavefront, 1 shadow wavefront, 2 query
};
} // namespace xn
