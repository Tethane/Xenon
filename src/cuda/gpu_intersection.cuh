#pragma once
#include "cuda/gpu_traverse.cuh"

namespace xn {
__device__ inline void gpu_prepare_intersection(GpuPathState &path, const GpuSceneData &scene) {
	if (scene.has_alpha_blend)
		path.ray.alpha_seed = path.rng.next_uint();
}

// Common transport boundary after either software CUDA or OptiX traversal.
// Keep environment/medium event handling shared to prevent backend drift.
__device__ inline void gpu_complete_intersection(GpuPathState *paths, int path_idx,
												 GpuHitRecord rec, bool found, GpuHitRecord *hits,
												 int *counters, const GpuSceneData &scene) {
	auto &ps = paths[path_idx];
	auto medium = gpu_sample_medium(scene.media, scene.num_media, ps.ray,
									found ? rec.t : kGpuInfinity, ps.rng);
	ps.throughput *= medium.weight;
	float3 interior = ps.absorption.current();

	if (max_component3(interior) > 0)
		ps.throughput *=
			gpu_beer_lambert(interior, medium.scattered ? length3(medium.position - ps.ray.origin)
														: (found ? rec.t : kGpuInfinity));

	if (max_component3(ps.throughput) == 0) {
		ps.set_active(false);

		return;
	}

	if (medium.scattered) {
		rec = GpuHitRecord{};
		rec.pos = medium.position;
		rec.mat_id = -1;
		rec.u = medium.g;
		found = true;
	}

	if (found) {
		int slot = atomicAdd(counters + 2, 1);

		if (slot < scene.num_pixels) {
			rec.path_idx = path_idx;
			hits[slot] = rec;
		}
	} else {
		if (scene.env.active()) {
			float pdf = scene.env.pdf(ps.ray.dir), a = ps.prev_bsdf_pdf_sa;
			float scale = fmaxf(a, pdf);
			float mis =
				ps.was_prev_delta()
					? 1.f
					: (scale > 0 ? (a / scale) * (a / scale) /
									   ((a / scale) * (a / scale) + (pdf / scale) * (pdf / scale))
								 : 0);
			ps.radiance += ps.throughput * scene.env.evaluate(normalize3(ps.ray.dir)) * mis;
		}
		ps.set_active(false);
	}
}
} // namespace xn
