// cuda/cuda_renderer.cu — CUDA wavefront path tracing renderer
//
// Full wavefront pipeline: raygen → intersect → NEE → shade → shadow → accumulate
// All work stays on device; only the final framebuffer is copied back per sample.

#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <chrono>
#ifdef XENON_HAS_OPTIX
#include "optix/optix_backend.h"
#else
// Keep the unique_ptr type complete in CPU/software-CUDA-only builds.
namespace xn {
class OptixBackend {};
} // namespace xn
#endif

#include "cuda/cuda_renderer.cuh"
#include "cuda/cuda_utils.cuh"
#include "cuda/gpu_traverse.cuh"
#include "cuda/gpu_intersection.cuh"
#include "cuda/gpu_bsdf.cuh"
#include "scene/scene.h"
#include "camera/camera.h"
#include "render/swapchain.h"

namespace xn {

// ─── Counter indices ─────────────────────────────────────────────────────────
enum CounterIdx { CNT_RAY = 0, CNT_RAY_NEXT = 1, CNT_HIT = 2, CNT_SHADOW = 3, NUM_COUNTERS = 4 };

// ─── Block size for all kernels ──────────────────────────────────────────────
constexpr int kBlockSize = 256;

inline int grid_size(int n) {
	return (n + kBlockSize - 1) / kBlockSize;
}

// ═════════════════════════════════════════════════════════════════════════════
// KERNELS
// ═════════════════════════════════════════════════════════════════════════════

// ─── Ray Generation ──────────────────────────────────────────────────────────

__global__ void raygen_kernel(GpuPathState *paths, int *ray_queue, int *counters, GpuCamera camera,
							  int spp, uint64_t seed, int width, int height) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int num_pixels = width * height;

	if (idx >= num_pixels)
		return;

	int x = idx % width;
	int y = idx / width;

	// Seed RNG with pixel index and sample count
	GpuPCG rng = gpu_seed_path_pcg(uint32_t(idx), uint32_t(spp), seed);

	float u = ((float)x + rng.next_float()) / (float)width;
	float v = ((float)y + rng.next_float()) / (float)height;

	GpuPathState &ps = paths[idx];
	ps.absorption.depth = 0;

	if (camera.medium_material >= 0)
		ps.absorption.enter(camera.medium_material, camera.absorption);

	if (camera.lens_radius > 0.f) {
		float lens_u = rng.next_float(), lens_v = rng.next_float();
		ps.ray = camera.get_ray(u, v, lens_u, lens_v);
	} else
		ps.ray = camera.get_ray(u, v);
	ps.throughput = make_f3(1.f);
	ps.radiance = make_f3(0.f);
	ps.rng = rng;
	ps.pixel_idx = idx;
	ps.depth = 0;
	ps.prev_bsdf_pdf_sa = 0.f;
	ps.eta_scale = 1.f;
	ps.prev_pos = make_f3(0.f);
	ps.flags = 1;			 // active
	ps.set_prev_delta(true); // first bounce counts as delta for MIS

	ray_queue[idx] = idx;

	if (idx == 0) {
		counters[CNT_RAY] = num_pixels;
	}
}

// ─── Intersection ────────────────────────────────────────────────────────────

__global__ void intersect_kernel(GpuPathState *paths, const int *ray_queue, const int *counters,
								 GpuHitRecord *hits, int *hit_counter, GpuSceneData scene) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int num_rays = counters[CNT_RAY];

	if (tid >= num_rays)
		return;

	int path_idx = ray_queue[tid];
	GpuPathState &ps = paths[path_idx];

	if (!ps.is_active())
		return;

	GpuHitRecord rec{};
	rec.t = kGpuInfinity;

	gpu_prepare_intersection(ps, scene);
	bool found = gpu_tlas_intersect(ps.ray, scene, rec);
	gpu_complete_intersection(paths, path_idx, rec, found, hits, hit_counter, scene);
}

// ─── NEE (Next Event Estimation) ─────────────────────────────────────────────

__device__ inline void queue_shadow(GpuShadowWork *work, int *counters, int path_idx, GpuRay ray,
									float3 contribution) {
	if (max_component3(contribution) <= 0.f)
		return;
	int slot = atomicAdd(&counters[CNT_SHADOW], 1);
	work[slot].path_idx = path_idx;
	work[slot].ray = ray;
	work[slot].contrib = contribution;
}

__global__ void nee_kernel(GpuPathState *paths, const GpuHitRecord *hits, const int *counters,
						   GpuShadowWork *shadow_work, int *shadow_counter, GpuSceneData scene,
						   int max_bounces) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= counters[CNT_HIT])
		return;
	GpuHitRecord hit = hits[tid];
	int path_idx = hit.path_idx;
	GpuPathState &ps = paths[path_idx];
	bool volume = hit.mat_id < 0;
	GpuMaterial mat{};

	if (!volume) {
		mat = gpu_evaluate_material(scene, hit);
		float3 emission = mat.emission;
		float light_pdf = 0;
		bool emission_two_sided = mat.emissionTwoSided != 0;
		int light_index = gpu_light_for_hit(scene, hit);

		if (light_index >= 0) {
			const auto &light = scene.lights[light_index];
			emission = light.emission;
			emission_two_sided = light.emissionTwoSided != 0;

			if (light.material_emission)
				emission *=
					gpu_texture_rgb(scene, scene.materials[hit.mat_id].emissionTexture, hit.tex_u,
									hit.tex_v, scene.materials[hit.mat_id].textureLod);
			float cosine = dot3(light.normal, -ps.ray.dir);

			if (light.emissionTwoSided)
				cosine = fabsf(cosine);

			if (cosine > 0)
				light_pdf =
					light.selection_pmf * length_sq3(hit.pos - ps.prev_pos) / (cosine * light.area);
		}
		float weight =
			ps.was_prev_delta() ? 1.f : gpu_power_heuristic(ps.prev_bsdf_pdf_sa, light_pdf);

		if (hit.front_face || emission_two_sided)
			ps.radiance += ps.throughput * emission * weight;

		if (mat.isDelta())
			return;
	}
	float3 wo = -ps.ray.dir, ns = volume ? wo : gpu_shading_normal(hit, wo, mat.isTransmissive());
	GpuOnb onb(ns);
	float3 wo_local = onb.to_local(wo);
	bool last = ps.depth + 1 == max_bounces;
	auto evaluate = [&](float3 wi, float3 &value, float &pdf) {
		if (volume) {
			pdf = gpu_henyey_greenstein(dot3(ps.ray.dir, wi), hit.u);
			value = make_f3(pdf);

			return true;
		}

		if (!gpu_valid_scattering(wo, wi, ns, hit.geo_normal))
			return false;
		float3 local = onb.to_local(wi);
		value = gpu_bsdf_eval_for_nee(wo_local, local, mat) * fabsf(local.z);
		pdf = gpu_bsdf_pdf_for_nee(wo_local, local, mat);

		return true;
	};
	auto origin_for = [&](float3 wi) {
		return volume ? hit.pos : gpu_offset_ray_origin(hit.pos, hit.geo_normal, wi, kGpuShadowEps);
	};
	auto contribute = [&](GpuRay shadow, float3 contribution) {
		if (max_component3(contribution) <= 0)
			return;

		if (scene.has_alpha_blend)
			shadow.alpha_seed = ps.rng.next_uint();
		contribution *= gpu_medium_transmittance(scene.media, scene.num_media, shadow, shadow.tmax);
		float3 sigma = ps.absorption.current();

		if (!volume && mat.isTransmissive() &&
			dot3(wo, hit.geo_normal) * dot3(shadow.dir, hit.geo_normal) < 0)
			sigma = hit.front_face ? mat.absorption : ps.absorption.after_leave(hit.mat_id);

		if (max_component3(sigma) > 0)
			contribution *= gpu_beer_lambert(sigma, shadow.tmax);
		queue_shadow(shadow_work, shadow_counter, path_idx, shadow, ps.throughput * contribution);
	};

	if (scene.num_lights > 0) {
		float pick_pdf;
		int idx =
			gpu_sample_light(scene, double(ps.rng.next_uint()) * (1. / 4294967296.), pick_pdf);
		const auto &light = scene.lights[idx];
		float u1 = ps.rng.next_float(), u2 = ps.rng.next_float(), su = sqrtf(u1);
		float3 position = light.p0 * (1 - su) + light.p1 * (u2 * su) + light.p2 * (su * (1 - u2));
		float3 segment = position - hit.pos;
		float dist_sq = length_sq3(segment);

		if (dist_sq > 0) {
			float3 wi = segment / sqrtf(dist_sq), value;
			float b_pdf;
			float cos_l = dot3(light.normal, -wi);

			if (light.emissionTwoSided)
				cos_l = fabsf(cos_l);

			if (cos_l > 0 && evaluate(wi, value, b_pdf)) {
				float pdf = pick_pdf * dist_sq / (cos_l * light.area);
				float mis = last ? 1.f : gpu_power_heuristic(pdf, b_pdf);
				float3 origin = origin_for(wi),
					   target = gpu_offset_ray_origin(position, light.normal, -wi, kGpuShadowEps);
				segment = target - origin;
				float distance = length3(segment);

				if (distance > 0)
					contribute({origin, segment / distance, 0.f, distance},
							   value * gpu_light_emission(scene, light, u2 * su, su * (1 - u2)) *
								   (mis / pdf));
			}
		}
	}

	if (scene.num_dir_lights > 0) {
		int idx = min(int(ps.rng.next_float() * scene.num_dir_lights), scene.num_dir_lights - 1);
		const auto &light = scene.dir_lights[idx];
		float3 wi = light.direction, value;
		float pdf;

		if (evaluate(wi, value, pdf))
			contribute({origin_for(wi), wi, 0.f, kGpuInfinity},
					   value * light.get_emission() * float(scene.num_dir_lights));
	}

	if (scene.num_point_lights > 0) {
		int idx =
			min(int(ps.rng.next_float() * scene.num_point_lights), scene.num_point_lights - 1);
		const auto &light = scene.point_lights[idx];
		float3 segment = light.position - hit.pos;
		float dist_sq = length_sq3(segment);

		if (dist_sq > 0) {
			float3 wi = segment / sqrtf(dist_sq), value;
			float pdf;

			if (evaluate(wi, value, pdf)) {
				float3 origin = origin_for(wi);
				segment = light.position - origin;
				float distance = length3(segment);

				if (distance > 0)
					contribute({origin, segment / distance, 0.f, distance},
							   value * light.intensity *
								   (light.falloff(-wi) * scene.num_point_lights / dist_sq));
			}
		}
	}

	if (scene.env.active()) {
		float u1 = ps.rng.next_float(), u2 = ps.rng.next_float();
		auto light = scene.env.sample(u1, u2);
		float3 value;
		float b_pdf;

		if (light.pdf > 0 && evaluate(light.direction, value, b_pdf)) {
			float mis = last ? 1.f : gpu_power_heuristic(light.pdf, b_pdf);
			contribute({origin_for(light.direction), light.direction, 0.f, kGpuInfinity},
					   value * light.radiance * (mis / light.pdf));
		}
	}
}

// ─── Shadow ──────────────────────────────────────────────────────────────────

__global__ void shadow_kernel(GpuPathState *paths, const GpuShadowWork *shadow_work,
							  const int *counters, GpuSceneData scene) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int num_shadow = counters[CNT_SHADOW];

	if (tid >= num_shadow)
		return;

	const GpuShadowWork &sw = shadow_work[tid];

	if (!gpu_tlas_intersects(sw.ray, scene)) {
		// No occlusion — add light contribution
		atomicAdd(&paths[sw.path_idx].radiance.x, sw.contrib.x);
		atomicAdd(&paths[sw.path_idx].radiance.y, sw.contrib.y);
		atomicAdd(&paths[sw.path_idx].radiance.z, sw.contrib.z);
	}
}

// ─── Shade (unified — handles all material types) ────────────────────────────

__global__ void shade_kernel(GpuPathState *paths, const GpuHitRecord *hits, const int *counters,
							 int *ray_queue_next, int *next_counter, GpuSceneData scene,
							 int max_bounces, int min_bounces) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int num_hits = counters[CNT_HIT];

	if (tid >= num_hits)
		return;

	GpuHitRecord hit = hits[tid];
	int path_idx = hit.path_idx;
	GpuPathState &ps = paths[path_idx];

	if (ps.depth + 1 >= max_bounces) {
		ps.set_active(false);

		return;
	}
	bool volume = hit.mat_id < 0;
	float3 wi;
	float sample_pdf;
	bool sample_delta = false;

	if (volume) {
		float u1 = ps.rng.next_float(), u2 = ps.rng.next_float();
		wi = gpu_sample_henyey_greenstein(ps.ray.dir, hit.u, u1, u2);
		sample_pdf = gpu_henyey_greenstein(dot3(ps.ray.dir, wi), hit.u);
	} else {
		GpuMaterial mat = gpu_evaluate_material(scene, hit);
		float3 wo = -ps.ray.dir, ns = gpu_shading_normal(hit, wo, mat.isTransmissive());
		GpuOnb onb(ns);
		float3 wo_local = onb.to_local(wo);
		GpuBSDFSample sample;

		if (!gpu_bsdf_sample(wo_local, mat, ps.rng, sample)) {
			ps.set_active(false);

			return;
		}
		wi = normalize3(onb.to_world(sample.wi));

		if (!gpu_valid_scattering(wo, wi, ns, hit.geo_normal)) {
			ps.set_active(false);

			return;
		}
		ps.throughput *= sample.f * (fabsf(sample.wi.z) / sample.pdf);
		ps.eta_scale *= sample.eta * sample.eta;
		sample_pdf = sample.pdf;
		sample_delta = sample.delta;

		if (mat.isTransmissive() && dot3(wo, hit.geo_normal) * dot3(wi, hit.geo_normal) < 0) {
			if (hit.front_face) {
				if ((max_component3(mat.absorption) > 0 || ps.absorption.depth > 0) &&
					!ps.absorption.enter(hit.mat_id, mat.absorption)) {
					ps.radiance = make_f3(nanf(""));
					ps.set_active(false);

					return;
				}
			} else
				ps.absorption.leave(hit.mat_id);
		}
	}

	if (max_component3(ps.throughput) == 0.f) {
		ps.set_active(false);

		return;
	}

	if (ps.depth >= min_bounces) {
		float p = fminf(fmaxf(max_component3(ps.throughput) * ps.eta_scale, 0.05f), 0.95f);

		if (ps.rng.next_float() >= p) {
			ps.set_active(false);

			return;
		}
		ps.throughput /= p;
	}
	ps.prev_pos = hit.pos;
	ps.ray = GpuRay{volume ? hit.pos : gpu_offset_ray_origin(hit.pos, hit.geo_normal, wi), wi};
	ps.depth++;
	ps.prev_bsdf_pdf_sa = sample_pdf;
	ps.set_prev_delta(sample_delta);

	// Enqueue for next bounce
	int slot = atomicAdd(&next_counter[CNT_RAY_NEXT], 1);

	if (slot < scene.num_pixels) { // bounds check
		ray_queue_next[slot] = path_idx;
	}
}

// ─── Accumulate ──────────────────────────────────────────────────────────────

__global__ void accumulate_kernel(const GpuPathState *paths, float *accum, int num_pixels) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= num_pixels)
		return;

	const GpuPathState &ps = paths[idx];
	float3 r = ps.radiance;

	accum[idx * 3 + 0] += r.x;
	accum[idx * 3 + 1] += r.y;
	accum[idx * 3 + 2] += r.z;
}

// ─── Linear HDR output ───────────────────────────────────────────────────────
// Average once, then publish top-down radiance. DisplayTransform handles
// exposure, tone mapping and sRGB identically for preview and PNG output.

__global__ void output_kernel(const float *accum, float *output, int spp, int width, int height) {
	int num_pixels = width * height;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= num_pixels)
		return;

	int dst = (height - 1 - idx / width) * width + idx % width;
	float inv_spp = 1.f / (float)spp;
	output[dst * 3 + 0] = accum[idx * 3 + 0] * inv_spp;
	output[dst * 3 + 1] = accum[idx * 3 + 1] * inv_spp;
	output[dst * 3 + 2] = accum[idx * 3 + 2] * inv_spp;
}

// ═════════════════════════════════════════════════════════════════════════════
// CudaRenderer Implementation
// ═════════════════════════════════════════════════════════════════════════════

CudaRenderer::CudaRenderer(int width, int height, int max_bounces, int min_bounces, bool use_optix)
	: width_(width), height_(height), max_bounces_(max_bounces), min_bounces_(min_bounces) {
	const char *host_counters = std::getenv("XENON_GPU_HOST_COUNTERS");
	device_queues_ = !(host_counters && std::strcmp(host_counters, "1") == 0);

	if (use_optix) {
#ifdef XENON_HAS_OPTIX
		optix_ = std::make_unique<OptixBackend>();
#else
		throw std::runtime_error("OptiX backend requested but this build has no OptiX support");
#endif
	}
	num_pixels_ = width_ * height_;
	alloc_buffers();
	cuda_print_device_info();
	std::printf("[CUDA] Queue counters: %s\n",
				device_queues_ ? "device" : "host (profiling control)");
}

CudaRenderer::~CudaRenderer() {
	free_buffers();
}

void CudaRenderer::alloc_buffers() {
	CUDA_CHECK(cudaMalloc(&d_paths_, num_pixels_ * sizeof(GpuPathState)));
	CUDA_CHECK(cudaMalloc(&d_ray_queue_, num_pixels_ * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_ray_next_, num_pixels_ * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_hits_, num_pixels_ * sizeof(GpuHitRecord)));
	CUDA_CHECK(cudaMalloc(&d_shadow_work_, 4 * num_pixels_ * sizeof(GpuShadowWork)));
	CUDA_CHECK(cudaMalloc(&d_shadow_queue_, num_pixels_ * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_counters_, NUM_COUNTERS * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_accum_, num_pixels_ * 3 * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_output_, num_pixels_ * 3 * sizeof(float)));

	// Clear accumulation buffer
	CUDA_CHECK(cudaMemset(d_accum_, 0, num_pixels_ * 3 * sizeof(float)));

	// Host output buffer
	h_output_ = new float[num_pixels_ * 3];

	std::printf("[CUDA] Allocated renderer buffers: %.2f MB device memory\n",
				(num_pixels_ * (sizeof(GpuPathState) + sizeof(int) * 3 + sizeof(GpuHitRecord) +
								4 * sizeof(GpuShadowWork) + sizeof(float) * 6) +
				 NUM_COUNTERS * sizeof(int)) /
					(1024.0 * 1024.0));
}

void CudaRenderer::free_buffers() {
	auto safe_free = [](auto *&p) {
		if (p) {
			cudaFree(p);
			p = nullptr;
		}
	};
	safe_free(d_paths_);
	safe_free(d_ray_queue_);
	safe_free(d_ray_next_);
	safe_free(d_hits_);
	safe_free(d_shadow_work_);
	safe_free(d_shadow_queue_);
	safe_free(d_counters_);
	safe_free(d_accum_);
	safe_free(d_output_);
	delete[] h_output_;
	h_output_ = nullptr;
}

void CudaRenderer::upload_scene(const Scene &scene) {
	gpu_scene_.upload(scene);
#ifdef XENON_HAS_OPTIX
	if (optix_)
		optix_->build(scene, gpu_scene_.get_scene_data());
#endif
}

double CudaRenderer::acceleration_build_seconds() const {
#ifdef XENON_HAS_OPTIX
	if (optix_)
		return optix_->build_seconds();
#endif
	return 0.0;
}

// ─── Render one sample ───────────────────────────────────────────────────────

void CudaRenderer::render_frame(const Scene &, const Camera &camera, TripleSwapchain &swapchain,
								bool publish_frame) {
	spp_++;

	GpuCamera gpu_cam = gpu_scene_.upload_camera(camera, width_, height_);
	GpuSceneData scene_data = gpu_scene_.get_scene_data();
	scene_data.num_pixels = num_pixels_;

	// Reset counters
	CUDA_CHECK(cudaMemsetAsync(d_counters_, 0, NUM_COUNTERS * sizeof(int)));

	// ── Ray generation ─────────────────────────────────────────────────────────
	raygen_kernel<<<grid_size(num_pixels_), kBlockSize>>>(d_paths_, d_ray_queue_, d_counters_,
														  gpu_cam, spp_, seed_, width_, height_);

	if (!device_queues_) {
		// ── Bounce loop ────────────────────────────────────────────────────────────
		for (int bounce = 0; bounce < max_bounces_; ++bounce) {
			// Read ray count from device
			int ray_count = 0;
			CUDA_CHECK(
				cudaMemcpy(&ray_count, &d_counters_[CNT_RAY], sizeof(int), cudaMemcpyDeviceToHost));

			if (ray_count == 0)
				break;

			// Reset hit and shadow counters
			int zeros[2] = {0, 0};
			CUDA_CHECK(
				cudaMemcpy(&d_counters_[CNT_HIT], zeros, 2 * sizeof(int), cudaMemcpyHostToDevice));

			// Intersect
#ifdef XENON_HAS_OPTIX
			if (optix_)
				optix_->intersect(scene_data, d_paths_, d_ray_queue_, d_counters_, d_hits_,
								  ray_count);
			else
#endif
				intersect_kernel<<<grid_size(ray_count), kBlockSize>>>(
					d_paths_, d_ray_queue_, d_counters_, d_hits_, d_counters_, scene_data);

			// Read hit count
			int hit_count = 0;
			CUDA_CHECK(
				cudaMemcpy(&hit_count, &d_counters_[CNT_HIT], sizeof(int), cudaMemcpyDeviceToHost));

			if (hit_count == 0)
				break;

			// Reset shadow counter
			int zero = 0;
			CUDA_CHECK(
				cudaMemcpy(&d_counters_[CNT_SHADOW], &zero, sizeof(int), cudaMemcpyHostToDevice));

			// NEE
			nee_kernel<<<grid_size(hit_count), kBlockSize>>>(d_paths_, d_hits_, d_counters_,
															 d_shadow_work_, d_counters_,
															 scene_data, max_bounces_);

			// Shadow rays
			int shadow_count = 0;
			CUDA_CHECK(cudaMemcpy(&shadow_count, &d_counters_[CNT_SHADOW], sizeof(int),
								  cudaMemcpyDeviceToHost));

			if (shadow_count > 0) {
#ifdef XENON_HAS_OPTIX
				if (optix_)
					optix_->shadow(scene_data, d_paths_, d_shadow_work_, d_counters_, shadow_count);
				else
#endif
					shadow_kernel<<<grid_size(shadow_count), kBlockSize>>>(d_paths_, d_shadow_work_,
																		   d_counters_, scene_data);
			}

			// Reset next ray counter
			CUDA_CHECK(
				cudaMemcpy(&d_counters_[CNT_RAY_NEXT], &zero, sizeof(int), cudaMemcpyHostToDevice));

			// Shade (all materials in one kernel)
			shade_kernel<<<grid_size(hit_count), kBlockSize>>>(d_paths_, d_hits_, d_counters_,
															   d_ray_next_, d_counters_, scene_data,
															   max_bounces_, min_bounces_);

			// Read next ray count, swap queues
			int next_ray_count = 0;
			CUDA_CHECK(cudaMemcpy(&next_ray_count, &d_counters_[CNT_RAY_NEXT], sizeof(int),
								  cudaMemcpyDeviceToHost));

			// Swap ray queues
			int *tmp = d_ray_queue_;
			d_ray_queue_ = d_ray_next_;
			d_ray_next_ = tmp;

			// Set ray count for next bounce
			CUDA_CHECK(cudaMemcpy(&d_counters_[CNT_RAY], &next_ray_count, sizeof(int),
								  cudaMemcpyHostToDevice));
		}

	} else {
		// Queue sizes stay on the device. Fixed-capacity launches check the active
		// counters in their first instruction; the default stream orders queue
		// resets, traversal, shading and queue advancement without host readbacks.
		for (int bounce = 0; bounce < max_bounces_; ++bounce) {
			CUDA_CHECK(cudaMemsetAsync(d_counters_ + CNT_RAY_NEXT, 0, 3 * sizeof(int)));
#ifdef XENON_HAS_OPTIX
			if (optix_)
				optix_->intersect(scene_data, d_paths_, d_ray_queue_, d_counters_, d_hits_,
								  num_pixels_);
			else
#endif
				intersect_kernel<<<grid_size(num_pixels_), kBlockSize>>>(
					d_paths_, d_ray_queue_, d_counters_, d_hits_, d_counters_, scene_data);

			nee_kernel<<<grid_size(num_pixels_), kBlockSize>>>(d_paths_, d_hits_, d_counters_,
															   d_shadow_work_, d_counters_,
															   scene_data, max_bounces_);
#ifdef XENON_HAS_OPTIX
			if (optix_)
				optix_->shadow(scene_data, d_paths_, d_shadow_work_, d_counters_, 4 * num_pixels_);
			else
#endif
				shadow_kernel<<<grid_size(4 * num_pixels_), kBlockSize>>>(d_paths_, d_shadow_work_,
																		  d_counters_, scene_data);

			shade_kernel<<<grid_size(num_pixels_), kBlockSize>>>(
				d_paths_, d_hits_, d_counters_, d_ray_next_, d_counters_, scene_data, max_bounces_,
				min_bounces_);
			CUDA_CHECK(cudaMemcpyAsync(d_counters_ + CNT_RAY, d_counters_ + CNT_RAY_NEXT,
									   sizeof(int), cudaMemcpyDeviceToDevice));
			std::swap(d_ray_queue_, d_ray_next_);
		}
	}

	// ── Accumulate ─────────────────────────────────────────────────────────────
	accumulate_kernel<<<grid_size(num_pixels_), kBlockSize>>>(d_paths_, d_accum_, num_pixels_);

	// Explicit sample boundary: callers can measure rendering independently of
	// readback, and cancellation remains responsive between samples.
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	if (publish_frame)
		publish(swapchain);
}

void CudaRenderer::publish(TripleSwapchain &swapchain) {
	if (spp_ == 0)
		return;
	auto start = std::chrono::steady_clock::now();

	// ── Linear HDR output ──────────────────────────────────────────────────────
	output_kernel<<<grid_size(num_pixels_), kBlockSize>>>(d_accum_, d_output_, spp_, width_,
														  height_);
	CUDA_CHECK(cudaDeviceSynchronize());

	// ── Copy back to host swapchain ────────────────────────────────────────────
	CUDA_CHECK(
		cudaMemcpy(h_output_, d_output_, num_pixels_ * 3 * sizeof(float), cudaMemcpyDeviceToHost));

	for (int i = 0; i < num_pixels_ * 3; ++i)
		if (!std::isfinite(h_output_[i]))
			throw std::runtime_error("Non-finite CUDA radiance");
	float *write_buf = swapchain.get_write_buffer();
	std::memcpy(write_buf, h_output_, num_pixels_ * 3 * sizeof(float));
	swapchain.swap_writer();
	readback_seconds_ +=
		std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

} // namespace xn
