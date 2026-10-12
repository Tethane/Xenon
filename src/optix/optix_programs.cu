#include <optix_device.h>
#include "optix/optix_params.h"
#include "cuda/gpu_intersection.cuh"

using namespace xn;
extern "C" {
__constant__ OptixParams params;
}

static __forceinline__ __device__ GpuHitRecord *payload_record() {
	return reinterpret_cast<GpuHitRecord *>((uint64_t(optixGetPayload_0()) << 32) |
											optixGetPayload_1());
}

static __forceinline__ __device__ bool primitive_hit(const GpuPrimitive &p, const GpuRay &ray,
													 GpuHitRecord &rec) {
	switch (p.type) {
		case 0:
			return gpu_sphere_intersect(p, ray, rec);
		case 1:
			return gpu_plane_intersect(p, ray, rec);
		case 2:
			return gpu_box_intersect(p, ray, rec);
		case 3:
			return gpu_disk_intersect(p, ray, rec);
		case 4:
			return gpu_quad_intersect(p, ray, rec);
	}

	return false;
}

// The built-in triangle intersector gives barycentrics; reconstruct the shared
// surface record without running a second software triangle intersection.
static __forceinline__ __device__ GpuHitRecord surface_record() {
	const auto &scene = params.scene;
	const auto &inst = scene.instances[optixGetInstanceId()];
	int primitive = optixGetPrimitiveIndex();
	GpuHitRecord rec{};
	GpuRay local{inst.world_to_local.transform_point(optixGetWorldRayOrigin()),
				 inst.world_to_local.transform_dir(optixGetWorldRayDirection()), optixGetRayTmin(),
				 kGpuInfinity};

	if (inst.mesh_id >= 0) {
		const auto &mi = scene.mesh_info[inst.mesh_id];
		int base = mi.tri_offset + primitive * 3;
		int i0 = mi.vertex_offset + scene.indices[base];
		int i1 = mi.vertex_offset + scene.indices[base + 1];
		int i2 = mi.vertex_offset + scene.indices[base + 2];
		float3 p0 = make_f3(scene.vertices_x[i0], scene.vertices_y[i0], scene.vertices_z[i0]);
		float3 p1 = make_f3(scene.vertices_x[i1], scene.vertices_y[i1], scene.vertices_z[i1]);
		float3 p2 = make_f3(scene.vertices_x[i2], scene.vertices_y[i2], scene.vertices_z[i2]);
		rec.t = optixGetRayTmax();
		rec.geo_normal = normalize3(cross3(p1 - p0, p2 - p0));
		rec.mat_id = scene.mat_ids[mi.mat_ids_offset + primitive];
		float2 uv = optixGetTriangleBarycentrics();
		rec.u = uv.x;
		rec.v = uv.y;
		// This shared helper also preserves UV seams and tangent handedness.
		gpu_mesh_surface(scene, inst.mesh_id, primitive, uv.x, uv.y, rec);
	} else {
		local.tmin = __uint_as_float(optixGetAttribute_0());
		primitive_hit(scene.primitives[inst.prim_offset + primitive], local, rec);
	}
	rec.prim_id = primitive;
	GpuRay world{optixGetWorldRayOrigin(), optixGetWorldRayDirection(), optixGetRayTmin(),
				 kGpuInfinity};
	gpu_tlas_fixup_hit(rec, world, inst);

	return rec;
}

extern "C" __global__ void __intersection__analytic() {
	const auto &inst = params.scene.instances[optixGetInstanceId()];
	const auto &p = params.scene.primitives[inst.prim_offset + optixGetPrimitiveIndex()];

	if (p.type == 1)
		return; // truly infinite planes are queried separately
	GpuRay local{optixGetObjectRayOrigin(), optixGetObjectRayDirection(), optixGetRayTmin(),
				 optixGetRayTmax()};
	GpuHitRecord rec{};
	// A rejected opacity cutout on a closed primitive can expose its back face.
	// Report both roots without replacing the shared analytic intersection math.
	for (int root = 0; root < 2; ++root) {
		rec = GpuHitRecord{};

		if (!primitive_hit(p, local, rec))
			break;

		if (optixReportIntersection(rec.t, 0, __float_as_uint(local.tmin)))
			break;
		local.tmin = nextafterf(rec.t, kGpuInfinity);
	}
}

extern "C" __global__ void __anyhit__surface() {
	// Match Xenon's half-open [tmin,tmax) shadow segment convention.
	if (optixGetRayTmax() >= payload_record()->t) {
		optixIgnoreIntersection();

		return;
	}
	const auto &scene = params.scene;
	const auto &inst = scene.instances[optixGetInstanceId()];
	int primitive = optixGetPrimitiveIndex();
	int mat_id = inst.mesh_id >= 0
					 ? scene.mat_ids[scene.mesh_info[inst.mesh_id].mat_ids_offset + primitive]
					 : scene.primitives[inst.prim_offset + primitive].mat_id;
	const auto &mat = scene.materials[mat_id];

	if (mat.opacityMode == 0 && mat.opacityTexture.texture < 0) {
		if (mat.opacity < mat.alphaCutoff)
			optixIgnoreIntersection();

		return;
	}
	GpuHitRecord rec = surface_record();

	if (!gpu_surface_opaque(scene, rec, optixGetPayload_2()))
		optixIgnoreIntersection();
}

extern "C" __global__ void __closesthit__surface() {
	*payload_record() = surface_record();
}

extern "C" __global__ void __miss__empty() {}

static __forceinline__ __device__ bool trace_ray(const GpuRay &ray, GpuHitRecord &rec,
												 bool shadow) {
	rec.t = ray.tmax;
	rec.mat_id = -1;

	if (params.handle) {
		uint64_t ptr = reinterpret_cast<uint64_t>(&rec);
		uint32_t hi = ptr >> 32, lo = ptr, seed = ray.alpha_seed;
		optixTrace(params.handle, ray.origin, ray.dir, ray.tmin, ray.tmax, 0.f, 255,
				   shadow ? OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT : OPTIX_RAY_FLAG_NONE, 0, 1, 0,
				   hi, lo, seed);
	}
	// Unbounded planes cannot be represented by a finite OptiX AABB. Evaluate
	// only the explicit plane list, so bounded primitives retain accelerated AS.
	for (int i = 0; i < params.num_planes && !(shadow && rec.mat_id >= 0); ++i) {
		auto ref = params.planes[i];
		const auto &inst = params.scene.instances[ref.instance];
		GpuRay local{inst.world_to_local.transform_point(ray.origin),
					 inst.world_to_local.transform_dir(ray.dir), ray.tmin, fminf(ray.tmax, rec.t)};
		GpuHitRecord candidate{};

		if (gpu_plane_intersect(params.scene.primitives[inst.prim_offset + ref.primitive], local,
								candidate)) {
			candidate.prim_id = ref.primitive;
			gpu_tlas_fixup_hit(candidate, ray, inst);

			if (gpu_surface_opaque(params.scene, candidate, ray.alpha_seed))
				rec = candidate;
		}
	}

	if (rec.mat_id < 0) {
		rec = GpuHitRecord{};

		return false;
	}

	return true;
}

extern "C" __global__ void __raygen__queries() {
	int tid = optixGetLaunchIndex().x;

	if (params.mode == 2) {
		GpuHitRecord rec{};
		trace_ray(params.rays[tid], rec, false);
		params.hits[tid] = rec;
		GpuHitRecord shadow{};
		params.occluded[tid] = trace_ray(params.rays[tid], shadow, true);

		return;
	}

	if (params.mode == 1) {
		if (tid >= params.counters[3])
			return;
		const auto &sw = params.shadows[tid];
		GpuHitRecord rec{};

		if (!trace_ray(sw.ray, rec, true)) {
			atomicAdd(&params.paths[sw.path_idx].radiance.x, sw.contrib.x);
			atomicAdd(&params.paths[sw.path_idx].radiance.y, sw.contrib.y);
			atomicAdd(&params.paths[sw.path_idx].radiance.z, sw.contrib.z);
		}

		return;
	}

	if (tid >= params.counters[0])
		return;
	int path_idx = params.queue[tid];
	auto &ps = params.paths[path_idx];

	if (!ps.is_active())
		return;
	gpu_prepare_intersection(ps, params.scene);
	GpuHitRecord rec{};
	bool found = trace_ray(ps.ray, rec, false);
	gpu_complete_intersection(params.paths, path_idx, rec, found, params.hits, params.counters,
							  params.scene);
}
