#pragma once
// cuda/gpu_traverse.cuh — Device-side BVH traversal (TLAS → BLAS)
//
// Scalar stack-based DFS. The GPU gets throughput from warp-level parallelism,
// not from per-ray SIMD (unlike the CPU backend's SSE pair test).

#include "cuda/gpu_types.cuh"
#include "cuda/gpu_medium.cuh"
#include "material/opacity.h"

namespace xn {

// ─── Device scene pointer bundle ─────────────────────────────────────────────
// Passed to all traversal / shading functions by const reference.

struct GpuSceneData {
	// Concatenated mesh data
	const float *vertices_x;
	const float *vertices_y;
	const float *vertices_z;
	const float *texcoords_u = nullptr;
	const float *texcoords_v = nullptr;
	const GpuTexture *textures = nullptr;
	const GpuTextureLevel *texture_levels = nullptr;
	const float4 *texture_pixels = nullptr;
	const float *normals_x;
	const float *normals_y;
	const float *normals_z;
	const int32_t *indices;
	const int32_t *mat_ids;

	// BVH
	const GpuBVHNode *blas_nodes;
	const uint32_t *blas_prims;
	const GpuBVHNode *tlas_nodes;
	const uint32_t *tlas_inst_indices;

	// Per-mesh info
	const GpuMeshInfo *mesh_info;

	// Instances
	const GpuInstance *instances;
	const GpuPrimitive *primitives;

	// Materials
	const GpuMaterial *materials;

	// Lights
	const GpuLight *lights;
	const GpuDirLight *dir_lights;
	const GpuPointLight *point_lights = nullptr;
	int num_point_lights = 0;
	GpuEnvironment env;

	const GpuMedium *media = nullptr;
	int num_media = 0;
	bool has_alpha_blend = false;

	// Counts
	int num_meshes;
	int num_materials;
	int num_lights;
	int num_dir_lights;
	int num_instances;
	int num_tlas_nodes;
	int num_pixels; // for bounds-checking atomicAdd in kernels
};

__device__ inline float4 gpu_texture_sample(const GpuSceneData &scene, GpuTextureMap map, float u,
											float v, float lod = 0.f) {
	if (map.texture < 0)
		return make_float4(1, 1, 1, 1);
	const auto &t = scene.textures[map.texture];
	lod = fminf(fmaxf(lod, 0.f), float(t.num_levels - 1));
	int l0 = int(lod), l1 = l0 + 1 < t.num_levels ? l0 + 1 : t.num_levels - 1;
	auto blend = [](float4 a, float4 b, float f) {
		return make_float4(a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f, a.z + (b.z - a.z) * f,
						   a.w + (b.w - a.w) * f);
	};
	auto bilinear = [&](int l) {
		const auto &level = scene.texture_levels[t.first_level + l];
		float x = (u - floorf(u)) * level.width - .5f,
			  y = (1.f - (v - floorf(v))) * level.height - .5f;
		int ix = int(floorf(x)), iy = int(floorf(y));
		float fx = x - ix, fy = y - iy;
		auto texel = [&](int px, int py) {
			px = (px % level.width + level.width) % level.width;
			py = (py % level.height + level.height) % level.height;

			return scene.texture_pixels[level.offset + py * level.width + px];
		};

		return blend(blend(texel(ix, iy), texel(ix + 1, iy), fx),
					 blend(texel(ix, iy + 1), texel(ix + 1, iy + 1), fx), fy);
	};

	return blend(bilinear(l0), bilinear(l1), lod - l0);
}

__device__ inline float3 gpu_texture_rgb(const GpuSceneData &scene, GpuTextureMap map, float u,
										 float v, float lod = 0.f) {
	float4 p = gpu_texture_sample(scene, map, u, v, lod);

	return make_f3(p.x, p.y, p.z);
}

__device__ inline float gpu_texture_scalar(const GpuSceneData &scene, GpuTextureMap map, float u,
										   float v, float lod = 0.f) {
	float4 p = gpu_texture_sample(scene, map, u, v, lod);

	return map.channel == 1 ? p.y : map.channel == 2 ? p.z : map.channel == 3 ? p.w : p.x;
}

__device__ inline float gpu_surface_opacity(const GpuSceneData &scene, const GpuHitRecord &hit) {
	const auto &m = scene.materials[hit.mat_id];

	return fminf(fmaxf(m.opacity * gpu_texture_scalar(scene, m.opacityTexture, hit.tex_u, hit.tex_v,
													  m.textureLod),
					   0.f),
				 1.f);
}

__device__ inline bool gpu_surface_opaque(const GpuSceneData &scene, const GpuHitRecord &hit,
										  uint32_t seed = 0) {
	if (!scene.materials || hit.mat_id < 0 || hit.mat_id >= scene.num_materials)
		return true;
	const auto &m = scene.materials[hit.mat_id];
	float a = gpu_surface_opacity(scene, hit);

	return m.opacityMode == 1
			   ? opacity_random(seed, hit.instance_id, hit.prim_id, hit.front_face) < a
			   : a >= m.alphaCutoff;
}

__device__ inline GpuMaterial gpu_evaluate_material(const GpuSceneData &scene, GpuHitRecord &hit) {
	GpuMaterial m = scene.materials[hit.mat_id];

	if (m.baseColorTexture.texture < 0 && m.roughnessTexture.texture < 0 &&
		m.metallicTexture.texture < 0 && m.emissionTexture.texture < 0 &&
		m.normalTexture.texture < 0 && m.bumpTexture.texture < 0)
		return m;
	float u = hit.tex_u, v = hit.tex_v, lod = m.textureLod;
	m.baseColor =
		clamp3f(m.baseColor * gpu_texture_rgb(scene, m.baseColorTexture, u, v, lod), 0.f, 1.f);
	m.roughness = fminf(
		fmaxf(m.roughness * gpu_texture_scalar(scene, m.roughnessTexture, u, v, lod), 0.f), 1.f);
	m.metallic = fminf(
		fmaxf(m.metallic * gpu_texture_scalar(scene, m.metallicTexture, u, v, lod), 0.f), 1.f);
	m.emission = m.emission * gpu_texture_rgb(scene, m.emissionTexture, u, v, lod);
	m.alpha = fmaxf(m.roughness * m.roughness, 1e-6f);
	float aspect = sqrtf(1.f - .9f * fabsf(m.anisotropy));
	m.alpha_x = fmaxf(m.alpha / aspect, 1e-6f);
	m.alpha_y = fmaxf(m.alpha * aspect, 1e-6f);

	if (m.anisotropy < 0.f) {
		float a = m.alpha_x;
		m.alpha_x = m.alpha_y;
		m.alpha_y = a;
	}
	float f0 = ((m.ior - 1.f) * (m.ior - 1.f)) / ((m.ior + 1.f) * (m.ior + 1.f));
	m.F0 = lerp3(make_f3(f0), m.baseColor, m.metallic);
	m.flags = 0;

	if (m.metallic > .5f)
		m.flags |= GpuMaterial::FLAG_CONDUCTOR;

	if (m.transmission > 0.f && m.metallic < 1.f)
		m.flags |= GpuMaterial::FLAG_TRANSMISSIVE;

	if (m.subsurface > 0.f)
		m.flags |= GpuMaterial::FLAG_SUBSURFACE;

	if ((fmaxf(m.alpha_x, m.alpha_y) < .001f || (m.ior == 1.f && m.metallic == 0.f)) &&
		(m.metallic == 1.f || m.transmission == 1.f) &&
		(m.clearcoat == 0.f || m.clearcoatIor == 1.f ||
		 m.clearcoatRoughness * m.clearcoatRoughness < .001f))
		m.flags |= GpuMaterial::FLAG_DELTA;

	if (max_component3(m.emission) > 0.f)
		m.flags |= GpuMaterial::FLAG_EMISSIVE;
	float3 n = hit.normal, tangent = hit.dpdu - n * dot3(n, hit.dpdu);

	if (length_sq3(tangent) > 1e-20f) {
		tangent = normalize3(tangent);
		float handedness = dot3(cross3(n, tangent), hit.dpdv) < 0.f ? -1.f : 1.f;
		float3 bitangent = cross3(n, tangent) * handedness;

		if (m.normalTexture.texture >= 0) {
			float3 t = gpu_texture_rgb(scene, m.normalTexture, u, v, lod) * 2.f - make_f3(1.f);
			t.x *= m.normalScale;
			t.y *= m.normalScale;
			n = normalize3(tangent * t.x + bitangent * t.y + n * t.z);
		}

		if (m.bumpTexture.texture >= 0) {
			const auto &level =
				scene.texture_levels[scene.textures[m.bumpTexture.texture].first_level];
			float du = 1.f / level.width, dv = 1.f / level.height;
			float dhdu = (gpu_texture_scalar(scene, m.bumpTexture, u + du, v, lod) -
						  gpu_texture_scalar(scene, m.bumpTexture, u - du, v, lod)) /
						 (2.f * du);
			float dhdv = (gpu_texture_scalar(scene, m.bumpTexture, u, v + dv, lod) -
						  gpu_texture_scalar(scene, m.bumpTexture, u, v - dv, lod)) /
						 (2.f * dv);
			float3 pu = hit.dpdu - n * dot3(n, hit.dpdu), pv = hit.dpdv - n * dot3(n, hit.dpdv);
			float a = dot3(pu, pu), b = dot3(pu, pv), c = dot3(pv, pv), det = a * c - b * b;

			if (det > 1e-20f)
				n = normalize3(n - m.bumpScale * (pu * ((c * dhdu - b * dhdv) / det) +
												  pv * ((a * dhdv - b * dhdu) / det)));
		}

		if (length_sq3(n) > 0.f && dot3(n, hit.geo_normal) > 0.f)
			hit.normal = n;
	}

	return m;
}

__device__ inline void gpu_mesh_surface(const GpuSceneData &scene, int mesh_id, int tri_idx,
										float u, float v, GpuHitRecord &rec) {
	const auto &mi = scene.mesh_info[mesh_id];
	int base = mi.tri_offset + tri_idx * 3, vo = mi.vertex_offset;
	int a = vo + scene.indices[base], b = vo + scene.indices[base + 1],
		c = vo + scene.indices[base + 2];
	auto point = [&](int i) {
		return make_f3(scene.vertices_x[i], scene.vertices_y[i], scene.vertices_z[i]);
	};
	float3 e1 = point(b) - point(a), e2 = point(c) - point(a), gn = normalize3(cross3(e1, e2));

	if (mi.has_normals) {
		auto normal = [&](int i) {
			return make_f3(scene.normals_x[i], scene.normals_y[i], scene.normals_z[i]);
		};
		rec.normal = normalize3(normal(a) * (1 - u - v) + normal(b) * u + normal(c) * v);
	} else
		rec.normal = gn;

	if (length_sq3(rec.normal) == 0.f)
		rec.normal = gn;

	if (dot3(rec.normal, gn) < 0.f)
		rec.normal = -rec.normal;
	rec.has_uv = mi.has_uv;
	rec.tex_u = rec.tex_v = 0;
	rec.dpdu = rec.dpdv = make_f3(0.f);

	if (mi.has_uv) {
		const float *tu = scene.texcoords_u;
		const float *tv = scene.texcoords_v;
		rec.tex_u = tu[a] * (1 - u - v) + tu[b] * u + tu[c] * v;
		rec.tex_v = tv[a] * (1 - u - v) + tv[b] * u + tv[c] * v;
		float du1 = tu[b] - tu[a], dv1 = tv[b] - tv[a], du2 = tu[c] - tu[a], dv2 = tv[c] - tv[a],
			  det = du1 * dv2 - dv1 * du2;

		if (fabsf(det) > 1e-12f) {
			rec.dpdu = (e1 * dv2 - e2 * dv1) / det;
			rec.dpdv = (e2 * du1 - e1 * du2) / det;
		}
	}
}

__device__ inline int gpu_light_for_hit(const GpuSceneData &scene, const GpuHitRecord &hit) {
	int lo = 0, hi = scene.num_lights;

	while (lo < hi) {
		int mid = lo + (hi - lo) / 2;
		const auto &l = scene.lights[mid];

		if (l.instance_id < hit.instance_id ||
			(l.instance_id == hit.instance_id && l.tri_idx < uint32_t(hit.prim_id)))
			lo = mid + 1;
		else
			hi = mid;
	}

	return lo < scene.num_lights && scene.lights[lo].instance_id == hit.instance_id &&
				   scene.lights[lo].tri_idx == uint32_t(hit.prim_id)
			   ? lo
			   : -1;
}

__device__ inline int gpu_sample_light(const GpuSceneData &scene, double u, float &pmf) {
	if (scene.num_lights == 0) {
		pmf = 0.f;

		return -1;
	}
	int lo = 0, hi = scene.num_lights - 1;

	while (lo < hi) {
		int mid = lo + (hi - lo) / 2;

		if (u < scene.lights[mid].selection_cdf)
			hi = mid;
		else
			lo = mid + 1;
	}
	pmf = scene.lights[lo].selection_pmf;

	return lo;
}

__device__ inline float3 gpu_light_emission(const GpuSceneData &scene, const GpuLight &light,
											float u, float v) {
	const auto &mi = scene.mesh_info[light.mesh_id];
	GpuHitRecord hit;
	hit.mat_id = scene.mat_ids[mi.mat_ids_offset + light.tri_idx];
	gpu_mesh_surface(scene, light.mesh_id, light.tri_idx, u, v, hit);
	const auto &m = scene.materials[hit.mat_id];
	float coverage = m.opacityMode == 1 ? gpu_surface_opacity(scene, hit)
										: (gpu_surface_opaque(scene, hit) ? 1.f : 0.f);

	if (!light.material_emission)
		return light.emission * coverage;

	return light.emission *
		   gpu_texture_rgb(scene, m.emissionTexture, hit.tex_u, hit.tex_v, m.textureLod) * coverage;
}

// ─── Möller–Trumbore triangle intersection ───────────────────────────────────

__device__ inline bool gpu_ray_triangle(const GpuRay &ray, float3 v0, float3 v1, float3 v2,
										float &t_out, float &u_out, float &v_out, float t_max) {
	auto triple = [](float3 a, float3 b, float3 c) -> double {
		return double(a.x) * (double(b.y) * c.z - double(b.z) * c.y) +
			   double(a.y) * (double(b.z) * c.x - double(b.x) * c.z) +
			   double(a.z) * (double(b.x) * c.y - double(b.y) * c.x);
	};
	float3 e1 = v1 - v0, e2 = v2 - v0, delta = ray.origin - v0;
	double det = triple(e1, ray.dir, e2);

	if (det == 0.0)
		return false;
	double bu = triple(delta, ray.dir, e2) / det;

	if (bu < 0.0 || bu > 1.0)
		return false;
	double bv = triple(ray.dir, delta, e1) / det;

	if (bv < 0.0 || bu + bv > 1.0)
		return false;
	double distance = triple(e2, delta, e1) / det;

	if (distance < ray.tmin || distance >= fminf(t_max, ray.tmax))
		return false;
	t_out = (float)distance;
	u_out = (float)bu;
	v_out = (float)bv;

	return true;
}

// ─── Triangle intersection with full HitRecord population ────────────────────

__device__ inline bool gpu_mesh_intersect_triangle(const GpuRay &ray, int tri_idx,
												   const GpuSceneData &scene, int mesh_id,
												   GpuHitRecord &rec, int instance_id = -1) {
	const GpuMeshInfo &mi = scene.mesh_info[mesh_id];
	int base = mi.tri_offset + tri_idx * 3;
	int i0 = scene.indices[base + 0];
	int i1 = scene.indices[base + 1];
	int i2 = scene.indices[base + 2];

	int vo = mi.vertex_offset;
	float3 p0 =
		make_f3(scene.vertices_x[vo + i0], scene.vertices_y[vo + i0], scene.vertices_z[vo + i0]);
	float3 p1 =
		make_f3(scene.vertices_x[vo + i1], scene.vertices_y[vo + i1], scene.vertices_z[vo + i1]);
	float3 p2 =
		make_f3(scene.vertices_x[vo + i2], scene.vertices_y[vo + i2], scene.vertices_z[vo + i2]);

	float t, u, v;

	if (!gpu_ray_triangle(ray, p0, p1, p2, t, u, v, rec.t))
		return false;

	GpuHitRecord opacity_hit;
	opacity_hit.mat_id = scene.mat_ids[mi.mat_ids_offset + tri_idx];
	gpu_mesh_surface(scene, mesh_id, tri_idx, u, v, opacity_hit);
	opacity_hit.prim_id = tri_idx;
	opacity_hit.instance_id = instance_id;
	opacity_hit.front_face = dot3(cross3(p1 - p0, p2 - p0), ray.dir) < 0.f;

	if (!gpu_surface_opaque(scene, opacity_hit, ray.alpha_seed))
		return false;
	rec.normal = opacity_hit.normal;
	rec.tex_u = opacity_hit.tex_u;
	rec.tex_v = opacity_hit.tex_v;
	rec.dpdu = opacity_hit.dpdu;
	rec.dpdv = opacity_hit.dpdv;
	rec.has_uv = opacity_hit.has_uv;
	rec.t = t;
	rec.pos = ray.at(t);
	rec.u = u;
	rec.v = v;
	rec.mat_id = scene.mat_ids[mi.mat_ids_offset + tri_idx];
	rec.prim_id = tri_idx;

	float3 gn = normalize3(cross3(p1 - p0, p2 - p0));
	rec.geo_normal = gn;
	rec.front_face = dot3(gn, ray.dir) < 0.f;

	return true;
}

// ─── AABB slab test (scalar) ─────────────────────────────────────────────────

__device__ inline bool gpu_aabb_intersect(float3 mn, float3 mx, float3 inv_dir, float3 origin,
										  float tmin, float tmax, float &tn_out) {
	for (int i = 0; i < 3; ++i) {
		float inv = (&inv_dir.x)[i], org = (&origin.x)[i];
		float lo = (&mn.x)[i], hi = (&mx.x)[i];

		if (inv == kGpuInfinity || inv == -kGpuInfinity) {
			if (org < lo || org > hi)
				return false;
			continue;
		}
		float a = (lo - org) * inv, b = (hi - org) * inv;
		tmin = fmaxf(tmin, fminf(a, b));
		tmax = fminf(tmax, fmaxf(a, b));
	}
	tn_out = tmin;

	return tmin <= tmax;
}

// ─── BLAS traversal — closest hit ────────────────────────────────────────────

__device__ inline bool gpu_blas_intersect(const GpuRay &ray, const GpuSceneData &scene, int mesh_id,
										  GpuHitRecord &rec, int instance_id = -1) {
	const GpuMeshInfo &mi = scene.mesh_info[mesh_id];

	if (mi.num_triangles == 0)
		return false;
	int node_off = mi.blas_node_offset;
	int prim_off = mi.blas_prim_offset;

	float3 inv_dir = make_f3(1.f / ray.dir.x, 1.f / ray.dir.y, 1.f / ray.dir.z);

	uint32_t stack[64];
	int ptr = 0;
	stack[ptr++] = 0;
	bool hit = false;

	while (ptr > 0) {
		uint32_t idx = stack[--ptr];
		const GpuBVHNode &node = scene.blas_nodes[node_off + idx];

		if (node.is_leaf()) {
			for (uint32_t i = 0; i < node.count; ++i) {
				uint32_t tri = scene.blas_prims[prim_off + node.child + i];

				if (gpu_mesh_intersect_triangle(ray, tri, scene, mesh_id, rec, instance_id))
					hit = true;
			}
			continue;
		}

		uint32_t left = node.child;
		uint32_t right = left + 1;
		float tn_l, tn_r;
		bool hit_l = gpu_aabb_intersect(scene.blas_nodes[node_off + left].mn,
										scene.blas_nodes[node_off + left].mx, inv_dir, ray.origin,
										ray.tmin, rec.t, tn_l);
		bool hit_r = gpu_aabb_intersect(scene.blas_nodes[node_off + right].mn,
										scene.blas_nodes[node_off + right].mx, inv_dir, ray.origin,
										ray.tmin, rec.t, tn_r);

		if (hit_l && hit_r) {
			if (tn_l < tn_r) {
				stack[ptr++] = right;
				stack[ptr++] = left;
			} else {
				stack[ptr++] = left;
				stack[ptr++] = right;
			}
		} else if (hit_l) {
			stack[ptr++] = left;
		} else if (hit_r) {
			stack[ptr++] = right;
		}
	}

	return hit;
}

// ─── BLAS traversal — any hit (shadow) ───────────────────────────────────────

__device__ inline bool gpu_blas_intersects(const GpuRay &ray, const GpuSceneData &scene,
										   int mesh_id, int instance_id = -1) {
	const GpuMeshInfo &mi = scene.mesh_info[mesh_id];

	if (mi.num_triangles == 0)
		return false;
	int node_off = mi.blas_node_offset;
	int prim_off = mi.blas_prim_offset;

	float3 inv_dir = make_f3(1.f / ray.dir.x, 1.f / ray.dir.y, 1.f / ray.dir.z);

	uint32_t stack[64];
	int ptr = 0;
	stack[ptr++] = 0;

	while (ptr > 0) {
		uint32_t idx = stack[--ptr];
		const GpuBVHNode &node = scene.blas_nodes[node_off + idx];

		if (node.is_leaf()) {
			for (uint32_t i = 0; i < node.count; ++i) {
				uint32_t tri = scene.blas_prims[prim_off + node.child + i];
				GpuHitRecord candidate;
				candidate.t = ray.tmax;

				if (gpu_mesh_intersect_triangle(ray, tri, scene, mesh_id, candidate, instance_id))
					return true;
			}
			continue;
		}

		uint32_t left = node.child;
		uint32_t right = left + 1;
		float tn;
		bool hit_l = gpu_aabb_intersect(scene.blas_nodes[node_off + left].mn,
										scene.blas_nodes[node_off + left].mx, inv_dir, ray.origin,
										ray.tmin, ray.tmax, tn);
		bool hit_r = gpu_aabb_intersect(scene.blas_nodes[node_off + right].mn,
										scene.blas_nodes[node_off + right].mx, inv_dir, ray.origin,
										ray.tmin, ray.tmax, tn);

		if (hit_l)
			stack[ptr++] = left;

		if (hit_r)
			stack[ptr++] = right;
	}

	return false;
}

__device__ inline void gpu_swap(float &a, float &b) {
	float t = a;
	a = b;
	b = t;
}

__device__ inline bool gpu_sphere_intersect(const GpuPrimitive &prim, const GpuRay &ray,
											GpuHitRecord &rec) {
	auto center = prim.center;
	auto radius = prim.radius;
	auto mat_id = prim.mat_id;
	float3 oc = ray.origin - center;
	double a = double(ray.dir.x) * ray.dir.x + double(ray.dir.y) * ray.dir.y +
			   double(ray.dir.z) * ray.dir.z;

	if (a == 0.0)
		return false;
	double hb = double(oc.x) * ray.dir.x + double(oc.y) * ray.dir.y + double(oc.z) * ray.dir.z;
	double c =
		double(oc.x) * oc.x + double(oc.y) * oc.y + double(oc.z) * oc.z - double(radius) * radius;
	double disc = hb * hb - a * c;

	if (disc < 0.0)
		return false;
	double root = sqrt(disc);
	double q = -hb - (hb < 0.0 ? -root : root);
	double t0 = q / a, t1 = q == 0.0 ? t0 : c / q;

	if (t0 > t1) {
		double tmp = t0;
		t0 = t1;
		t1 = tmp;
	}
	double distance = t0;

	if (distance < ray.tmin)
		distance = t1;

	if (distance < ray.tmin || distance >= fminf(rec.t, ray.tmax))
		return false;
	float t = (float)distance;

	float3 outward = (ray.at(t) - center) / radius; // unit outward normal

	// Spherical UV: u = phi/(2pi), v = (pi-theta)/pi  (standard latitude-longitude)
	float phi = atan2f(-outward.z, outward.x) + kGpuPi;
	float theta = acosf(fminf(fmaxf(-outward.y, -1.f), 1.f));

	rec.t = t;
	rec.pos = ray.at(t);
	rec.geo_normal = outward;
	rec.normal = outward;
	rec.front_face = dot3(outward, ray.dir) < 0.f;
	rec.mat_id = mat_id;
	rec.u = phi * (0.5f * kGpuInvPi); // [0, 1]
	rec.v = theta * kGpuInvPi;		  // [0, 1]
	// prim_id set by caller
	return true;
}

__device__ inline bool gpu_plane_intersect(const GpuPrimitive &prim, const GpuRay &ray,
										   GpuHitRecord &rec) {
	auto normal = prim.normal;
	auto offset = prim.offset;
	auto mat_id = prim.mat_id;
	float denom = dot3(normal, ray.dir);

	if (denom == 0.f)
		return false;
	float t = (offset - dot3(normal, ray.origin)) / denom;

	if (t < ray.tmin || t >= fminf(rec.t, ray.tmax))
		return false;

	float3 p = ray.at(t);
	float3 n = normal;

	rec.t = t;
	rec.pos = p;
	rec.geo_normal = n;
	rec.normal = n;
	rec.front_face = denom < 0.f;
	rec.mat_id = mat_id;
	// Planar UV: arbitrary tiling in world XZ or XY depending on normal axis
	rec.u = p.x - floorf(p.x);
	rec.v = p.z - floorf(p.z);

	return true;
}

__device__ inline bool gpu_box_intersect(const GpuPrimitive &prim, const GpuRay &ray,
										 GpuHitRecord &rec) {
	auto center = prim.center;
	auto half = prim.half;
	auto mat_id = prim.mat_id;
	const float3 mn = center - half;
	const float3 mx = center + half;
	float near_t = -kGpuInfinity, far_t = kGpuInfinity;
	int near_axis = 0, far_axis = 0;
	float near_sign = -1.f, far_sign = 1.f;

	for (int i = 0; i < 3; ++i) {
		if ((&ray.dir.x)[i] == 0.f) {
			if ((&ray.origin.x)[i] < (&mn.x)[i] || (&ray.origin.x)[i] > (&mx.x)[i])
				return false;
			continue;
		}
		float t0 = ((&mn.x)[i] - (&ray.origin.x)[i]) / (&ray.dir.x)[i];
		float t1 = ((&mx.x)[i] - (&ray.origin.x)[i]) / (&ray.dir.x)[i];
		float sign = -1.f;

		if (t0 > t1) {
			gpu_swap(t0, t1);
			sign = 1.f;
		}

		if (t0 > near_t) {
			near_t = t0;
			near_axis = i;
			near_sign = sign;
		}

		if (t1 < far_t) {
			far_t = t1;
			far_axis = i;
			far_sign = -sign;
		}

		if (near_t > far_t)
			return false;
	}
	float t = near_t;

	if (t < ray.tmin) {
		t = far_t;
		near_axis = far_axis;
		near_sign = far_sign;
	}

	if (t < ray.tmin || t >= fminf(rec.t, ray.tmax))
		return false;
	rec.t = t;
	rec.pos = ray.at(t);
	rec.mat_id = mat_id;
	float3 n = make_f3(0.f);
	(&n.x)[near_axis] = near_sign;
	rec.geo_normal = n;
	rec.normal = n;
	rec.front_face = dot3(n, ray.dir) < 0.f;

	// UV: project onto face
	int u_axis = (near_axis + 1) % 3;
	int v_axis = (near_axis + 2) % 3;
	rec.u = ((&rec.pos.x)[u_axis] - (&mn.x)[u_axis]) / (2.f * (&half.x)[u_axis]);
	rec.v = ((&rec.pos.x)[v_axis] - (&mn.x)[v_axis]) / (2.f * (&half.x)[v_axis]);

	return true;
}

__device__ inline bool gpu_disk_intersect(const GpuPrimitive &prim, const GpuRay &ray,
										  GpuHitRecord &rec) {
	auto center = prim.center;
	auto normal = prim.normal;
	auto radius = prim.radius;
	auto mat_id = prim.mat_id;
	float denom = dot3(normal, ray.dir);

	if (denom == 0.f)
		return false;
	float t = dot3(center - ray.origin, normal) / denom;

	if (t < ray.tmin || t >= fminf(rec.t, ray.tmax))
		return false;

	float3 p = ray.at(t);
	float3 op = p - center;

	if (dot3(op, op) > radius * radius)
		return false; // outside disk

	float3 n = normal;
	rec.t = t;
	rec.pos = p;
	rec.geo_normal = n;
	rec.normal = n;
	rec.front_face = denom < 0.f;
	rec.mat_id = mat_id;

	// Polar UV: u = angle / (2π), v = radial distance / radius
	// Build a local 2D frame in the disk plane
	float3 ref = (fabsf(normal.x) < 0.9f) ? make_f3(1, 0, 0) : make_f3(0, 1, 0);
	float3 tang = normalize3(cross3(normal, ref));
	float3 bita = cross3(normal, tang);
	float px = dot3(op, tang);
	float py = dot3(op, bita);
	float phi = atan2f(py, px);

	if (phi < 0.f)
		phi += 2.f * kGpuPi;
	rec.u = phi * (0.5f * kGpuInvPi);
	rec.v = sqrtf(dot3(op, op)) / radius;

	return true;
}

__device__ inline bool gpu_quad_intersect(const GpuPrimitive &prim, const GpuRay &ray,
										  GpuHitRecord &rec) {
	auto origin = prim.origin;
	auto u_vec = prim.u_vec;
	auto v_vec = prim.v_vec;
	auto mat_id = prim.mat_id;
	float3 n_raw = cross3(u_vec, v_vec); // not necessarily unit length
	float area_sq = dot3(n_raw, n_raw);

	if (area_sq == 0.f)
		return false;

	float denom = dot3(n_raw, ray.dir);

	if (denom == 0.f)
		return false;
	float t = dot3(n_raw, origin - ray.origin) / denom;

	if (t < ray.tmin || t >= fminf(rec.t, ray.tmax))
		return false;

	// Project hit point onto quad's local frame to find barycentric (s, tv)
	float3 p = ray.at(t);
	float3 op = p - origin;

	// s = (p - origin) × v_vec · n̂ / |u × v|,  using triple products
	float s = dot3(cross3(op, v_vec), n_raw) / area_sq;
	float tv = dot3(cross3(u_vec, op), n_raw) / area_sq;

	if (s < 0.f || s > 1.f || tv < 0.f || tv > 1.f)
		return false;

	float3 n = normalize3(n_raw);

	rec.t = t;
	rec.pos = p;
	rec.geo_normal = n;
	rec.normal = n;
	rec.front_face = denom < 0.f;
	rec.mat_id = mat_id;
	rec.u = s;
	rec.v = tv;

	return true;
}

__device__ inline bool gpu_instance_intersect(const GpuRay &ray, const GpuSceneData &scene,
											  const GpuInstance &inst, GpuHitRecord &rec) {
	if (inst.mesh_id >= 0)
		return gpu_blas_intersect(ray, scene, inst.mesh_id, rec, inst.instance_id);
	bool hit = false;

	for (int i = 0; i < inst.prim_count; ++i) {
		const GpuPrimitive &p = scene.primitives[inst.prim_offset + i];
		GpuRay query = ray;
		query.tmax = fminf(ray.tmax, rec.t);

		for (int root = 0; root < 2; ++root) {
			GpuHitRecord candidate;
			candidate.t = query.tmax;
			bool found = false;

			switch (p.type) {
				case 0:
					found = gpu_sphere_intersect(p, query, candidate);
					break;
				case 1:
					found = gpu_plane_intersect(p, query, candidate);
					break;
				case 2:
					found = gpu_box_intersect(p, query, candidate);
					break;
				case 3:
					found = gpu_disk_intersect(p, query, candidate);
					break;
				case 4:
					found = gpu_quad_intersect(p, query, candidate);
					break;
			}

			if (!found)
				break;
			candidate.prim_id = i;
			candidate.instance_id = inst.instance_id;

			if (gpu_surface_opaque(scene, candidate, ray.alpha_seed)) {
				rec = candidate;
				hit = true;
				break;
			}
			query.tmin = nextafterf(candidate.t, kGpuInfinity);
		}
	}

	return hit;
}

// ─── TLAS traversal — closest hit ────────────────────────────────────────────

__device__ inline void gpu_tlas_fixup_hit(GpuHitRecord &rec, const GpuRay &world_ray,
										  const GpuInstance &inst) {
	rec.pos = world_ray.at(rec.t);
	rec.dpdu = inst.local_to_world.transform_dir(rec.dpdu);
	rec.dpdv = inst.local_to_world.transform_dir(rec.dpdv);
	rec.normal = normalize3(inst.world_to_local.transform_normal(rec.normal));
	rec.geo_normal = normalize3(inst.world_to_local.transform_normal(rec.geo_normal));
	rec.front_face = dot3(rec.geo_normal, world_ray.dir) < 0.f;

	if (dot3(rec.normal, rec.geo_normal) < 0.f)
		rec.normal = -rec.normal;
	rec.instance_id = inst.instance_id;
}

__device__ inline bool gpu_tlas_intersect(const GpuRay &world_ray, const GpuSceneData &scene,
										  GpuHitRecord &rec) {
	if (scene.num_tlas_nodes == 0)
		return false;

	float3 inv_dir = make_f3(1.f / world_ray.dir.x, 1.f / world_ray.dir.y, 1.f / world_ray.dir.z);

	uint32_t stack[64];
	int ptr = 0;
	stack[ptr++] = 0;
	bool any_hit = false;

	while (ptr > 0) {
		uint32_t idx = stack[--ptr];
		const GpuBVHNode &node = scene.tlas_nodes[idx];

		if (node.is_leaf()) {
			for (uint32_t i = 0; i < node.count; ++i) {
				uint32_t inst_idx = scene.tlas_inst_indices[node.child + i];
				const GpuInstance &inst = scene.instances[inst_idx];

				// Transform ray to local space
				GpuRay local_ray;
				local_ray.origin = inst.world_to_local.transform_point(world_ray.origin);
				local_ray.dir = inst.world_to_local.transform_dir(world_ray.dir);
				local_ray.tmin = world_ray.tmin;
				local_ray.alpha_seed = world_ray.alpha_seed;
				local_ray.tmax = fminf(rec.t, world_ray.tmax); // use current best t

				if (gpu_instance_intersect(local_ray, scene, inst, rec)) {
					gpu_tlas_fixup_hit(rec, world_ray, inst);
					any_hit = true;
				}
			}
			continue;
		}

		uint32_t left = node.child;
		uint32_t right = left + 1;
		float tn_l, tn_r;
		bool hit_l = gpu_aabb_intersect(scene.tlas_nodes[left].mn, scene.tlas_nodes[left].mx,
										inv_dir, world_ray.origin, world_ray.tmin, rec.t, tn_l);
		bool hit_r = gpu_aabb_intersect(scene.tlas_nodes[right].mn, scene.tlas_nodes[right].mx,
										inv_dir, world_ray.origin, world_ray.tmin, rec.t, tn_r);

		if (hit_l && hit_r) {
			if (tn_l < tn_r) {
				stack[ptr++] = right;
				stack[ptr++] = left;
			} else {
				stack[ptr++] = left;
				stack[ptr++] = right;
			}
		} else if (hit_l) {
			stack[ptr++] = left;
		} else if (hit_r) {
			stack[ptr++] = right;
		}
	}

	return any_hit;
}

// ─── TLAS traversal — any hit (shadow) ───────────────────────────────────────

__device__ inline bool gpu_tlas_intersects(const GpuRay &world_ray, const GpuSceneData &scene) {
	if (scene.num_tlas_nodes == 0)
		return false;

	float3 inv_dir = make_f3(1.f / world_ray.dir.x, 1.f / world_ray.dir.y, 1.f / world_ray.dir.z);

	uint32_t stack[64];
	int ptr = 0;
	stack[ptr++] = 0;

	while (ptr > 0) {
		uint32_t idx = stack[--ptr];
		const GpuBVHNode &node = scene.tlas_nodes[idx];

		if (node.is_leaf()) {
			for (uint32_t i = 0; i < node.count; ++i) {
				uint32_t inst_idx = scene.tlas_inst_indices[node.child + i];
				const GpuInstance &inst = scene.instances[inst_idx];

				GpuRay local_ray;
				local_ray.origin = inst.world_to_local.transform_point(world_ray.origin);
				local_ray.dir = inst.world_to_local.transform_dir(world_ray.dir);
				local_ray.tmin = world_ray.tmin;
				local_ray.alpha_seed = world_ray.alpha_seed;
				local_ray.tmax = world_ray.tmax;

				if (inst.mesh_id >= 0) {
					if (gpu_blas_intersects(local_ray, scene, inst.mesh_id, inst.instance_id))
						return true;
				} else {
					GpuHitRecord scratch;
					scratch.t = local_ray.tmax;

					if (gpu_instance_intersect(local_ray, scene, inst, scratch))
						return true;
				}
			}
			continue;
		}

		uint32_t left = node.child;
		uint32_t right = left + 1;
		float tn;
		bool hit_l =
			gpu_aabb_intersect(scene.tlas_nodes[left].mn, scene.tlas_nodes[left].mx, inv_dir,
							   world_ray.origin, world_ray.tmin, world_ray.tmax, tn);
		bool hit_r =
			gpu_aabb_intersect(scene.tlas_nodes[right].mn, scene.tlas_nodes[right].mx, inv_dir,
							   world_ray.origin, world_ray.tmin, world_ray.tmax, tn);

		if (hit_l)
			stack[ptr++] = left;

		if (hit_r)
			stack[ptr++] = right;
	}

	return false;
}

} // namespace xn
