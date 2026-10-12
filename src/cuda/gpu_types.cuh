#pragma once
// cuda/gpu_types.cuh — GPU-compatible type definitions for device code
//
// All types are plain structs usable in both __host__ and __device__ code.
// No std::vector, no virtual dispatch, no heap allocation.

#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include "volume/absorption.h"
#include "camera/path_seed.h"

namespace xn {

// ─── float3 math helpers ─────────────────────────────────────────────────────
// CUDA's built-in float3 lacks operators; add the essentials inline.

__host__ __device__ inline float3 make_f3(float x, float y, float z) {
	return make_float3(x, y, z);
}

__host__ __device__ inline float3 make_f3(float s) {
	return make_float3(s, s, s);
}

__host__ __device__ inline float3 operator+(float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(float3 a, float3 b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(float3 a, float3 b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline float3 operator*(float3 a, float s) {
	return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 operator*(float s, float3 a) {
	return a * s;
}

__host__ __device__ inline float3 operator/(float3 a, float s) {
	float r = 1.f / s;

	return a * r;
}

__host__ __device__ inline float3 operator-(float3 a) {
	return make_float3(-a.x, -a.y, -a.z);
}

__host__ __device__ inline float3 &operator+=(float3 &a, float3 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;

	return a;
}

__host__ __device__ inline float3 &operator*=(float3 &a, float s) {
	a.x *= s;
	a.y *= s;
	a.z *= s;

	return a;
}

__host__ __device__ inline float3 &operator*=(float3 &a, float3 b) {
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;

	return a;
}

__host__ __device__ inline float3 &operator/=(float3 &a, float s) {
	float r = 1.f / s;
	a.x *= r;
	a.y *= r;
	a.z *= r;

	return a;
}

__host__ __device__ inline float dot3(float3 a, float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 cross3(float3 a, float3 b) {
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__host__ __device__ inline float length_sq3(float3 v) {
	return dot3(v, v);
}

__host__ __device__ inline float length3(float3 v) {
	return sqrtf(length_sq3(v));
}

__host__ __device__ inline float3 normalize3(float3 v) {
	float len = length3(v);

	return len > 0.f ? v / len : make_f3(0.f);
}

__host__ __device__ inline float3 reflect3(float3 v, float3 n) {
	return v - 2.f * dot3(v, n) * n;
}

__host__ __device__ inline float3 faceforward3(float3 n, float3 ref) {
	return dot3(n, ref) < 0.f ? -n : n;
}

__host__ __device__ inline float3 lerp3(float3 a, float3 b, float t) {
	return a + t * (b - a);
}

__host__ __device__ inline float max_component3(float3 v) {
	return fmaxf(fmaxf(v.x, v.y), v.z);
}

__host__ __device__ inline float3 min3f(float3 a, float3 b) {
	return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__host__ __device__ inline float3 max3f(float3 a, float3 b) {
	return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__host__ __device__ inline float3 abs3f(float3 v) {
	return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}

__host__ __device__ inline float3 clamp3f(float3 v, float lo, float hi) {
	return make_float3(fminf(fmaxf(v.x, lo), hi), fminf(fmaxf(v.y, lo), hi),
					   fminf(fmaxf(v.z, lo), hi));
}

__host__ __device__ inline bool refract3(float3 v, float3 n, float ni_over_nt, float3 &out) {
	float3 uv = normalize3(v);
	float cos_i = -dot3(uv, n);
	float sin2_t = ni_over_nt * ni_over_nt * (1.f - cos_i * cos_i);

	if (sin2_t >= 1.f)
		return false;
	out = ni_over_nt * v + (ni_over_nt * cos_i - sqrtf(1.f - sin2_t)) * n;

	return true;
}

// ─── Constants ───────────────────────────────────────────────────────────────

constexpr float kGpuInfinity = std::numeric_limits<float>::infinity();
constexpr float kGpuRayEps = 1e-4f;
constexpr float kGpuShadowEps = 5e-4f;
constexpr float kGpuPi = 3.14159265358979323846f;
constexpr float kGpuInvPi = 1.f / kGpuPi;

// ─── GpuRay ──────────────────────────────────────────────────────────────────

struct GpuRay {
	float3 origin;
	float3 dir;
	float tmin = kGpuRayEps;
	float tmax = kGpuInfinity;
	uint32_t alpha_seed = 0;

	__host__ __device__ float3 at(float t) const {
		return origin + t * dir;
	}
};

// ─── GpuHitRecord ────────────────────────────────────────────────────────────

struct GpuHitRecord {
	float t = kGpuInfinity;
	float3 pos = {};
	float3 normal = {};
	float3 geo_normal = {};
	int mat_id = -1;
	int prim_id = -1;
	int instance_id = -1;
	int path_idx = -1;
	float u = 0, v = 0;
	float tex_u = 0, tex_v = 0;
	float3 dpdu{}, dpdv{};
	bool has_uv = false;
	bool front_face = true;

	__host__ __device__ bool valid() const {
		return t < kGpuInfinity;
	}
};

// ─── GpuPCG — device-side PCG random number generator ────────────────────────

struct GpuPCG {
	uint64_t state;
	uint64_t inc;

	__device__ uint32_t next_uint() {
		uint64_t oldstate = state;
		state = oldstate * 6364136223846793005ULL + (inc | 1);
		uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
		uint32_t rot = (uint32_t)(oldstate >> 59u);

		return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
	}

	__device__ float next_float() {
		return (next_uint() >> 8) * (1.f / 16777216.f);
	}
};

__host__ __device__ inline GpuPCG gpu_seed_pcg(uint64_t seed, uint64_t seq = 0) {
	GpuPCG s = {0, (seq << 1u) | 1u};
	// Advance twice to warm up the state
	uint64_t oldstate = s.state;
	s.state = oldstate * 6364136223846793005ULL + (s.inc | 1);
	s.state += seed;
	oldstate = s.state;
	s.state = oldstate * 6364136223846793005ULL + (s.inc | 1);

	return s;
}

__host__ __device__ inline GpuPCG gpu_seed_path_pcg(uint32_t pixel, uint32_t sample,
													uint64_t user_seed = 0) {
	const auto seed = make_path_seed(pixel, sample, user_seed);

	return gpu_seed_pcg(seed.state, seed.stream);
}

// ─── GpuMaterial ─────────────────────────────────────────────────────────────
// Flat POD copy of Material without std::string

struct GpuTexture {
	int first_level = 0, num_levels = 0;
};

struct GpuTextureLevel {
	int width = 0, height = 0, offset = 0;
};

struct GpuTextureMap {
	int texture = -1, channel = 0;
};

struct GpuMaterial {
	float3 baseColor;
	float roughness;
	float diffuseRoughness = 0.f;
	int multipleScattering = 0;
	float clearcoat = 0.f, clearcoatRoughness = .03f, clearcoatIor = 1.5f;
	float3 absorption{};
	float metallic;
	float ior;
	float transmission;
	float subsurface;
	float3 subsurfaceColor;
	float alpha;
	float alpha_x;
	float alpha_y;
	float3 F0;
	float3 emission;
	int emissionTwoSided = 0;
	float anisotropy = 0, opacity = 1, alphaCutoff = .5f, normalScale = 1, bumpScale = 1,
		  textureLod = 0;
	int opacityMode = 0; // 0 mask, 1 stochastic blend
	GpuTextureMap baseColorTexture, roughnessTexture, metallicTexture, emissionTexture;
	GpuTextureMap opacityTexture, normalTexture, bumpTexture;

	// Flags — packed into uint32_t to guarantee identical host/device layout.
	// bool fields cause unpredictable padding between host (CXX) and device (nvcc).
	uint32_t flags;

	static constexpr uint32_t FLAG_CONDUCTOR = 1u << 0;
	static constexpr uint32_t FLAG_TRANSMISSIVE = 1u << 1;
	static constexpr uint32_t FLAG_SUBSURFACE = 1u << 2;
	static constexpr uint32_t FLAG_DELTA = 1u << 3;
	static constexpr uint32_t FLAG_EMISSIVE = 1u << 4;

	__host__ __device__ bool isConductor() const {
		return flags & FLAG_CONDUCTOR;
	}

	__host__ __device__ bool isTransmissive() const {
		return flags & FLAG_TRANSMISSIVE;
	}

	__host__ __device__ bool hasSubsurface() const {
		return flags & FLAG_SUBSURFACE;
	}

	__host__ __device__ bool isDelta() const {
		return flags & FLAG_DELTA;
	}

	__host__ __device__ bool isEmissive() const {
		return flags & FLAG_EMISSIVE;
	}
};

// ─── GpuPathState ────────────────────────────────────────────────────────────

struct GpuPathState {
	AbsorptionStack<float3> absorption;
	GpuRay ray;
	float3 throughput;
	float3 radiance;
	GpuPCG rng;
	int pixel_idx;
	int depth;
	float prev_bsdf_pdf_sa;
	float3 prev_pos;
	float eta_scale;
	uint8_t flags; // bit 0: active, bit 1: specular, bit 2: prev_was_delta

	__device__ bool is_active() const {
		return flags & 1;
	}

	__device__ bool is_specular() const {
		return flags & 2;
	}

	__device__ bool was_prev_delta() const {
		return flags & 4;
	}

	__device__ void set_active(bool b) {
		if (b)
			flags |= 1;
		else
			flags &= ~1;
	}

	__device__ void set_specular(bool b) {
		if (b)
			flags |= 2;
		else
			flags &= ~2;
	}

	__device__ void set_prev_delta(bool b) {
		if (b)
			flags |= 4;
		else
			flags &= ~4;
	}
};

// ─── GpuLight ────────────────────────────────────────────────────────────────

struct GpuLight {
	int instance_id;
	float3 p0, p1, p2, normal;
	uint32_t mesh_id;
	uint32_t tri_idx;
	float3 emission;
	float area;
	int material_emission = 1;
	int emissionTwoSided = 0;
	float selection_pmf = 0.f;
	double selection_cdf = 0.;
};

struct GpuDirLight {
	float3 direction;
	float3 color;
	float intensity;

	__host__ __device__ float3 get_emission() const {
		return color * intensity;
	}
};

struct GpuPointLight {
	float3 position{}, intensity{}, direction{};
	float cos_inner = -1.f, cos_outer = -1.f;

	__device__ float falloff(float3 toward_surface) const {
		if (cos_outer < 0.f)
			return 1.f;
		float c = dot3(direction, toward_surface);

		if (c <= cos_outer)
			return 0.f;

		if (c >= cos_inner)
			return 1.f;
		float t = (c - cos_outer) / (cos_inner - cos_outer);

		return t * t * (3.f - 2.f * t);
	}
};

// ─── GpuEnvironment ──────────────────────────────────────────────────────────

__host__ __device__ inline int gpu_env_min(int a, int b) {
	return a < b ? a : b;
}

__host__ __device__ inline int gpu_env_max(int a, int b) {
	return a > b ? a : b;
}

struct GpuEnvironmentSample {
	float3 direction{}, radiance{};
	float pdf = 0;
};

struct GpuEnvironment {
	float3 zenith_color, horizon_color, ground_color;
	float horizon_sharpness, intensity;
	const float4 *pixels = nullptr;
	const double *cdf = nullptr;
	int width = 0, height = 0;
	float rotation = 0;

	__device__ float3 pixel(int x, int y) const {
		x = (x % width + width) % width;
		y = gpu_env_min(gpu_env_max(y, 0), height - 1);
		float4 p = pixels[y * width + x];

		return make_f3(p.x, p.y, p.z);
	}

	__device__ float3 evaluate(float3 d) const {
		if (pixels) {
			float u = (atan2f(d.z, d.x) + rotation) / (2 * kGpuPi) + .5f;
			u -= floorf(u);
			float v = atan2f(hypotf(d.x, d.z), d.y) / kGpuPi;
			float px = u * width - .5f, py = v * height - .5f;
			int x = int(floorf(px)), y = int(floorf(py));
			float fx = px - x, fy = py - y;

			return intensity * lerp3(lerp3(pixel(x, y), pixel(x + 1, y), fx),
									 lerp3(pixel(x, y + 1), pixel(x + 1, y + 1), fx), fy);
		}

		if (d.y < 0)
			return intensity * lerp3(horizon_color, ground_color, fminf(-d.y, 1.f));

		return intensity * lerp3(horizon_color, zenith_color, powf(d.y, 1.f / horizon_sharpness));
	}

	__device__ float pdf(float3 d) const {
		if (!pixels)
			return 1.f / (4 * kGpuPi);
		float u = (atan2f(d.z, d.x) + rotation) / (2 * kGpuPi) + .5f;
		u -= floorf(u);
		int x = gpu_env_min(width - 1, int(u * width));
		int y = gpu_env_min(height - 1, int(atan2f(hypotf(d.x, d.z), d.y) / kGpuPi * height));
		double area = (cos(double(kGpuPi) * y / height) - cos(double(kGpuPi) * (y + 1) / height)) *
					  2 * double(kGpuPi) / width;
		int i = y * width + x;

		return area > 0 ? float((cdf[i + 1] - cdf[i]) / area) : 0;
	}

	__device__ GpuEnvironmentSample sample(float u, float v) const {
		GpuEnvironmentSample result;

		if (pixels) {
			int lo = 0, hi = width * height;

			while (lo < hi) {
				int mid = (lo + hi) / 2;

				if (cdf[mid + 1] <= u)
					lo = mid + 1;
				else
					hi = mid;
			}
			int i = gpu_env_min(lo, width * height - 1), x = i % width, y = i / width;
			double probability = cdf[i + 1] - cdf[i];
			float residual =
				probability > 0 ? float(fmin(fmax((u - cdf[i]) / probability, 0.0), 1.0)) : .5f;
			float phi = ((x + residual) / width - .5f) * 2 * kGpuPi - rotation;
			double cos0 = cos(double(kGpuPi) * y / height),
				   cos1 = cos(double(kGpuPi) * (y + 1) / height);
			double cosine = cos0 + (cos1 - cos0) * v;
			float sine = float(sqrt(fmax(0.0, 1 - cosine * cosine)));
			result.direction = make_f3(sine * cosf(phi), float(cosine), sine * sinf(phi));
			double area = (cos0 - cos1) * 2 * double(kGpuPi) / width;
			result.pdf = area > 0 ? float(probability / area) : 0;
		} else {
			float cosine = 1 - 2 * u, sine = sqrtf(fmaxf(0.f, 1 - cosine * cosine)),
				  phi = 2 * kGpuPi * v;
			result.direction = make_f3(sine * cosf(phi), cosine, sine * sinf(phi));
			result.pdf = 1.f / (4 * kGpuPi);
		}
		result.radiance = evaluate(result.direction);

		return result;
	}

	__device__ bool active() const {
		return intensity > 0;
	}
};

// ─── GpuShadowWork ───────────────────────────────────────────────────────────

struct GpuShadowWork {
	int path_idx;
	GpuRay ray;
	float t_max;
	float3 contrib;
	int origin_prim_id;
	int light_prim_id;
};

// ─── GpuCamera ───────────────────────────────────────────────────────────────

struct GpuCamera {
	int medium_material = -1;
	float3 absorption{};
	float3 eye;
	float3 lower_left_corner;
	float3 horizontal;
	float3 vertical;
	float3 right{}, up{};
	float lens_radius = 0, focus_distance = 1;

	__device__ GpuRay get_ray(float s, float t) const {
		float3 target = lower_left_corner + s * horizontal + t * vertical;

		return GpuRay{eye, normalize3(target - eye)};
	}

	__device__ GpuRay get_ray(float s, float t, float lu, float lv) const {
		if (lens_radius == 0.f)
			return get_ray(s, t);
		float r = lens_radius * sqrtf(lu), angle = 2.f * kGpuPi * lv;
		float3 offset = right * (r * cosf(angle)) + up * (r * sinf(angle));
		float3 direction = lower_left_corner + s * horizontal + t * vertical - eye;

		return GpuRay{eye + offset, normalize3(focus_distance * direction - offset)};
	}
};

// ─── GpuMat4x3 — column-major 3×4 affine matrix ─────────────────────────────

struct GpuMat4x3 {
	float3 cols[4];

	__device__ float3 transform_point(float3 p) const {
		return cols[0] * p.x + cols[1] * p.y + cols[2] * p.z + cols[3];
	}

	__device__ float3 transform_dir(float3 d) const {
		return cols[0] * d.x + cols[1] * d.y + cols[2] * d.z;
	}

	__device__ float3 transform_normal(float3 n) const {
		return make_float3(cols[0].x * n.x + cols[0].y * n.y + cols[0].z * n.z,
						   cols[1].x * n.x + cols[1].y * n.y + cols[1].z * n.z,
						   cols[2].x * n.x + cols[2].y * n.y + cols[2].z * n.z);
	}
};

// ─── GpuInstance ─────────────────────────────────────────────────────────────

struct GpuPrimitive {
	float3 center{}, normal{}, half{}, origin{}, u_vec{}, v_vec{};
	float radius = 0.f, offset = 0.f;
	int type = 0, mat_id = 0;
};

struct GpuInstance {
	int prim_offset = 0, prim_count = 0;
	int mesh_id; // index into mesh_offsets
	int instance_id;
	GpuMat4x3 local_to_world;
	GpuMat4x3 world_to_local;
	float3 world_aabb_mn;
	float3 world_aabb_mx;
};

// ─── GpuBVHNode — 32 bytes, mirrors CPU BLASNode/TLASNode ────────────────────

struct GpuBVHNode {
	float3 mn;		// AABB min
	float3 mx;		// AABB max
	uint32_t child; // internal: left child index; leaf: first prim slot
	uint32_t count; // 0 → internal, >0 → leaf

	__device__ bool is_leaf() const {
		return count > 0;
	}
};

// ─── Mesh offset table ───────────────────────────────────────────────────────
// Per-mesh metadata for indexing into the concatenated arrays.

struct GpuMeshInfo {
	int vertex_offset;	  // into vertices_x/y/z
	int tri_offset;		  // into indices[] (element offset, not triangle offset)
	int mat_ids_offset;	  // into mat_ids[]
	int blas_node_offset; // into blas_nodes[]
	int blas_prim_offset; // into blas_prim_indices[]
	int num_triangles;
	int num_vertices;
	int has_normals; // 0 or 1
	int has_uv;
};

// ─── ONB on device ───────────────────────────────────────────────────────────

struct GpuOnb {
	float3 u, v, n;

	__device__ GpuOnb() : u{}, v{}, n{} {}

	__device__ explicit GpuOnb(float3 normal) : n(normal) {
		float s = (n.z >= 0.f) ? 1.f : -1.f;
		float a = -1.f / (s + n.z);
		float b = n.x * n.y * a;
		u = make_float3(1.f + s * n.x * n.x * a, s * b, -s * n.x);
		v = make_float3(b, s + n.y * n.y * a, -n.y);
	}

	__device__ float3 to_world(float3 local) const {
		return local.x * u + local.y * v + local.z * n;
	}

	__device__ float3 to_local(float3 world) const {
		return make_float3(dot3(world, u), dot3(world, v), dot3(world, n));
	}
};

// ─── Ray origin offset (device) ──────────────────────────────────────────────

__device__ inline float3 gpu_offset_ray_origin(float3 p, float3 ng, float3 dir, float eps) {
	float3 n = ng;

	if (dot3(n, dir) < 0.f)
		n = -n;
	float3 result = p + n * eps;

	for (int i = 0; i < 3; ++i) {
		if ((&n.x)[i] > 0.f)
			(&result.x)[i] = nextafterf((&result.x)[i], kGpuInfinity);
		else if ((&n.x)[i] < 0.f)
			(&result.x)[i] = nextafterf((&result.x)[i], -kGpuInfinity);
	}

	return result;
}

__device__ inline float3 gpu_offset_ray_origin(float3 p, float3 ng, float3 dir) {
	return gpu_offset_ray_origin(p, ng, dir, kGpuRayEps);
}

__device__ inline float3 gpu_shading_normal(const GpuHitRecord &hit, float3 wo, bool transmissive) {
	float3 n = hit.normal;

	if (dot3(n, wo) * dot3(hit.geo_normal, wo) <= 0.f)
		n = hit.geo_normal;

	if (!transmissive && dot3(n, wo) < 0.f)
		n = -n;

	return n;
}

__device__ inline bool gpu_valid_scattering(float3 wo, float3 wi, float3 ns, float3 ng) {
	return dot3(wo, ns) * dot3(wi, ns) * dot3(wo, ng) * dot3(wi, ng) > 0.f;
}

} // namespace xn
