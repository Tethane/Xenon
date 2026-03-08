#pragma once
// cuda/gpu_types.cuh — GPU-compatible type definitions for device code
//
// All types are plain structs usable in both __host__ and __device__ code.
// No std::vector, no virtual dispatch, no heap allocation.

#include <cuda_runtime.h>
#include <cstdint>

namespace xn {

// ─── float3 math helpers ─────────────────────────────────────────────────────
// CUDA's built-in float3 lacks operators; add the essentials inline.

__host__ __device__ inline float3 make_f3(float x, float y, float z) { return make_float3(x, y, z); }
__host__ __device__ inline float3 make_f3(float s) { return make_float3(s, s, s); }

__host__ __device__ inline float3 operator+(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ inline float3 operator-(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ inline float3 operator*(float3 a, float3 b) { return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); }
__host__ __device__ inline float3 operator*(float3 a, float s)  { return make_float3(a.x*s, a.y*s, a.z*s); }
__host__ __device__ inline float3 operator*(float s, float3 a)  { return a * s; }
__host__ __device__ inline float3 operator/(float3 a, float s)  { float r = 1.f/s; return a * r; }
__host__ __device__ inline float3 operator-(float3 a)            { return make_float3(-a.x, -a.y, -a.z); }

__host__ __device__ inline float3& operator+=(float3& a, float3 b) { a.x+=b.x; a.y+=b.y; a.z+=b.z; return a; }
__host__ __device__ inline float3& operator*=(float3& a, float s)  { a.x*=s; a.y*=s; a.z*=s; return a; }
__host__ __device__ inline float3& operator*=(float3& a, float3 b) { a.x*=b.x; a.y*=b.y; a.z*=b.z; return a; }
__host__ __device__ inline float3& operator/=(float3& a, float s)  { float r=1.f/s; a.x*=r; a.y*=r; a.z*=r; return a; }

__host__ __device__ inline float  dot3(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ inline float3 cross3(float3 a, float3 b) {
  return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
__host__ __device__ inline float  length_sq3(float3 v) { return dot3(v, v); }
__host__ __device__ inline float  length3(float3 v) { return sqrtf(length_sq3(v)); }
__host__ __device__ inline float3 normalize3(float3 v) { return v / length3(v); }
__host__ __device__ inline float3 reflect3(float3 v, float3 n) { return v - 2.f * dot3(v, n) * n; }
__host__ __device__ inline float3 faceforward3(float3 n, float3 ref) { return dot3(n, ref) < 0.f ? -n : n; }
__host__ __device__ inline float3 lerp3(float3 a, float3 b, float t) { return a + t * (b - a); }
__host__ __device__ inline float  max_component3(float3 v) { return fmaxf(fmaxf(v.x, v.y), v.z); }
__host__ __device__ inline float3 min3f(float3 a, float3 b) { return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z)); }
__host__ __device__ inline float3 max3f(float3 a, float3 b) { return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z)); }
__host__ __device__ inline float3 abs3f(float3 v) { return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }
__host__ __device__ inline float3 clamp3f(float3 v, float lo, float hi) {
  return make_float3(fminf(fmaxf(v.x,lo),hi), fminf(fmaxf(v.y,lo),hi), fminf(fmaxf(v.z,lo),hi));
}

__host__ __device__ inline bool refract3(float3 v, float3 n, float ni_over_nt, float3& out) {
  float3 uv = normalize3(v);
  float cos_i = -dot3(uv, n);
  float sin2_t = ni_over_nt * ni_over_nt * (1.f - cos_i * cos_i);
  if (sin2_t >= 1.f) return false;
  out = ni_over_nt * v + (ni_over_nt * cos_i - sqrtf(1.f - sin2_t)) * n;
  return true;
}

// ─── Constants ───────────────────────────────────────────────────────────────

constexpr float kGpuInfinity   = 1e30f;   // use finite value for device code
constexpr float kGpuRayEps     = 1e-4f;
constexpr float kGpuShadowEps  = 5e-4f;
constexpr float kGpuPi         = 3.14159265358979323846f;
constexpr float kGpuInvPi      = 1.f / kGpuPi;

// ─── GpuRay ──────────────────────────────────────────────────────────────────

struct GpuRay {
  float3 origin;
  float3 dir;
  float  tmin = kGpuRayEps;
  float  tmax = kGpuInfinity;

  __host__ __device__ float3 at(float t) const { return origin + t * dir; }
};

// ─── GpuHitRecord ────────────────────────────────────────────────────────────

struct GpuHitRecord {
  float  t          = kGpuInfinity;
  float3 pos        = {};
  float3 normal     = {};
  float3 geo_normal = {};
  int    mat_id     = -1;
  int    prim_id    = -1;
  int    instance_id = -1;
  float  u = 0, v = 0;
  bool   front_face = true;

  __host__ __device__ bool valid() const { return t < kGpuInfinity; }
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

// ─── GpuMaterial ─────────────────────────────────────────────────────────────
// Flat POD copy of Material without std::string

struct GpuMaterial {
  float3 baseColor;
  float  roughness;
  float  metallic;
  float  ior;
  float  transmission;
  float  subsurface;
  float3 subsurfaceColor;
  float  alpha;
  float  alpha_x;
  float  alpha_y;
  float3 F0;
  float3 emission;

  // Flags — packed into uint32_t to guarantee identical host/device layout.
  // bool fields cause unpredictable padding between host (CXX) and device (nvcc).
  uint32_t flags;

  static constexpr uint32_t FLAG_CONDUCTOR    = 1u << 0;
  static constexpr uint32_t FLAG_TRANSMISSIVE = 1u << 1;
  static constexpr uint32_t FLAG_SUBSURFACE   = 1u << 2;
  static constexpr uint32_t FLAG_DELTA        = 1u << 3;
  static constexpr uint32_t FLAG_EMISSIVE     = 1u << 4;

  __host__ __device__ bool isConductor()    const { return flags & FLAG_CONDUCTOR; }
  __host__ __device__ bool isTransmissive() const { return flags & FLAG_TRANSMISSIVE; }
  __host__ __device__ bool hasSubsurface()  const { return flags & FLAG_SUBSURFACE; }
  __host__ __device__ bool isDelta()        const { return flags & FLAG_DELTA; }
  __host__ __device__ bool isEmissive()     const { return flags & FLAG_EMISSIVE; }
};

// ─── GpuPathState ────────────────────────────────────────────────────────────

struct GpuPathState {
  GpuRay  ray;
  float3  throughput;
  float3  radiance;
  GpuPCG  rng;
  int     pixel_idx;
  int     depth;
  float   prev_bsdf_pdf_sa;
  uint8_t flags;  // bit 0: active, bit 1: specular, bit 2: prev_was_delta

  __device__ bool is_active()         const { return flags & 1; }
  __device__ bool is_specular()       const { return flags & 2; }
  __device__ bool was_prev_delta()    const { return flags & 4; }
  __device__ void set_active(bool b)        { if (b) flags |= 1; else flags &= ~1; }
  __device__ void set_specular(bool b)      { if (b) flags |= 2; else flags &= ~2; }
  __device__ void set_prev_delta(bool b)    { if (b) flags |= 4; else flags &= ~4; }
};

// ─── GpuLight ────────────────────────────────────────────────────────────────

struct GpuLight {
  uint32_t mesh_id;
  uint32_t tri_idx;
  float3   emission;
  float    area;
};

struct GpuDirLight {
  float3 direction;
  float3 color;
  float  intensity;

  __host__ __device__ float3 get_emission() const { return color * intensity; }
};

// ─── GpuEnvironment ──────────────────────────────────────────────────────────

struct GpuEnvironment {
  float3 zenith_color;
  float3 horizon_color;
  float3 ground_color;
  float  horizon_sharpness;
  float  intensity;

  __device__ float3 evaluate(float3 d) const {
    float y = d.y;
    if (y < 0.f) {
      float t = fminf(-y, 1.f);
      return intensity * lerp3(horizon_color, ground_color, t);
    } else {
      float t = powf(y, 1.0f / horizon_sharpness);
      return intensity * lerp3(horizon_color, zenith_color, t);
    }
  }

  __device__ bool active() const { return intensity > 0.f; }
};

// ─── GpuShadowWork ───────────────────────────────────────────────────────────

struct GpuShadowWork {
  int    path_idx;
  GpuRay ray;
  float  t_max;
  float3 contrib;
  int    origin_prim_id;
  int    light_prim_id;
};

// ─── GpuCamera ───────────────────────────────────────────────────────────────

struct GpuCamera {
  float3 eye;
  float3 lower_left_corner;
  float3 horizontal;
  float3 vertical;

  __device__ GpuRay get_ray(float s, float t) const {
    float3 target = lower_left_corner + s * horizontal + t * vertical;
    return GpuRay{eye, normalize3(target - eye)};
  }
};

// ─── GpuMat4x3 — column-major 3×4 affine matrix ─────────────────────────────

struct GpuMat4x3 {
  float3 cols[4];

  __device__ float3 transform_point(float3 p) const {
    return cols[0]*p.x + cols[1]*p.y + cols[2]*p.z + cols[3];
  }
  __device__ float3 transform_dir(float3 d) const {
    return cols[0]*d.x + cols[1]*d.y + cols[2]*d.z;
  }
  __device__ float3 transform_normal(float3 n) const {
    return make_float3(
      cols[0].x*n.x + cols[0].y*n.y + cols[0].z*n.z,
      cols[1].x*n.x + cols[1].y*n.y + cols[1].z*n.z,
      cols[2].x*n.x + cols[2].y*n.y + cols[2].z*n.z
    );
  }
};

// ─── GpuInstance ─────────────────────────────────────────────────────────────

struct GpuInstance {
  int        mesh_id;      // index into mesh_offsets
  int        instance_id;
  GpuMat4x3  local_to_world;
  GpuMat4x3  world_to_local;
  float3     world_aabb_mn;
  float3     world_aabb_mx;
};

// ─── GpuBVHNode — 32 bytes, mirrors CPU BLASNode/TLASNode ────────────────────

struct GpuBVHNode {
  float3   mn;        // AABB min
  float3   mx;        // AABB max
  uint32_t child;     // internal: left child index; leaf: first prim slot
  uint32_t count;     // 0 → internal, >0 → leaf

  __device__ bool is_leaf() const { return count > 0; }
};

// ─── Mesh offset table ───────────────────────────────────────────────────────
// Per-mesh metadata for indexing into the concatenated arrays.

struct GpuMeshInfo {
  int vertex_offset;      // into vertices_x/y/z
  int tri_offset;         // into indices[] (element offset, not triangle offset)
  int mat_ids_offset;     // into mat_ids[]
  int blas_node_offset;   // into blas_nodes[]
  int blas_prim_offset;   // into blas_prim_indices[]
  int num_triangles;
  int num_vertices;
  int has_normals;        // 0 or 1
};

// ─── ONB on device ───────────────────────────────────────────────────────────

struct GpuOnb {
  float3 u, v, n;

  __device__ GpuOnb() : u{}, v{}, n{} {}

  __device__ explicit GpuOnb(float3 normal) : n(normal) {
    float s = (n.z >= 0.f) ? 1.f : -1.f;
    float a = -1.f / (s + n.z);
    float b = n.x * n.y * a;
    u = make_float3(1.f + s*n.x*n.x*a, s*b, -s*n.x);
    v = make_float3(b, s + n.y*n.y*a, -n.y);
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
  if (dot3(n, dir) < 0.f) n = -n;
  return p + n * eps;
}

__device__ inline float3 gpu_offset_ray_origin(float3 p, float3 ng, float3 dir) {
  return gpu_offset_ray_origin(p, ng, dir, kGpuRayEps);
}

} // namespace xn
