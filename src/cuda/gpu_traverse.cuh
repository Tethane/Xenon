#pragma once
// cuda/gpu_traverse.cuh — Device-side BVH traversal (TLAS → BLAS)
//
// Scalar stack-based DFS. The GPU gets throughput from warp-level parallelism,
// not from per-ray SIMD (unlike the CPU backend's SSE pair test).

#include "cuda/gpu_types.cuh"

namespace xn {

// ─── Device scene pointer bundle ─────────────────────────────────────────────
// Passed to all traversal / shading functions by const reference.

struct GpuSceneData {
  // Concatenated mesh data
  const float*     vertices_x;
  const float*     vertices_y;
  const float*     vertices_z;
  const float*     normals_x;
  const float*     normals_y;
  const float*     normals_z;
  const int32_t*   indices;
  const int32_t*   mat_ids;

  // BVH
  const GpuBVHNode* blas_nodes;
  const uint32_t*   blas_prims;
  const GpuBVHNode* tlas_nodes;
  const uint32_t*   tlas_inst_indices;

  // Per-mesh info
  const GpuMeshInfo* mesh_info;

  // Instances
  const GpuInstance* instances;

  // Materials
  const GpuMaterial* materials;

  // Lights
  const GpuLight*   lights;
  const GpuDirLight* dir_lights;
  GpuEnvironment    env;

  // Counts
  int num_meshes;
  int num_materials;
  int num_lights;
  int num_dir_lights;
  int num_instances;
  int num_tlas_nodes;
  int num_pixels;      // for bounds-checking atomicAdd in kernels
};

// ─── Möller–Trumbore triangle intersection ───────────────────────────────────

__device__ inline bool gpu_ray_triangle(
    const GpuRay& ray,
    float3 v0, float3 v1, float3 v2,
    float& t_out, float& u_out, float& v_out,
    float t_max)
{
  float3 e1 = v1 - v0;
  float3 e2 = v2 - v0;
  float3 h  = cross3(ray.dir, e2);
  float  a  = dot3(e1, h);
  if (fabsf(a) < 1e-7f) return false;

  float inv_a = 1.f / a;
  float3 s = ray.origin - v0;
  float  u = inv_a * dot3(s, h);
  if (u < 0.f || u > 1.f) return false;

  float3 q = cross3(s, e1);
  float  v = inv_a * dot3(ray.dir, q);
  if (v < 0.f || u + v > 1.f) return false;

  float t = inv_a * dot3(e2, q);
  if (t < ray.tmin || t > t_max) return false;

  t_out = t; u_out = u; v_out = v;
  return true;
}

// ─── Triangle intersection with full HitRecord population ────────────────────

__device__ inline bool gpu_mesh_intersect_triangle(
    const GpuRay& ray, int tri_idx,
    const GpuSceneData& scene, int mesh_id,
    GpuHitRecord& rec)
{
  const GpuMeshInfo& mi = scene.mesh_info[mesh_id];
  int base = mi.tri_offset + tri_idx * 3;
  int i0 = scene.indices[base + 0];
  int i1 = scene.indices[base + 1];
  int i2 = scene.indices[base + 2];

  int vo = mi.vertex_offset;
  float3 p0 = make_f3(scene.vertices_x[vo+i0], scene.vertices_y[vo+i0], scene.vertices_z[vo+i0]);
  float3 p1 = make_f3(scene.vertices_x[vo+i1], scene.vertices_y[vo+i1], scene.vertices_z[vo+i1]);
  float3 p2 = make_f3(scene.vertices_x[vo+i2], scene.vertices_y[vo+i2], scene.vertices_z[vo+i2]);

  float t, u, v;
  if (!gpu_ray_triangle(ray, p0, p1, p2, t, u, v, rec.t)) return false;

  rec.t   = t;
  rec.pos = ray.at(t);
  rec.u   = u;
  rec.v   = v;
  rec.mat_id = scene.mat_ids[mi.mat_ids_offset + tri_idx];
  rec.prim_id = tri_idx;

  float3 gn = normalize3(cross3(p1 - p0, p2 - p0));
  rec.geo_normal = gn;
  rec.front_face = dot3(gn, ray.dir) < 0.f;

  if (mi.has_normals) {
    int no = mi.vertex_offset; // normals use same offset as vertices
    float3 n0 = make_f3(scene.normals_x[no+i0], scene.normals_y[no+i0], scene.normals_z[no+i0]);
    float3 n1 = make_f3(scene.normals_x[no+i1], scene.normals_y[no+i1], scene.normals_z[no+i1]);
    float3 n2 = make_f3(scene.normals_x[no+i2], scene.normals_y[no+i2], scene.normals_z[no+i2]);
    rec.normal = normalize3(n0 * (1.f - u - v) + n1 * u + n2 * v);
  } else {
    rec.normal = gn;
  }
  return true;
}

// ─── AABB slab test (scalar) ─────────────────────────────────────────────────

__device__ inline bool gpu_aabb_intersect(
    float3 mn, float3 mx,
    float3 inv_dir, float3 origin,
    float tmin, float tmax,
    float& tn_out)
{
  float tx0 = (mn.x - origin.x) * inv_dir.x;
  float tx1 = (mx.x - origin.x) * inv_dir.x;
  float ty0 = (mn.y - origin.y) * inv_dir.y;
  float ty1 = (mx.y - origin.y) * inv_dir.y;
  float tz0 = (mn.z - origin.z) * inv_dir.z;
  float tz1 = (mx.z - origin.z) * inv_dir.z;

  float tn = fmaxf(fmaxf(fminf(tx0,tx1), fminf(ty0,ty1)), fmaxf(fminf(tz0,tz1), tmin));
  float tf = fminf(fminf(fmaxf(tx0,tx1), fmaxf(ty0,ty1)), fminf(fmaxf(tz0,tz1), tmax));

  tn_out = tn;
  return tn <= tf;
}

// ─── BLAS traversal — closest hit ────────────────────────────────────────────

__device__ inline bool gpu_blas_intersect(
    const GpuRay& ray, const GpuSceneData& scene, int mesh_id, GpuHitRecord& rec)
{
  const GpuMeshInfo& mi = scene.mesh_info[mesh_id];
  int node_off = mi.blas_node_offset;
  int prim_off = mi.blas_prim_offset;

  float3 inv_dir = make_f3(1.f/ray.dir.x, 1.f/ray.dir.y, 1.f/ray.dir.z);

  uint32_t stack[32];
  int ptr = 0;
  stack[ptr++] = 0;
  bool hit = false;

  while (ptr > 0) {
    uint32_t idx = stack[--ptr];
    const GpuBVHNode& node = scene.blas_nodes[node_off + idx];

    if (node.is_leaf()) {
      for (uint32_t i = 0; i < node.count; ++i) {
        uint32_t tri = scene.blas_prims[prim_off + node.child + i];
        if (gpu_mesh_intersect_triangle(ray, tri, scene, mesh_id, rec))
          hit = true;
      }
      continue;
    }

    uint32_t left  = node.child;
    uint32_t right = left + 1;
    float tn_l, tn_r;
    bool hit_l = gpu_aabb_intersect(
      scene.blas_nodes[node_off+left].mn, scene.blas_nodes[node_off+left].mx,
      inv_dir, ray.origin, ray.tmin, rec.t, tn_l);
    bool hit_r = gpu_aabb_intersect(
      scene.blas_nodes[node_off+right].mn, scene.blas_nodes[node_off+right].mx,
      inv_dir, ray.origin, ray.tmin, rec.t, tn_r);

    if (hit_l && hit_r) {
      if (tn_l < tn_r) { stack[ptr++] = right; stack[ptr++] = left; }
      else             { stack[ptr++] = left;  stack[ptr++] = right; }
    } else if (hit_l) { stack[ptr++] = left; }
    else if (hit_r)   { stack[ptr++] = right; }
  }
  return hit;
}

// ─── BLAS traversal — any hit (shadow) ───────────────────────────────────────

__device__ inline bool gpu_blas_intersects(
    const GpuRay& ray, const GpuSceneData& scene, int mesh_id)
{
  const GpuMeshInfo& mi = scene.mesh_info[mesh_id];
  int node_off = mi.blas_node_offset;
  int prim_off = mi.blas_prim_offset;

  float3 inv_dir = make_f3(1.f/ray.dir.x, 1.f/ray.dir.y, 1.f/ray.dir.z);

  uint32_t stack[32];
  int ptr = 0;
  stack[ptr++] = 0;

  while (ptr > 0) {
    uint32_t idx = stack[--ptr];
    const GpuBVHNode& node = scene.blas_nodes[node_off + idx];

    if (node.is_leaf()) {
      for (uint32_t i = 0; i < node.count; ++i) {
        uint32_t tri = scene.blas_prims[prim_off + node.child + i];
        // Inline triangle intersection (no need to populate full HitRecord)
        const GpuMeshInfo& mi2 = scene.mesh_info[mesh_id];
        int base = mi2.tri_offset + tri * 3;
        int i0 = scene.indices[base]; int i1 = scene.indices[base+1]; int i2 = scene.indices[base+2];
        int vo = mi2.vertex_offset;
        float3 p0 = make_f3(scene.vertices_x[vo+i0], scene.vertices_y[vo+i0], scene.vertices_z[vo+i0]);
        float3 p1 = make_f3(scene.vertices_x[vo+i1], scene.vertices_y[vo+i1], scene.vertices_z[vo+i1]);
        float3 p2 = make_f3(scene.vertices_x[vo+i2], scene.vertices_y[vo+i2], scene.vertices_z[vo+i2]);
        float t, u, v;
        if (gpu_ray_triangle(ray, p0, p1, p2, t, u, v, ray.tmax))
          return true;
      }
      continue;
    }

    uint32_t left  = node.child;
    uint32_t right = left + 1;
    float tn;
    bool hit_l = gpu_aabb_intersect(
      scene.blas_nodes[node_off+left].mn, scene.blas_nodes[node_off+left].mx,
      inv_dir, ray.origin, ray.tmin, ray.tmax, tn);
    bool hit_r = gpu_aabb_intersect(
      scene.blas_nodes[node_off+right].mn, scene.blas_nodes[node_off+right].mx,
      inv_dir, ray.origin, ray.tmin, ray.tmax, tn);

    if (hit_l) stack[ptr++] = left;
    if (hit_r) stack[ptr++] = right;
  }
  return false;
}

// ─── TLAS traversal — closest hit ────────────────────────────────────────────

__device__ inline void gpu_tlas_fixup_hit(
    GpuHitRecord& rec, const GpuRay& world_ray,
    const GpuInstance& inst)
{
  rec.pos = world_ray.at(rec.t);
  // Normal transform: local→world uses transpose(inverse(M)) = local_to_world^T.
  // GpuMat4x3::transform_normal already computes M^T * n, so passing
  // local_to_world gives the correct cofactor-based normal transform.
  rec.normal = normalize3(inst.local_to_world.transform_normal(rec.normal));
  rec.geo_normal = normalize3(inst.local_to_world.transform_normal(rec.geo_normal));
  rec.front_face = dot3(rec.geo_normal, world_ray.dir) < 0.f;
  // Face-forward shading normal to agree with geometric normal orientation
  if (dot3(rec.normal, rec.geo_normal) < 0.f) rec.normal = -rec.normal;
  // Face-forward geo_normal against incoming ray
  if (!rec.front_face) rec.geo_normal = -rec.geo_normal;
  rec.instance_id = inst.instance_id;
}

__device__ inline bool gpu_tlas_intersect(
    const GpuRay& world_ray, const GpuSceneData& scene, GpuHitRecord& rec)
{
  if (scene.num_tlas_nodes == 0) return false;

  float3 inv_dir = make_f3(1.f/world_ray.dir.x, 1.f/world_ray.dir.y, 1.f/world_ray.dir.z);

  uint32_t stack[32];
  int ptr = 0;
  stack[ptr++] = 0;
  bool any_hit = false;

  while (ptr > 0) {
    uint32_t idx = stack[--ptr];
    const GpuBVHNode& node = scene.tlas_nodes[idx];

    if (node.is_leaf()) {
      for (uint32_t i = 0; i < node.count; ++i) {
        uint32_t inst_idx = scene.tlas_inst_indices[node.child + i];
        const GpuInstance& inst = scene.instances[inst_idx];

        // Transform ray to local space
        GpuRay local_ray;
        local_ray.origin = inst.world_to_local.transform_point(world_ray.origin);
        local_ray.dir    = inst.world_to_local.transform_dir(world_ray.dir);
        local_ray.tmin   = world_ray.tmin;
        local_ray.tmax   = rec.t; // use current best t

        if (gpu_blas_intersect(local_ray, scene, inst.mesh_id, rec)) {
          gpu_tlas_fixup_hit(rec, world_ray, inst);
          any_hit = true;
        }
      }
      continue;
    }

    uint32_t left  = node.child;
    uint32_t right = left + 1;
    float tn_l, tn_r;
    bool hit_l = gpu_aabb_intersect(scene.tlas_nodes[left].mn, scene.tlas_nodes[left].mx,
                                     inv_dir, world_ray.origin, world_ray.tmin, rec.t, tn_l);
    bool hit_r = gpu_aabb_intersect(scene.tlas_nodes[right].mn, scene.tlas_nodes[right].mx,
                                     inv_dir, world_ray.origin, world_ray.tmin, rec.t, tn_r);

    if (hit_l && hit_r) {
      if (tn_l < tn_r) { stack[ptr++] = right; stack[ptr++] = left; }
      else             { stack[ptr++] = left;  stack[ptr++] = right; }
    } else if (hit_l) { stack[ptr++] = left; }
    else if (hit_r)   { stack[ptr++] = right; }
  }
  return any_hit;
}

// ─── TLAS traversal — any hit (shadow) ───────────────────────────────────────

__device__ inline bool gpu_tlas_intersects(
    const GpuRay& world_ray, const GpuSceneData& scene)
{
  if (scene.num_tlas_nodes == 0) return false;

  float3 inv_dir = make_f3(1.f/world_ray.dir.x, 1.f/world_ray.dir.y, 1.f/world_ray.dir.z);

  uint32_t stack[32];
  int ptr = 0;
  stack[ptr++] = 0;

  while (ptr > 0) {
    uint32_t idx = stack[--ptr];
    const GpuBVHNode& node = scene.tlas_nodes[idx];

    if (node.is_leaf()) {
      for (uint32_t i = 0; i < node.count; ++i) {
        uint32_t inst_idx = scene.tlas_inst_indices[node.child + i];
        const GpuInstance& inst = scene.instances[inst_idx];

        GpuRay local_ray;
        local_ray.origin = inst.world_to_local.transform_point(world_ray.origin);
        local_ray.dir    = inst.world_to_local.transform_dir(world_ray.dir);
        local_ray.tmin   = world_ray.tmin;
        local_ray.tmax   = world_ray.tmax;

        if (gpu_blas_intersects(local_ray, scene, inst.mesh_id))
          return true;
      }
      continue;
    }

    uint32_t left  = node.child;
    uint32_t right = left + 1;
    float tn;
    bool hit_l = gpu_aabb_intersect(scene.tlas_nodes[left].mn, scene.tlas_nodes[left].mx,
                                     inv_dir, world_ray.origin, world_ray.tmin, world_ray.tmax, tn);
    bool hit_r = gpu_aabb_intersect(scene.tlas_nodes[right].mn, scene.tlas_nodes[right].mx,
                                     inv_dir, world_ray.origin, world_ray.tmin, world_ray.tmax, tn);

    if (hit_l) stack[ptr++] = left;
    if (hit_r) stack[ptr++] = right;
  }
  return false;
}

} // namespace xn
