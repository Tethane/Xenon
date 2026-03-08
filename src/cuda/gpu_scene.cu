// cuda/gpu_scene.cu — Host-side scene flattening and GPU upload
//
// Reads the CPU Scene and produces a flat GpuSceneData on device memory.
// Called once after scene loading / acceleration structure build.

#include <cstdio>
#include <vector>

#include "cuda/gpu_scene.cuh"
#include "cuda/cuda_utils.cuh"
#include "scene/scene.h"
#include "camera/camera.h"

namespace xn {

// ─── Helper: upload a vector to device memory ────────────────────────────────

template<typename T>
static T* upload_vec(const std::vector<T>& src) {
  if (src.empty()) return nullptr;
  T* d_ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&d_ptr, src.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_ptr, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d_ptr;
}

template<typename T>
static T* upload_data(const T* src, size_t count) {
  if (count == 0 || !src) return nullptr;
  T* d_ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_ptr, src, count * sizeof(T), cudaMemcpyHostToDevice));
  return d_ptr;
}

// ─── Upload ──────────────────────────────────────────────────────────────────

void GpuSceneHost::upload(const Scene& scene) {
  if (uploaded_) free_device();

  // Count totals
  int total_vertices  = 0;
  int total_indices   = 0;
  int total_mat_ids   = 0;
  int total_blas_nodes = 0;
  int total_blas_prims = 0;

  int num_meshes = (int)scene.meshes.size();

  for (int i = 0; i < num_meshes; ++i) {
    total_vertices   += scene.meshes[i].num_vertices();
    total_indices    += (int)scene.meshes[i].indices.size();
    total_mat_ids    += (int)scene.meshes[i].mat_ids.size();
    total_blas_nodes += (int)scene.blas_list[i]->nodes().size();
    total_blas_prims += (int)scene.blas_list[i]->prims().size();
  }

  std::printf("[CUDA] Uploading scene: %d meshes, %d vertices, %d triangles, %d materials\n",
              num_meshes, total_vertices, total_indices / 3, (int)scene.materials.size());

  // ── Concatenate mesh data ──────────────────────────────────────────────────

  std::vector<float> all_vx, all_vy, all_vz;
  std::vector<float> all_nx, all_ny, all_nz;
  std::vector<int32_t> all_indices, all_mat_ids;
  std::vector<GpuBVHNode> all_blas_nodes;
  std::vector<uint32_t> all_blas_prims;
  std::vector<GpuMeshInfo> mesh_info;

  all_vx.reserve(total_vertices);
  all_vy.reserve(total_vertices);
  all_vz.reserve(total_vertices);
  all_nx.reserve(total_vertices);
  all_ny.reserve(total_vertices);
  all_nz.reserve(total_vertices);
  all_indices.reserve(total_indices);
  all_mat_ids.reserve(total_mat_ids);
  all_blas_nodes.reserve(total_blas_nodes);
  all_blas_prims.reserve(total_blas_prims);
  mesh_info.reserve(num_meshes);

  for (int i = 0; i < num_meshes; ++i) {
    const auto& mesh = scene.meshes[i];
    const auto& blas = *scene.blas_list[i];

    GpuMeshInfo mi{};
    mi.vertex_offset    = (int)all_vx.size();
    mi.tri_offset       = (int)all_indices.size();
    mi.mat_ids_offset   = (int)all_mat_ids.size();
    mi.blas_node_offset = (int)all_blas_nodes.size();
    mi.blas_prim_offset = (int)all_blas_prims.size();
    mi.num_triangles    = mesh.num_triangles();
    mi.num_vertices     = mesh.num_vertices();
    mi.has_normals      = mesh.nx.empty() ? 0 : 1;

    // Vertices
    all_vx.insert(all_vx.end(), mesh.vx.begin(), mesh.vx.end());
    all_vy.insert(all_vy.end(), mesh.vy.begin(), mesh.vy.end());
    all_vz.insert(all_vz.end(), mesh.vz.begin(), mesh.vz.end());

    // Normals
    if (mi.has_normals) {
      all_nx.insert(all_nx.end(), mesh.nx.begin(), mesh.nx.end());
      all_ny.insert(all_ny.end(), mesh.ny.begin(), mesh.ny.end());
      all_nz.insert(all_nz.end(), mesh.nz.begin(), mesh.nz.end());
    } else {
      all_nx.resize(all_nx.size() + mesh.num_vertices(), 0.f);
      all_ny.resize(all_ny.size() + mesh.num_vertices(), 0.f);
      all_nz.resize(all_nz.size() + mesh.num_vertices(), 0.f);
    }

    // Indices
    all_indices.insert(all_indices.end(), mesh.indices.begin(), mesh.indices.end());

    // Material IDs
    all_mat_ids.insert(all_mat_ids.end(), mesh.mat_ids.begin(), mesh.mat_ids.end());

    // BLAS nodes — convert from BLASNode to GpuBVHNode
    const auto& cpu_nodes = blas.nodes();
    for (const auto& node : cpu_nodes) {
      GpuBVHNode gn{};
      gn.mn    = make_float3(node.bbox.mn.x, node.bbox.mn.y, node.bbox.mn.z);
      gn.mx    = make_float3(node.bbox.mx.x, node.bbox.mx.y, node.bbox.mx.z);
      gn.child = node.child;
      gn.count = node.count;
      all_blas_nodes.push_back(gn);
    }

    // BLAS prims
    const auto& cpu_prims = blas.prims();
    all_blas_prims.insert(all_blas_prims.end(), cpu_prims.begin(), cpu_prims.end());

    mesh_info.push_back(mi);
  }

  // ── TLAS ───────────────────────────────────────────────────────────────────

  const auto& tlas_cpu_nodes = scene.tlas.nodes();
  const auto& tlas_cpu_inst  = scene.tlas.inst_indices();
  const auto& tlas_cpu_instances = scene.tlas.instances();

  // Build CPU→GPU instance remap table.
  // PrimBLAS instances are skipped on GPU, so TLAS leaf indices must be
  // rewritten to point into the filtered GPU instances array.
  int num_cpu_instances = (int)tlas_cpu_instances.size();
  std::vector<int> cpu_to_gpu_inst(num_cpu_instances, -1);  // -1 = skipped

  // Convert instances — only mesh instances (skip prim BLAS for GPU)
  std::vector<GpuInstance> gpu_instances;
  gpu_instances.reserve(tlas_cpu_instances.size());
  for (int ci = 0; ci < num_cpu_instances; ++ci) {
    const auto& inst = tlas_cpu_instances[ci];
    // Only upload mesh-based instances; skip PrimBLAS instances
    if (inst.geom.type != GeomHandle::MESH_BLAS) continue;

    int gpu_idx = (int)gpu_instances.size();
    cpu_to_gpu_inst[ci] = gpu_idx;

    GpuInstance gi{};
    gi.instance_id = inst.instance_id;

    // Find which mesh this BLAS belongs to
    for (int m = 0; m < num_meshes; ++m) {
      if (scene.blas_list[m].get() == inst.geom.mesh) {
        gi.mesh_id = m;
        break;
      }
    }

    // Copy transforms
    for (int c = 0; c < 4; ++c) {
      gi.local_to_world.cols[c] = make_float3(
        inst.xform.local_to_world.cols[c].x,
        inst.xform.local_to_world.cols[c].y,
        inst.xform.local_to_world.cols[c].z);
      gi.world_to_local.cols[c] = make_float3(
        inst.xform.world_to_local.cols[c].x,
        inst.xform.world_to_local.cols[c].y,
        inst.xform.world_to_local.cols[c].z);
    }

    gi.world_aabb_mn = make_float3(inst.world_aabb.mn.x, inst.world_aabb.mn.y, inst.world_aabb.mn.z);
    gi.world_aabb_mx = make_float3(inst.world_aabb.mx.x, inst.world_aabb.mx.y, inst.world_aabb.mx.z);

    gpu_instances.push_back(gi);
  }

  // Rewrite TLAS inst_indices to use GPU instance indices.
  // Entries that reference PrimBLAS (cpu_to_gpu_inst == -1) are kept but
  // will never be reached as long as the TLAS node AABBs are correct.
  // We rebuild the TLAS nodes too: if a leaf references only PrimBLAS
  // instances, we zero its count so gpu_tlas_intersect skips it.
  std::vector<GpuBVHNode> tlas_nodes;
  tlas_nodes.reserve(tlas_cpu_nodes.size());
  for (const auto& node : tlas_cpu_nodes) {
    GpuBVHNode gn{};
    gn.mn    = make_float3(node.bbox.mn.x, node.bbox.mn.y, node.bbox.mn.z);
    gn.mx    = make_float3(node.bbox.mx.x, node.bbox.mx.y, node.bbox.mx.z);
    gn.child = node.child;
    gn.count = node.count;
    tlas_nodes.push_back(gn);
  }

  // Remap tlas_inst_indices from CPU instance IDs to GPU instance IDs
  std::vector<uint32_t> gpu_tlas_inst_indices;
  gpu_tlas_inst_indices.reserve(tlas_cpu_inst.size());
  for (uint32_t ci : tlas_cpu_inst) {
    if ((int)ci < num_cpu_instances && cpu_to_gpu_inst[ci] >= 0) {
      gpu_tlas_inst_indices.push_back((uint32_t)cpu_to_gpu_inst[ci]);
    } else {
      // PrimBLAS instance — push sentinel (will be skipped by count=0 fix below)
      gpu_tlas_inst_indices.push_back(0xFFFFFFFF);
    }
  }

  // Fix up TLAS leaf nodes: replace leaves that reference only PrimBLAS
  // instances with a safe count/child so they are skipped during traversal.
  for (auto& gn : tlas_nodes) {
    if (gn.count == 0) continue; // internal node
    // Count how many valid GPU instances this leaf references
    uint32_t valid = 0;
    uint32_t first_valid = gn.child;
    for (uint32_t i = 0; i < gn.count; ++i) {
      uint32_t idx = gn.child + i;
      if (idx < gpu_tlas_inst_indices.size() && gpu_tlas_inst_indices[idx] != 0xFFFFFFFF) {
        if (valid == 0) first_valid = idx;
        // Compact: move valid entries forward
        if (idx != first_valid + valid) {
          gpu_tlas_inst_indices[first_valid + valid] = gpu_tlas_inst_indices[idx];
        }
        valid++;
      }
    }
    gn.child = first_valid;
    gn.count = valid;
  }

  // ── Materials ──────────────────────────────────────────────────────────────

  std::vector<GpuMaterial> gpu_mats;
  gpu_mats.reserve(scene.materials.size());
  for (const auto& m : scene.materials) {
    GpuMaterial gm{};
    gm.baseColor       = make_float3(m.baseColor.x, m.baseColor.y, m.baseColor.z);
    gm.roughness       = m.roughness;
    gm.metallic        = m.metallic;
    gm.ior             = m.ior;
    gm.transmission    = m.transmission;
    gm.subsurface      = m.subsurface;
    gm.subsurfaceColor = make_float3(m.subsurfaceColor.x, m.subsurfaceColor.y, m.subsurfaceColor.z);
    gm.alpha           = m.alpha;
    gm.alpha_x         = m.alpha;  // isotropic for now
    gm.alpha_y         = m.alpha;
    gm.F0              = make_float3(m.F0.x, m.F0.y, m.F0.z);
    gm.emission        = make_float3(m.emission.x, m.emission.y, m.emission.z);
    gm.flags = 0;
    if (m.isConductor)    gm.flags |= GpuMaterial::FLAG_CONDUCTOR;
    if (m.isTransmissive) gm.flags |= GpuMaterial::FLAG_TRANSMISSIVE;
    if (m.hasSubsurface)  gm.flags |= GpuMaterial::FLAG_SUBSURFACE;
    if (m.isDelta)        gm.flags |= GpuMaterial::FLAG_DELTA;
    if (m.isEmissive)     gm.flags |= GpuMaterial::FLAG_EMISSIVE;
    gpu_mats.push_back(gm);
  }

  // ── Lights ─────────────────────────────────────────────────────────────────

  std::vector<GpuLight> gpu_lights;
  gpu_lights.reserve(scene.lights.size());
  for (const auto& l : scene.lights) {
    GpuLight gl{};
    gl.mesh_id  = l.mesh_id;
    gl.tri_idx  = l.tri_idx;
    gl.emission = make_float3(l.emission.x, l.emission.y, l.emission.z);
    gl.area     = l.area;
    gpu_lights.push_back(gl);
  }

  std::vector<GpuDirLight> gpu_dir_lights;
  gpu_dir_lights.reserve(scene.dir_lights.size());
  for (const auto& dl : scene.dir_lights) {
    GpuDirLight gdl{};
    gdl.direction = make_float3(dl.direction.x, dl.direction.y, dl.direction.z);
    gdl.color     = make_float3(dl.color.x, dl.color.y, dl.color.z);
    gdl.intensity = dl.intensity;
    gpu_dir_lights.push_back(gdl);
  }

  // ── Upload to device ───────────────────────────────────────────────────────

  d_vx_        = upload_vec(all_vx);
  d_vy_        = upload_vec(all_vy);
  d_vz_        = upload_vec(all_vz);
  d_nx_        = upload_vec(all_nx);
  d_ny_        = upload_vec(all_ny);
  d_nz_        = upload_vec(all_nz);
  d_indices_   = upload_vec(all_indices);
  d_mat_ids_   = upload_vec(all_mat_ids);
  d_blas_nodes_ = upload_vec(all_blas_nodes);
  d_blas_prims_ = upload_vec(all_blas_prims);
  d_tlas_nodes_ = upload_vec(tlas_nodes);
  d_tlas_inst_indices_ = upload_vec(gpu_tlas_inst_indices);
  d_mesh_info_  = upload_vec(mesh_info);
  d_instances_  = upload_vec(gpu_instances);
  d_materials_  = upload_vec(gpu_mats);
  d_lights_     = upload_vec(gpu_lights);
  d_dir_lights_ = upload_vec(gpu_dir_lights);

  // ── Fill scene data struct ─────────────────────────────────────────────────

  scene_data_.vertices_x       = d_vx_;
  scene_data_.vertices_y       = d_vy_;
  scene_data_.vertices_z       = d_vz_;
  scene_data_.normals_x        = d_nx_;
  scene_data_.normals_y        = d_ny_;
  scene_data_.normals_z        = d_nz_;
  scene_data_.indices          = d_indices_;
  scene_data_.mat_ids          = d_mat_ids_;
  scene_data_.blas_nodes       = d_blas_nodes_;
  scene_data_.blas_prims       = d_blas_prims_;
  scene_data_.tlas_nodes       = d_tlas_nodes_;
  scene_data_.tlas_inst_indices = d_tlas_inst_indices_;
  scene_data_.mesh_info        = d_mesh_info_;
  scene_data_.instances        = d_instances_;
  scene_data_.materials        = d_materials_;
  scene_data_.lights           = d_lights_;
  scene_data_.dir_lights       = d_dir_lights_;

  // Environment
  scene_data_.env.zenith_color      = make_float3(scene.sky.zenith_color.x, scene.sky.zenith_color.y, scene.sky.zenith_color.z);
  scene_data_.env.horizon_color     = make_float3(scene.sky.horizon_color.x, scene.sky.horizon_color.y, scene.sky.horizon_color.z);
  scene_data_.env.ground_color      = make_float3(scene.sky.ground_color.x, scene.sky.ground_color.y, scene.sky.ground_color.z);
  scene_data_.env.horizon_sharpness = scene.sky.horizon_sharpness;
  scene_data_.env.intensity         = scene.sky.intensity;

  // Counts
  scene_data_.num_meshes      = num_meshes;
  scene_data_.num_materials   = (int)scene.materials.size();
  scene_data_.num_lights      = (int)gpu_lights.size();
  scene_data_.num_dir_lights  = (int)gpu_dir_lights.size();
  scene_data_.num_instances   = (int)gpu_instances.size();
  scene_data_.num_tlas_nodes  = (int)tlas_nodes.size();

  uploaded_ = true;

  size_t total_bytes = 0;
  total_bytes += total_vertices * 3 * sizeof(float) * 2; // positions + normals
  total_bytes += total_indices * sizeof(int32_t);
  total_bytes += total_mat_ids * sizeof(int32_t);
  total_bytes += all_blas_nodes.size() * sizeof(GpuBVHNode);
  total_bytes += all_blas_prims.size() * sizeof(uint32_t);
  total_bytes += tlas_nodes.size() * sizeof(GpuBVHNode);
  total_bytes += tlas_cpu_inst.size() * sizeof(uint32_t);
  total_bytes += mesh_info.size() * sizeof(GpuMeshInfo);
  total_bytes += gpu_instances.size() * sizeof(GpuInstance);
  total_bytes += gpu_mats.size() * sizeof(GpuMaterial);
  total_bytes += gpu_lights.size() * sizeof(GpuLight);
  total_bytes += gpu_dir_lights.size() * sizeof(GpuDirLight);

  std::printf("[CUDA] Scene uploaded: %.2f MB device memory\n",
              total_bytes / (1024.0 * 1024.0));
  std::printf("[CUDA]   %d BLAS nodes, %d TLAS nodes, %d instances\n",
              (int)all_blas_nodes.size(), (int)tlas_nodes.size(), (int)gpu_instances.size());
}

// ─── Free device memory ──────────────────────────────────────────────────────

void GpuSceneHost::free_device() {
  if (!uploaded_) return;

  auto safe_free = [](auto*& p) {
    if (p) { cudaFree(p); p = nullptr; }
  };

  safe_free(d_vx_); safe_free(d_vy_); safe_free(d_vz_);
  safe_free(d_nx_); safe_free(d_ny_); safe_free(d_nz_);
  safe_free(d_indices_); safe_free(d_mat_ids_);
  safe_free(d_blas_nodes_); safe_free(d_blas_prims_);
  safe_free(d_tlas_nodes_); safe_free(d_tlas_inst_indices_);
  safe_free(d_mesh_info_); safe_free(d_instances_);
  safe_free(d_materials_); safe_free(d_lights_); safe_free(d_dir_lights_);

  scene_data_ = {};
  uploaded_ = false;
}

GpuSceneHost::~GpuSceneHost() {
  free_device();
}

// ─── Camera upload ───────────────────────────────────────────────────────────

GpuCamera GpuSceneHost::upload_camera(const Camera& cam, int width, int height) const {
  GpuCamera gc{};
  Vec3 eye = cam.get_eye();
  Vec3 ll  = cam.get_lower_left_corner();
  Vec3 h   = cam.get_horizontal();
  Vec3 v   = cam.get_vertical();

  gc.eye               = make_float3(eye.x, eye.y, eye.z);
  gc.lower_left_corner = make_float3(ll.x, ll.y, ll.z);
  gc.horizontal        = make_float3(h.x, h.y, h.z);
  gc.vertical          = make_float3(v.x, v.y, v.z);

  return gc;
}

} // namespace xn
