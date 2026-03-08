#pragma once
// cuda/gpu_scene.cuh — Host-side GPU scene manager
//
// GpuSceneHost flattens the CPU Scene into concatenated device arrays
// and uploads them. Called once after scene loading.

#include "cuda/gpu_types.cuh"
#include "cuda/gpu_traverse.cuh"  // for GpuSceneData

namespace xn {

// Forward declarations (CPU types)
struct Scene;
struct Camera;

class GpuSceneHost {
public:
  GpuSceneHost() = default;
  ~GpuSceneHost();

  GpuSceneHost(const GpuSceneHost&) = delete;
  GpuSceneHost& operator=(const GpuSceneHost&) = delete;

  // Build flat GPU representation from CPU scene.
  // Camera is uploaded separately (it can change per frame).
  void upload(const Scene& scene);

  // Free all device memory.
  void free_device();

  // Get the device-side data bundle (for passing to kernels).
  GpuSceneData get_scene_data() const { return scene_data_; }

  // Upload camera parameters
  GpuCamera upload_camera(const Camera& camera, int width, int height) const;

private:
  GpuSceneData scene_data_ = {};
  bool         uploaded_   = false;

  // Device-owned raw pointers
  float*       d_vx_ = nullptr;
  float*       d_vy_ = nullptr;
  float*       d_vz_ = nullptr;
  float*       d_nx_ = nullptr;
  float*       d_ny_ = nullptr;
  float*       d_nz_ = nullptr;
  int32_t*     d_indices_ = nullptr;
  int32_t*     d_mat_ids_ = nullptr;
  GpuBVHNode*  d_blas_nodes_ = nullptr;
  uint32_t*    d_blas_prims_ = nullptr;
  GpuBVHNode*  d_tlas_nodes_ = nullptr;
  uint32_t*    d_tlas_inst_indices_ = nullptr;
  GpuMeshInfo* d_mesh_info_ = nullptr;
  GpuInstance*  d_instances_ = nullptr;
  GpuMaterial* d_materials_ = nullptr;
  GpuLight*    d_lights_ = nullptr;
  GpuDirLight* d_dir_lights_ = nullptr;
};

} // namespace xn
