// cuda/cuda_renderer.cu — CUDA wavefront path tracing renderer
//
// Full wavefront pipeline: raygen → intersect → NEE → shade → shadow → accumulate
// All work stays on device; only the final framebuffer is copied back per sample.

#include "cuda/cuda_renderer.cuh"
#include "cuda/cuda_utils.cuh"
#include "cuda/gpu_traverse.cuh"
#include "cuda/gpu_bsdf.cuh"
#include "scene/scene.h"
#include "camera/camera.h"
#include "render/swapchain.h"

#include <cstdio>
#include <cstring>

namespace xn {

// ─── Counter indices ─────────────────────────────────────────────────────────
enum CounterIdx {
  CNT_RAY      = 0,
  CNT_RAY_NEXT = 1,
  CNT_HIT      = 2,
  CNT_SHADOW   = 3,
  NUM_COUNTERS = 4
};

// ─── Block size for all kernels ──────────────────────────────────────────────
constexpr int kBlockSize = 256;

inline int grid_size(int n) { return (n + kBlockSize - 1) / kBlockSize; }

// ═════════════════════════════════════════════════════════════════════════════
// KERNELS
// ═════════════════════════════════════════════════════════════════════════════

// ─── Ray Generation ──────────────────────────────────────────────────────────

__global__ void raygen_kernel(
    GpuPathState* paths,
    int*          ray_queue,
    int*          counters,
    GpuCamera     camera,
    int           spp,
    int           width,
    int           height)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_pixels = width * height;
  if (idx >= num_pixels) return;

  int x = idx % width;
  int y = idx / width;

  // Seed RNG with pixel index and sample count
  GpuPCG rng = gpu_seed_pcg((uint64_t)idx * 1337 + spp * 7919, (uint64_t)idx);

  float u = ((float)x + rng.next_float()) / (float)width;
  float v = ((float)height - (float)y - rng.next_float()) / (float)height;

  GpuPathState& ps = paths[idx];
  ps.ray        = camera.get_ray(u, v);
  ps.throughput = make_f3(1.f);
  ps.radiance   = make_f3(0.f);
  ps.rng        = rng;
  ps.pixel_idx  = idx;
  ps.depth      = 0;
  ps.prev_bsdf_pdf_sa = 0.f;
  ps.flags      = 1; // active
  ps.set_prev_delta(true); // first bounce counts as delta for MIS

  ray_queue[idx] = idx;
  if (idx == 0) {
    counters[CNT_RAY] = num_pixels;
  }
}

// ─── Intersection ────────────────────────────────────────────────────────────

__global__ void intersect_kernel(
    GpuPathState*   paths,
    const int*      ray_queue,
    const int*      counters,
    GpuHitRecord*   hits,
    int*            hit_counter,
    GpuSceneData    scene)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_rays = counters[CNT_RAY];
  if (tid >= num_rays) return;

  int path_idx = ray_queue[tid];
  GpuPathState& ps = paths[path_idx];
  if (!ps.is_active()) return;

  GpuHitRecord rec{};
  rec.t = kGpuInfinity;

  if (gpu_tlas_intersect(ps.ray, scene, rec)) {
    int slot = atomicAdd(&hit_counter[CNT_HIT], 1);
    if (slot < scene.num_pixels) {  // bounds check
      hits[slot] = rec;
      hits[slot].prim_id = path_idx;  // reuse prim_id field to store path index
    }
  } else {
    // Miss — evaluate environment
    if (scene.env.active()) {
      float3 sky = scene.env.evaluate(normalize3(ps.ray.dir));

      // MIS weight for environment (if prev was delta, weight = 1)
      if (ps.was_prev_delta()) {
        ps.radiance += ps.throughput * sky;
      } else {
        // For non-delta previous bounces, environment gets full contribution
        // (no light PDF for environment in this simple implementation)
        ps.radiance += ps.throughput * sky;
      }
    }
    ps.set_active(false);
  }
}

// ─── NEE (Next Event Estimation) ─────────────────────────────────────────────

__global__ void nee_kernel(
    GpuPathState*   paths,
    const GpuHitRecord* hits,
    const int*      counters,
    GpuShadowWork*  shadow_work,
    int*            shadow_counter,
    GpuSceneData    scene,
    int             min_bounces)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_hits = counters[CNT_HIT];
  if (tid >= num_hits) return;

  const GpuHitRecord& hit = hits[tid];
  int path_idx = hit.prim_id;  // stored by intersect_kernel
  GpuPathState& ps = paths[path_idx];
  if (!ps.is_active()) return;

  const GpuMaterial& mat = scene.materials[hit.mat_id];

  // Add emission from hit surface (area lights)
  if (mat.isEmissive()) {
    // MIS weight: if prev was delta, full weight; otherwise compute
    if (ps.was_prev_delta() || ps.depth == 0) {
      ps.radiance += ps.throughput * mat.emission;
    } else if (ps.prev_bsdf_pdf_sa > 0.f) {
      // Compute light PDF for MIS
      float dist2 = hit.t * hit.t;
      float cos_light = fabsf(dot3(hit.geo_normal, normalize3(ps.ray.dir)));
      if (cos_light > 1e-6f && scene.num_lights > 0) {
        // Find light area (approximate: use first matching light)
        float light_pdf = 0.f;
        for (int i = 0; i < scene.num_lights; ++i) {
          if (scene.lights[i].mesh_id == (uint32_t)hit.instance_id &&
              scene.lights[i].tri_idx == (uint32_t)hit.prim_id) {
            float area = scene.lights[i].area;
            light_pdf = (1.f / (float)scene.num_lights) * (dist2 / (cos_light * area));
            break;
          }
        }
        if (light_pdf > 0.f) {
          float w = gpu_power_heuristic(ps.prev_bsdf_pdf_sa, light_pdf);
          ps.radiance += ps.throughput * mat.emission * w;
        } else {
          ps.radiance += ps.throughput * mat.emission;
        }
      } else {
        ps.radiance += ps.throughput * mat.emission;
      }
    }
  }

  // Skip NEE for delta materials
  if (mat.isDelta()) return;

  // ── Area light NEE ─────────────────────────────────────────────────────────
  if (scene.num_lights > 0) {
    float u_pick = ps.rng.next_float();
    int light_idx = (int)(u_pick * (float)scene.num_lights);
    light_idx = min(light_idx, scene.num_lights - 1);
    float light_pick_pdf = 1.f / (float)scene.num_lights;

    const GpuLight& light = scene.lights[light_idx];
    const GpuMeshInfo& lmi = scene.mesh_info[light.mesh_id];

    // Sample a point on the light triangle
    int tri = light.tri_idx;
    int base = lmi.tri_offset + tri * 3;
    int i0 = scene.indices[base]; int i1 = scene.indices[base+1]; int i2 = scene.indices[base+2];
    int vo = lmi.vertex_offset;
    float3 lp0 = make_f3(scene.vertices_x[vo+i0], scene.vertices_y[vo+i0], scene.vertices_z[vo+i0]);
    float3 lp1 = make_f3(scene.vertices_x[vo+i1], scene.vertices_y[vo+i1], scene.vertices_z[vo+i1]);
    float3 lp2 = make_f3(scene.vertices_x[vo+i2], scene.vertices_y[vo+i2], scene.vertices_z[vo+i2]);

    // Uniform triangle sampling
    float su = ps.rng.next_float();
    float sv = ps.rng.next_float();
    if (su + sv > 1.f) { su = 1.f - su; sv = 1.f - sv; }
    float3 light_pos = lp0 * (1.f - su - sv) + lp1 * su + lp2 * sv;
    float3 light_normal = normalize3(cross3(lp1 - lp0, lp2 - lp0));

    float3 to_light = light_pos - hit.pos;
    float dist2 = dot3(to_light, to_light);
    float dist = sqrtf(dist2);
    float3 wi_world = to_light / dist;

    float cos_light = fabsf(dot3(light_normal, -wi_world));
    float cos_surface = dot3(hit.geo_normal, wi_world);
    if (cos_light < 1e-6f || cos_surface < 1e-6f) return; // skip invalid geometry

    // Light PDF in solid angle
    float light_pdf_sa = light_pick_pdf * dist2 / (cos_light * light.area);

    // BSDF eval — CPU uses geo_normal for NEE ONB (cpu_tiled.cpp:114)
    GpuOnb onb(hit.geo_normal);
    float3 wo_local = onb.to_local(normalize3(-ps.ray.dir));
    float3 wi_local = onb.to_local(wi_world);
    if (wi_local.z < 1e-6f) return; // below surface

    float3 f = gpu_bsdf_eval_for_nee(wo_local, wi_local, mat);
    float bsdf_pdf = gpu_bsdf_pdf_for_nee(wo_local, wi_local, mat);

    float3 contrib = ps.throughput * f * cos_surface * light.emission / (light_pdf_sa + 1e-10f);

    // MIS weight
    float w = gpu_power_heuristic(light_pdf_sa, bsdf_pdf);
    contrib = contrib * w;

    // Only queue shadow ray if contributing
    if (max_component3(contrib) > 1e-8f) {
      int slot = atomicAdd(&shadow_counter[CNT_SHADOW], 1);
      if (slot >= scene.num_pixels) return; // bounds check
      GpuShadowWork& sw = shadow_work[slot];
      sw.path_idx = path_idx;
      sw.ray.origin = gpu_offset_ray_origin(hit.pos, hit.geo_normal, wi_world, kGpuShadowEps);
      sw.ray.dir    = wi_world;
      sw.ray.tmin   = kGpuRayEps;
      sw.ray.tmax   = dist - 2.f * kGpuShadowEps;
      sw.t_max      = dist;
      sw.contrib    = contrib;
      sw.origin_prim_id = hit.prim_id;
      sw.light_prim_id  = tri;
    }
  }

  // ── Directional light NEE ──────────────────────────────────────────────────
  for (int di = 0; di < scene.num_dir_lights; ++di) {
    const GpuDirLight& dl = scene.dir_lights[di];
    float3 wi_world = dl.direction;
    float cos_surface = dot3(hit.geo_normal, wi_world);
    if (cos_surface < 1e-6f) continue;

    // CPU uses geo_normal for NEE ONB (cpu_tiled.cpp:151)
    GpuOnb onb(hit.geo_normal);
    float3 wo_local = onb.to_local(normalize3(-ps.ray.dir));
    float3 wi_local = onb.to_local(wi_world);
    if (wi_local.z < 1e-6f) continue;

    float3 f = gpu_bsdf_eval_for_nee(wo_local, wi_local, mat);
    float bsdf_pdf = gpu_bsdf_pdf_for_nee(wo_local, wi_local, mat);
    float3 Le = dl.get_emission();

    // Directional light: near-delta solid-angle PDF (matches cpu_tiled.cpp:163-165)
    constexpr float kDirLightPdfSentinel = 1e8f;
    float dl_pick_pdf = 1.f / (float)scene.num_dir_lights;
    float light_pdf_sa = kDirLightPdfSentinel * dl_pick_pdf;
    float mis_w = gpu_power_heuristic(light_pdf_sa, bsdf_pdf);
    float3 contrib = ps.throughput * f * cos_surface * (Le / light_pdf_sa) * mis_w;

    if (max_component3(contrib) > 1e-8f) {
      int slot = atomicAdd(&shadow_counter[CNT_SHADOW], 1);
      if (slot >= scene.num_pixels) continue; // bounds check
      GpuShadowWork& sw = shadow_work[slot];
      sw.path_idx = path_idx;
      sw.ray.origin = gpu_offset_ray_origin(hit.pos, hit.geo_normal, wi_world, kGpuShadowEps);
      sw.ray.dir    = wi_world;
      sw.ray.tmin   = kGpuRayEps;
      sw.ray.tmax   = kGpuInfinity;
      sw.t_max      = kGpuInfinity;
      sw.contrib    = contrib;
    }
  }
}

// ─── Shadow ──────────────────────────────────────────────────────────────────

__global__ void shadow_kernel(
    GpuPathState*        paths,
    const GpuShadowWork* shadow_work,
    const int*           counters,
    GpuSceneData         scene)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_shadow = counters[CNT_SHADOW];
  if (tid >= num_shadow) return;

  const GpuShadowWork& sw = shadow_work[tid];
  if (!gpu_tlas_intersects(sw.ray, scene)) {
    // No occlusion — add light contribution
    paths[sw.path_idx].radiance += sw.contrib;
  }
}

// ─── Shade (unified — handles all material types) ────────────────────────────

__global__ void shade_kernel(
    GpuPathState*       paths,
    const GpuHitRecord* hits,
    const int*          counters,
    int*                ray_queue_next,
    int*                next_counter,
    GpuSceneData        scene,
    int                 max_bounces,
    int                 min_bounces)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_hits = counters[CNT_HIT];
  if (tid >= num_hits) return;

  const GpuHitRecord& hit = hits[tid];
  int path_idx = hit.prim_id;
  GpuPathState& ps = paths[path_idx];
  if (!ps.is_active()) return;

  const GpuMaterial& mat = scene.materials[hit.mat_id];

  // Skip emissive-only surfaces (already handled in NEE)
  if (mat.isEmissive() && !mat.isConductor() && !mat.isTransmissive() &&
      length_sq3(mat.baseColor) < 1e-6f) {
    ps.set_active(false);
    return;
  }

  // Max depth check
  if (ps.depth >= max_bounces) {
    ps.set_active(false);
    return;
  }

  // Russian roulette
  if (ps.depth >= min_bounces) {
    float q = fmaxf(0.05f, 1.f - max_component3(ps.throughput));
    if (ps.rng.next_float() < q) {
      ps.set_active(false);
      return;
    }
    ps.throughput = ps.throughput / (1.f - q);
  }

  // Sample BSDF
  GpuOnb onb(hit.normal);
  float3 wo_local = onb.to_local(normalize3(-ps.ray.dir));
  if (wo_local.z < 1e-6f && !mat.isTransmissive()) {
    // Below surface for non-transmissive
    ps.set_active(false);
    return;
  }

  GpuBSDFSample sample;
  if (!gpu_bsdf_sample(wo_local, mat, ps.rng, sample) || sample.pdf < 1e-10f) {
    ps.set_active(false);
    return;
  }

  // Transform sampled direction to world space
  float3 wi_world = normalize3(onb.to_world(sample.wi));

  // Update throughput
  float cos_theta = fabsf(sample.wi.z);
  if (sample.delta) {
    ps.throughput *= sample.f;
  } else {
    ps.throughput *= sample.f * cos_theta / (sample.pdf + 1e-10f);
  }

  // Firefly clamp
  float tp_max = max_component3(ps.throughput);
  if (tp_max > 100.f) {
    ps.throughput = ps.throughput * (100.f / tp_max);
  }

  // Set up next ray
  float3 ray_origin = gpu_offset_ray_origin(hit.pos, hit.geo_normal, wi_world);
  ps.ray = GpuRay{ray_origin, wi_world};
  ps.depth++;
  ps.prev_bsdf_pdf_sa = sample.pdf;
  ps.set_prev_delta(sample.delta);

  // Enqueue for next bounce
  int slot = atomicAdd(&next_counter[CNT_RAY_NEXT], 1);
  if (slot < scene.num_pixels) {  // bounds check
    ray_queue_next[slot] = path_idx;
  }
}

// ─── Accumulate ──────────────────────────────────────────────────────────────

__global__ void accumulate_kernel(
    const GpuPathState* paths,
    float*              accum,
    int                 num_pixels)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pixels) return;

  const GpuPathState& ps = paths[idx];
  // NaN / Inf guard
  float3 r = ps.radiance;
  if (isnan(r.x) || isnan(r.y) || isnan(r.z) ||
      isinf(r.x) || isinf(r.y) || isinf(r.z)) {
    return; // skip bad samples
  }

  accum[idx * 3 + 0] += r.x;
  accum[idx * 3 + 1] += r.y;
  accum[idx * 3 + 2] += r.z;
}

// ─── Tonemap + output ────────────────────────────────────────────────────────
// Output linear HDR (averaged over spp). Gamma + tonemap is handled by
// save_image() in main.cpp so GPU and CPU paths produce identical output.

__global__ void tonemap_kernel(
    const float* accum,
    float*       output,
    int          spp,
    int          num_pixels)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pixels) return;

  float inv_spp = 1.f / (float)spp;
  output[idx * 3 + 0] = accum[idx * 3 + 0] * inv_spp;
  output[idx * 3 + 1] = accum[idx * 3 + 1] * inv_spp;
  output[idx * 3 + 2] = accum[idx * 3 + 2] * inv_spp;
}

// ═════════════════════════════════════════════════════════════════════════════
// CudaRenderer Implementation
// ═════════════════════════════════════════════════════════════════════════════

CudaRenderer::CudaRenderer(int width, int height, int max_bounces, int min_bounces)
    : width_(width), height_(height), max_bounces_(max_bounces), min_bounces_(min_bounces)
{
  num_pixels_ = width_ * height_;
  alloc_buffers();
  cuda_print_device_info();
}

CudaRenderer::~CudaRenderer() {
  free_buffers();
}

void CudaRenderer::alloc_buffers() {
  CUDA_CHECK(cudaMalloc(&d_paths_,       num_pixels_ * sizeof(GpuPathState)));
  CUDA_CHECK(cudaMalloc(&d_ray_queue_,   num_pixels_ * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_ray_next_,    num_pixels_ * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_hits_,        num_pixels_ * sizeof(GpuHitRecord)));
  CUDA_CHECK(cudaMalloc(&d_shadow_work_, num_pixels_ * sizeof(GpuShadowWork)));
  CUDA_CHECK(cudaMalloc(&d_shadow_queue_, num_pixels_ * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_counters_,    NUM_COUNTERS * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_accum_,       num_pixels_ * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output_,      num_pixels_ * 3 * sizeof(float)));

  // Clear accumulation buffer
  CUDA_CHECK(cudaMemset(d_accum_, 0, num_pixels_ * 3 * sizeof(float)));

  // Host output buffer
  h_output_ = new float[num_pixels_ * 3];

  std::printf("[CUDA] Allocated renderer buffers: %.2f MB device memory\n",
              (num_pixels_ * (sizeof(GpuPathState) + sizeof(int)*3 + sizeof(GpuHitRecord)
               + sizeof(GpuShadowWork) + sizeof(int) + sizeof(float)*6)
               + NUM_COUNTERS * sizeof(int)) / (1024.0 * 1024.0));
}

void CudaRenderer::free_buffers() {
  auto safe_free = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
  safe_free(d_paths_);
  safe_free(d_ray_queue_);
  safe_free(d_ray_next_);
  safe_free(d_hits_);
  safe_free(d_shadow_work_);
  safe_free(d_shadow_queue_);
  safe_free(d_counters_);
  safe_free(d_accum_);
  safe_free(d_output_);
  delete[] h_output_; h_output_ = nullptr;
}

void CudaRenderer::upload_scene(const Scene& scene) {
  gpu_scene_.upload(scene);
}

// ─── Render one sample ───────────────────────────────────────────────────────

void CudaRenderer::render_frame(const Scene& scene, const Camera& camera, TripleSwapchain& swapchain) {
  spp_++;

  GpuCamera gpu_cam = gpu_scene_.upload_camera(camera, width_, height_);
  GpuSceneData scene_data = gpu_scene_.get_scene_data();
  scene_data.num_pixels = num_pixels_;

  // Reset counters
  CUDA_CHECK(cudaMemset(d_counters_, 0, NUM_COUNTERS * sizeof(int)));

  // ── Ray generation ─────────────────────────────────────────────────────────
  raygen_kernel<<<grid_size(num_pixels_), kBlockSize>>>(
    d_paths_, d_ray_queue_, d_counters_, gpu_cam, spp_, width_, height_);

  // ── Bounce loop ────────────────────────────────────────────────────────────
  for (int bounce = 0; bounce < max_bounces_; ++bounce) {
    // Read ray count from device
    int ray_count = 0;
    CUDA_CHECK(cudaMemcpy(&ray_count, &d_counters_[CNT_RAY], sizeof(int), cudaMemcpyDeviceToHost));
    if (ray_count == 0) break;

    // Reset hit and shadow counters
    int zeros[2] = {0, 0};
    CUDA_CHECK(cudaMemcpy(&d_counters_[CNT_HIT], zeros, 2 * sizeof(int), cudaMemcpyHostToDevice));

    // Intersect
    intersect_kernel<<<grid_size(ray_count), kBlockSize>>>(
      d_paths_, d_ray_queue_, d_counters_, d_hits_, d_counters_, scene_data);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read hit count
    int hit_count = 0;
    CUDA_CHECK(cudaMemcpy(&hit_count, &d_counters_[CNT_HIT], sizeof(int), cudaMemcpyDeviceToHost));
    if (hit_count == 0) break;

    // Reset shadow counter
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(&d_counters_[CNT_SHADOW], &zero, sizeof(int), cudaMemcpyHostToDevice));

    // NEE
    nee_kernel<<<grid_size(hit_count), kBlockSize>>>(
      d_paths_, d_hits_, d_counters_, d_shadow_work_, d_counters_, scene_data, min_bounces_);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Shadow rays
    int shadow_count = 0;
    CUDA_CHECK(cudaMemcpy(&shadow_count, &d_counters_[CNT_SHADOW], sizeof(int), cudaMemcpyDeviceToHost));
    if (shadow_count > 0) {
      shadow_kernel<<<grid_size(shadow_count), kBlockSize>>>(
        d_paths_, d_shadow_work_, d_counters_, scene_data);
    }

    // Reset next ray counter
    CUDA_CHECK(cudaMemcpy(&d_counters_[CNT_RAY_NEXT], &zero, sizeof(int), cudaMemcpyHostToDevice));

    // Shade (all materials in one kernel)
    shade_kernel<<<grid_size(hit_count), kBlockSize>>>(
      d_paths_, d_hits_, d_counters_, d_ray_next_, d_counters_,
      scene_data, max_bounces_, min_bounces_);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read next ray count, swap queues
    int next_ray_count = 0;
    CUDA_CHECK(cudaMemcpy(&next_ray_count, &d_counters_[CNT_RAY_NEXT], sizeof(int), cudaMemcpyDeviceToHost));

    // Swap ray queues
    int* tmp = d_ray_queue_;
    d_ray_queue_ = d_ray_next_;
    d_ray_next_ = tmp;

    // Set ray count for next bounce
    CUDA_CHECK(cudaMemcpy(&d_counters_[CNT_RAY], &next_ray_count, sizeof(int), cudaMemcpyHostToDevice));
  }

  // ── Accumulate ─────────────────────────────────────────────────────────────
  accumulate_kernel<<<grid_size(num_pixels_), kBlockSize>>>(
    d_paths_, d_accum_, num_pixels_);

  // ── Tonemap + output ───────────────────────────────────────────────────────
  tonemap_kernel<<<grid_size(num_pixels_), kBlockSize>>>(
    d_accum_, d_output_, spp_, num_pixels_);
  CUDA_CHECK(cudaDeviceSynchronize());

  // ── Copy back to host swapchain ────────────────────────────────────────────
  CUDA_CHECK(cudaMemcpy(h_output_, d_output_, num_pixels_ * 3 * sizeof(float), cudaMemcpyDeviceToHost));

  float* write_buf = swapchain.get_write_buffer();
  std::memcpy(write_buf, h_output_, num_pixels_ * 3 * sizeof(float));
  swapchain.swap_writer();
}

} // namespace xn
