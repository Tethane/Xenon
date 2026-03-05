#include "render/wavefront.h"
#include "camera/sampler.h"
#include "math/simd.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace xn {

WavefrontRenderer::WavefrontRenderer(int width, int height) : width_(width), height_(height) {
  accumulation_buffer_.resize(width * height, Vec3(0.f));
  pool_ = new ThreadPool(std::thread::hardware_concurrency());
}

WavefrontRenderer::~WavefrontRenderer() {
  delete pool_;
}

void raygen(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<RayWorkItem>& q_rays, const Camera& camera, int spp, int height, int width) {
  int grain = 2048;
  int num_paths = paths.size();
  int tile_size = 32;

  pool.parallel_for(num_paths, grain, [&](int begin, int end) {
    thread_local std::vector<RayWorkItem> local_rays;
    local_rays.clear();
    local_rays.reserve(end - begin);

    for (int i = begin; i < end; ++i) {
      int x = i % tile_size;
      int y = i / tile_size;

      uint64_t seed = (uint64_t)y * tile_size + x + (uint64_t)spp * 0xdeadbeef;
      paths[i].rng = seed_pcg(seed);

      float u = (float)x + paths[i].rng.next_float();
      float v = (float)y + paths[i].rng.next_float();

      paths[i].ray = camera.get_ray(u / (float)width, ((float)height - v) / (float)height);
      paths[i].throughput = Vec3(1.f);
      paths[i].pixel_idx = i;

      local_rays.push_back({i});
    }

    flush_local_to_queue(q_rays, local_rays);
  });
}

void closest_hit(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<RayWorkItem>& q_rays, WavefrontQueue<HitWorkItem>& q_hits, const Scene& scene) {
  q_hits.size.store(0, std::memory_order_relaxed);

  int num_rays = q_rays.size.load(std::memory_order_relaxed);
  int grain = 512;

  pool.parallel_for(num_rays, grain, [&](int begin, int end) {
    thread_local std::vector<HitWorkItem> local_hits;
    local_hits.clear();
    local_hits.reserve(end - begin);

    for (int i = begin; i < end; ++i) {
      int p_idx = q_rays.items[i].path_idx;
      HitRecord rec;
      if (scene.intersect(paths[p_idx].ray, rec)) {
        local_hits.push_back({p_idx, rec});
      } else {
        paths[p_idx].active = false;
      }
    }

    flush_local_to_queue(q_hits, local_hits);
  });
}

void next_event_estimation(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q_hits, WavefrontQueue<ShadowWorkItem>& q_shadow, const Scene& scene, int min_bounces) {
  q_shadow.size.store(0, std::memory_order_relaxed);

  int num_hits = q_hits.size.load(std::memory_order_relaxed);
  int grain = 512;

  pool.parallel_for(num_hits, grain, [&](int begin, int end) {
    thread_local std::vector<ShadowWorkItem> local_shadow;
    local_shadow.clear();

    for (int i = begin; i < end; ++i) {
      const auto& work = q_hits.items[i];
      PathState& path = paths[work.path_idx];
      const HitRecord& rec = work.hit;
      const PrincipledBSDF& mat = scene.materials[rec.mat_id];

      // Filter Emissive and Adjust MIS
      bool hit_light = false;
      for (const auto& light : scene.lights) {
        if (light.tri_idx == (uint32_t)rec.prim_id) {
          float mis_w = 1.0f;
          if (!path.prev_was_delta) {
            float light_pdf_area = 1.0f / (light.area * (float)scene.lights.size());
            float dist_sq = rec.t * rec.t;
            float cos_l = std::abs(dot(rec.geo_normal, -path.ray.dir));
            if (cos_l > 1e-6f) {
              float light_pdf_sa = (light_pdf_area * dist_sq) / cos_l;
              mis_w = mis_weight_power2(path.prev_bsdf_pdf_sa, light_pdf_sa);
            } else {
              mis_w = 0.0f;
            }
          }
          path.radiance += path.throughput * light.emission * mis_w;
          hit_light = true;
          break;
        }
      }

      if(hit_light) {
        path.active = false;
        continue;
      }

      // NEE
      if (!scene.lights.empty()) {
        float pick_pdf;
        const Light& l = scene.sample_light(path.rng.next_float(), pick_pdf);
        const auto& mesh = scene.meshes[l.mesh_id];
        Vec3 p[3];
        mesh.get_triangle(l.tri_idx, p[0], p[1], p[2]);

        float u1 = path.rng.next_float();
        float u2 = path.rng.next_float();
        float su1 = std::sqrt(u1);
        float b0 = 1.0f - su1;
        float b1 = u2 * su1;
        float b2 = 1.0f - b0 - b1;
        Vec3 light_pos = p[0] * b0 + p[1] * b1 + p[2] * b2;

        Vec3 light_norm = mesh.geo_normal(l.tri_idx);

        Vec3 shadow_dir = light_pos - rec.pos;
        float dist_sq = shadow_dir.length_sq();
        float dist = std::sqrt(dist_sq);
        Vec3 shadow_wi = shadow_dir / dist;
        float cos_l = std::abs(dot(light_norm, -shadow_wi));

        Onb onb(rec.normal);
        Vec3 wo_local = onb.to_local(-path.ray.dir);
        Vec3 wi_local = onb.to_local(shadow_wi);

        if (cos_l > 1e-6f && wi_local.z > 1e-6f) {
          Vec3 f = bsdf_eval(wo_local, wi_local, mat);
          float b_pdf = bsdf_pdf(wo_local, wi_local, mat);
          float light_pdf_sa = (1.0f / l.area) * (dist_sq / cos_l) * pick_pdf; 
          float mis_w = mis_weight_power2(light_pdf_sa, b_pdf);
          float cos_s = std::abs(wi_local.z);

          Vec3 contrib = path.throughput * f * cos_s * (l.emission / light_pdf_sa) *mis_w;

          Vec3 origin = offset_ray_origin(rec.pos, rec.geo_normal, shadow_wi, kShadowEps);

          Ray shadow_ray{origin, shadow_wi, kShadowEps, dist - kShadowEps};
          local_shadow.push_back({
            work.path_idx,
            shadow_ray,
            shadow_ray.tmax,
            contrib,
            rec.prim_id,
            (int)l.tri_idx
          });
        }
      }

      // Russian Roullette
      if (path.depth >= min_bounces) {
        float p = std::max(0.05f, max_component(path.throughput));
        if (path.rng.next_float() > p) {
          path.active = false;
          continue;
        }
        path.throughput /= p;
      }
    }

    flush_local_to_queue(q_shadow, local_shadow);
  });
}

void classification(ThreadPool& pool, WavefrontQueue<HitWorkItem>& q_hits, WavefrontQueue<HitWorkItem>& q_hit_diffuse, WavefrontQueue<HitWorkItem>& q_hit_glossy_refl, WavefrontQueue<HitWorkItem>& q_hit_glossy_trans, WavefrontQueue<HitWorkItem>& q_hit_delta, const Scene& scene) {
  q_hit_diffuse.size.store(0, std::memory_order_relaxed);
  q_hit_glossy_refl.size.store(0, std::memory_order_relaxed);
  q_hit_glossy_trans.size.store(0, std::memory_order_relaxed);
  q_hit_delta.size.store(0, std::memory_order_relaxed);

  int num_hits = q_hits.size.load(std::memory_order_relaxed);
  int grain = 2048;

  pool.parallel_for(num_hits, grain, [&](int begin, int end) {
    thread_local std::vector<HitWorkItem> ld;
    thread_local std::vector<HitWorkItem> lgr;
    thread_local std::vector<HitWorkItem> lgt;
    thread_local std::vector<HitWorkItem> ldel;

    ld.clear(); ld.reserve(end - begin);
    lgr.clear(); lgr.reserve(end - begin);
    lgt.clear(); lgt.reserve(end - begin);
    ldel.clear(); ldel.reserve(end - begin);

    for (int i = begin; i < end; ++i) {
      const HitWorkItem& work = q_hits.items[i];
      const PrincipledBSDF& mat = scene.materials[work.hit.mat_id];

      float alpha = mat.roughness * mat.roughness;
      bool is_delta = (alpha < 0.005f);

      if (mat.transmission > 0.1f) {
        (is_delta ? ldel : lgt).push_back(work);
      } else if (mat.metallic  > 0.1f || alpha < 0.1f) {
        (is_delta ? ldel : lgr).push_back(work);
      } else {
        ld.push_back(work);
      }
    }

    flush_local_to_queue(q_hit_diffuse, ld);
    flush_local_to_queue(q_hit_glossy_refl, lgr);
    flush_local_to_queue(q_hit_glossy_trans, lgt);
    flush_local_to_queue(q_hit_delta, ldel);
  });
}

void shade_diffuse(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q_hits_diffuse, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene) {
  int num_hits = q_hits_diffuse.size.load(std::memory_order_relaxed);
  if (num_hits == 0) return;

  int grain = 1024;

  pool.parallel_for(num_hits, grain, [&](int begin, int end) {
    thread_local std::vector<RayWorkItem> local_rays_next;
    local_rays_next.clear();
    local_rays_next.reserve(end - begin);

    for (int i = begin; i < end; ++i) {
      const auto& work = q_hits_diffuse.items[i];
      PathState& path = paths[work.path_idx];
      const HitRecord& rec = work.hit;
      const PrincipledBSDF& mat = scene.materials[rec.mat_id];

      // BSDF Sampling (TODO: Fix for only diffuse)
      Onb onb(rec.normal);
      Vec3 wo_local = onb.to_local(-path.ray.dir);
      BSDFSample sample;
      if (!bsdf_sample(wo_local, mat, path.rng, sample)) {
        path.active = false;
        continue;
      }

      path.throughput *= sample.f * std::abs(sample.wi.z) / sample.pdf;

      Vec3 wi_world = onb.to_world(sample.wi);
      Vec3 origin = offset_ray_origin(rec.pos, rec.geo_normal, wi_world, kRayEps);

      path.ray = Ray{origin, wi_world};
      path.prev_bsdf_pdf_sa = sample.pdf;
      path.prev_was_delta = false;
      path.depth++;

      local_rays_next.push_back({work.path_idx});
    }

    flush_local_to_queue(q_rays_next, local_rays_next);
  });
}

void shade_glossy_refl(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q_hits_glossy_refl, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene) {
  int num_hits = q_hits_glossy_refl.size.load(std::memory_order_relaxed);
  if (num_hits == 0) return;

  int grain = 1024;

  pool.parallel_for(num_hits, grain, [&](int begin, int end) {
    thread_local std::vector<RayWorkItem> local_rays_next;
    local_rays_next.clear();
    local_rays_next.reserve(end - begin);

    for (int i = begin; i < end; ++i) {
      const auto& work = q_hits_glossy_refl.items[i];
      PathState& path = paths[work.path_idx];
      const HitRecord& rec = work.hit;
      const PrincipledBSDF& mat = scene.materials[rec.mat_id];

      // BSDF Sampling (TODO: Fix for only glossy refl)
      Onb onb(rec.normal);
      Vec3 wo_local = onb.to_local(-path.ray.dir);
      BSDFSample sample;
      if (!bsdf_sample(wo_local, mat, path.rng, sample)) {
        path.active = false;
        continue;
      }

      path.throughput *= sample.f * std::abs(sample.wi.z) / sample.pdf;

      Vec3 wi_world = onb.to_world(sample.wi);
      Vec3 origin = offset_ray_origin(rec.pos, rec.geo_normal, wi_world, kRayEps);

      path.ray = Ray{origin, wi_world};
      path.prev_bsdf_pdf_sa = sample.pdf;
      path.prev_was_delta = false;
      path.depth++;

      local_rays_next.push_back({work.path_idx});
    }

    flush_local_to_queue(q_rays_next, local_rays_next);
  });
}

void shade_glossy_trans(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q_hits_glossy_trans, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene) {
  
  int num_hits = q_hits_glossy_trans.size.load(std::memory_order_relaxed);
  if (num_hits == 0) return;

  int grain = 1024;

  pool.parallel_for(num_hits, grain, [&](int begin, int end) {
    thread_local std::vector<RayWorkItem> local_rays_next;
    local_rays_next.clear();
    local_rays_next.reserve(end - begin);

    for (int i = begin; i < end; ++i) {
      const auto& work = q_hits_glossy_trans.items[i];
      PathState& path = paths[work.path_idx];
      const HitRecord& rec = work.hit;
      const PrincipledBSDF& mat = scene.materials[rec.mat_id];

      // BSDF Sampling (TODO: Fix for only glossy trans)
      Onb onb(rec.normal);
      Vec3 wo_local = onb.to_local(-path.ray.dir);
      BSDFSample sample;
      if (!bsdf_sample(wo_local, mat, path.rng, sample)) {
        path.active = false;
        continue;
      }

      path.throughput *= sample.f * std::abs(sample.wi.z) / sample.pdf;

      Vec3 wi_world = onb.to_world(sample.wi);
      Vec3 origin = offset_ray_origin(rec.pos, rec.geo_normal, wi_world, kRayEps);

      path.ray = Ray{origin, wi_world};
      path.prev_bsdf_pdf_sa = sample.pdf;
      path.prev_was_delta = false;
      path.depth++;

      local_rays_next.push_back({work.path_idx});
    }

    flush_local_to_queue(q_rays_next, local_rays_next);
  });
}

void shade_delta(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q_hits_delta, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene) {
  
  int num_hits = q_hits_delta.size.load(std::memory_order_relaxed);
  if (num_hits == 0) return;

  int grain = 1024;

  pool.parallel_for(num_hits, grain, [&](int begin, int end) {
    thread_local std::vector<RayWorkItem> local_rays_next;
    local_rays_next.clear();
    local_rays_next.reserve(end - begin);

    for (int i = begin; i < end; ++i) {
      const auto& work = q_hits_delta.items[i];
      PathState& path = paths[work.path_idx];
      const HitRecord& rec = work.hit;
      const PrincipledBSDF& mat = scene.materials[rec.mat_id];

      // BSDF Sampling (TODO: Fix for only delta)
      Onb onb(rec.normal);
      Vec3 wo_local = onb.to_local(-path.ray.dir);
      BSDFSample sample;
      if (!bsdf_sample(wo_local, mat, path.rng, sample)) {
        path.active = false;
        continue;
      }

      path.throughput *= sample.f;

      Vec3 wi_world = onb.to_world(sample.wi);
      Vec3 origin = offset_ray_origin(rec.pos, rec.geo_normal, wi_world, kRayEps);

      path.ray = Ray{origin, wi_world};
      path.prev_bsdf_pdf_sa = sample.pdf;
      path.prev_was_delta = true;
      path.depth++;

      local_rays_next.push_back({work.path_idx});
    }

    flush_local_to_queue(q_rays_next, local_rays_next);
  });
}

void shadow(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<ShadowWorkItem>& q_shadow, const Scene& scene) {
  int num_shadows = q_shadow.size.load(std::memory_order_relaxed);
  int grain = 4096;

  pool.parallel_for(num_shadows, grain, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      const auto& work = q_shadow.items[i];
      if (!scene.intersects(work.ray)) {
        paths[work.path_idx].radiance += work.contrib;
      }
    }
  });
}

// Renders the entire frame at once, no tiling (ports easily to GPU wavefront)
// Wavefront kernels handle all of the parallelization. Much more SIMD and GPU friendly.
// Raygen -> 
// For each bounce
//    Classification ->
//    Diffuse, Glossy Refl, Glossy Trans, Delta
//    Shadow
//    Accumulate
//    Swap Queues
// Copy Accumulation to Writer Buffer
// Swap Writer

void WavefrontRenderer::render_frame(const Scene &scene, const Camera &camera, TripleSwapchain &swapchain) {
  spp_++;

  int num_paths = accumulation_buffer_.size();

  std::vector<PathState> paths(num_paths);

  WavefrontQueue<RayWorkItem> q_rays;
  WavefrontQueue<RayWorkItem> q_rays_next;
  WavefrontQueue<HitWorkItem> q_hits;
  WavefrontQueue<ShadowWorkItem> q_shadow;

  WavefrontQueue<HitWorkItem> q_hit_diffuse;
  WavefrontQueue<HitWorkItem> q_hit_glossy_refl;
  WavefrontQueue<HitWorkItem> q_hit_glossy_trans;
  WavefrontQueue<HitWorkItem> q_hit_delta;

  q_rays.reset(num_paths);
  q_rays_next.reset(num_paths);
  q_hits.reset(num_paths);
  q_shadow.reset(num_paths);

  q_hit_diffuse.reset(num_paths);
  q_hit_glossy_refl.reset(num_paths);
  q_hit_glossy_trans.reset(num_paths);
  q_hit_delta.reset(num_paths);

  // Raygen Kernel
  raygen(*pool_, paths, q_rays, camera, spp_, height_, width_);

  // Rendering Loop
  for (int bounce = 0; bounce < max_bounces_; ++bounce) {
    closest_hit(*pool_, paths, q_rays, q_hits, scene);
    next_event_estimation(*pool_, paths, q_hits, q_shadow, scene, min_bounces_);
    classification(*pool_, q_hits, q_hit_diffuse, q_hit_glossy_refl, q_hit_glossy_trans, q_hit_delta, scene);
    shade_diffuse(*pool_, paths, q_hit_diffuse, q_rays_next, scene);
    shade_glossy_refl(*pool_, paths, q_hit_glossy_refl, q_rays_next, scene);
    shade_delta(*pool_, paths, q_hit_glossy_refl, q_rays_next, scene);
    shadow(*pool_, paths, q_shadow, scene);

    std::swap(q_rays.items, q_rays_next.items);
    q_rays.size.store(q_rays_next.size.load());
    q_rays_next.size.store(0);
  }

  for (int i = 0; i < num_paths; ++i) {
    accumulation_buffer_[paths[i].pixel_idx] += paths[i].radiance;
  }
}

} // namespace xn
