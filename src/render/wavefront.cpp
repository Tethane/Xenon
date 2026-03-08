// render/wavefront.cpp — Wavefront CPU path integrator (staged kernel pipeline)

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>

#include "render/wavefront.h"
#include "material/bsdf.h"
#include "camera/sampler.h"
#include "math/simd.h"

#ifdef XN_DEBUG_QUEUES
#include <cstdio>
#endif

namespace xn {

WavefrontRenderer::WavefrontRenderer(int width, int height) : width_(width), height_(height) {
  accumulation_buffer_.resize(width * height, Vec3(0.f));
  pool_ = new ThreadPool(std::thread::hardware_concurrency());
}

WavefrontRenderer::~WavefrontRenderer() {
  delete pool_;
}

void raygen(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<RayWorkItem>& q_rays, const Camera& camera, int spp, int height, int width) {
  int grain = 32768;
  int num_paths = paths.size();

  pool.parallel_for(num_paths, grain, [&](int begin, int end) {
    thread_local std::vector<RayWorkItem> local_rays;
    local_rays.clear();
    local_rays.reserve(end - begin);

    for (int i = begin; i < end; ++i) {
      int x = i % width;
      int y = i / width;

      uint64_t seed = (uint64_t)i + (uint64_t)spp * 0xdeadbeef;
      paths[i].rng = seed_pcg(seed);

      float u = (float)x + paths[i].rng.next_float();
      float v = (float)y + paths[i].rng.next_float();

      paths[i].ray = camera.get_ray(u / (float)width, v / (float)height);
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
  int grain = 32768;

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

// ═════════════════════════════════════════════════════════════════════════════
// NEE — Next Event Estimation
// Uses bsdf_eval_for_nee / bsdf_pdf_for_nee for correct MIS with Material
// ═════════════════════════════════════════════════════════════════════════════

void next_event_estimation(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q_hits, WavefrontQueue<ShadowWorkItem>& q_shadow, const Scene& scene, int min_bounces) {
  q_shadow.size.store(0, std::memory_order_relaxed);

  int num_hits = q_hits.size.load(std::memory_order_relaxed);
  int grain = 16384;

  pool.parallel_for(num_hits, grain, [&](int begin, int end) {
    thread_local std::vector<ShadowWorkItem> local_shadow;
    local_shadow.clear();
    local_shadow.reserve(end - begin);

    for (int i = begin; i < end; ++i) {
      const auto& work = q_hits.items[i];
      PathState& path = paths[work.path_idx];
      const HitRecord& rec = work.hit;
      const Material& mat = scene.materials[rec.mat_id];

      // ── Emissive hit check + MIS ──────────────────────────────────────
      bool hit_light = false;
      for (const auto& light : scene.lights) {
        if (light.tri_idx == (uint32_t)rec.prim_id) {
          float mis_w = 1.0f;
          if (path.depth > 0 && !path.prev_was_delta) {
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

      if (hit_light) {
        path.active = false;
        continue;
      }

      // ── NEE: Sample a light ───────────────────────────────────────────
      if (!scene.lights.empty() && !mat.isDelta) {
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
        Vec3 light_pos = p[0] * b0 + p[1] * b1 + p[2] * (1.0f - b0 - b1);

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
          // Use unified NEE eval/pdf from bsdf.h
          Vec3 f = bsdf_eval_for_nee(wo_local, wi_local, mat);
          float b_pdf = bsdf_pdf_for_nee(wo_local, wi_local, mat);
          float light_pdf_sa = (1.0f / l.area) * (dist_sq / cos_l) * pick_pdf;
          float mis_w = mis_weight_power2(light_pdf_sa, b_pdf);
          float cos_s = std::abs(wi_local.z);

          Vec3 contrib = path.throughput * f * cos_s * (l.emission / light_pdf_sa) * mis_w;

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

      // ── Russian Roulette ──────────────────────────────────────────────
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

// ═════════════════════════════════════════════════════════════════════════════
// Classification — selects exactly ONE lobe per hit, enqueues to proper queue
// All material-type branching happens here, NOT in shade kernels.
// ═════════════════════════════════════════════════════════════════════════════

void classification(ThreadPool& pool, std::vector<PathState>& paths,
    WavefrontQueue<HitWorkItem>& q_hits,
    WavefrontQueue<HitWorkItem>& q_diffuse,
    WavefrontQueue<HitWorkItem>& q_microfacet_refl,
    WavefrontQueue<HitWorkItem>& q_microfacet_trans,
    WavefrontQueue<HitWorkItem>& q_delta_refl,
    WavefrontQueue<HitWorkItem>& q_delta_trans,
    WavefrontQueue<HitWorkItem>& q_subsurface,
    const Scene& scene) {

  q_diffuse.size.store(0, std::memory_order_relaxed);
  q_microfacet_refl.size.store(0, std::memory_order_relaxed);
  q_microfacet_trans.size.store(0, std::memory_order_relaxed);
  q_delta_refl.size.store(0, std::memory_order_relaxed);
  q_delta_trans.size.store(0, std::memory_order_relaxed);
  q_subsurface.size.store(0, std::memory_order_relaxed);

  int num_hits = q_hits.size.load(std::memory_order_relaxed);
  int grain = 32768;

  pool.parallel_for(num_hits, grain, [&](int begin, int end) {
    thread_local std::vector<HitWorkItem> l_diff, l_mrefl, l_mtrans,
                                          l_drefl, l_dtrans, l_sub;
    l_diff.clear();   l_mrefl.clear();  l_mtrans.clear();
    l_drefl.clear();  l_dtrans.clear(); l_sub.clear();

    int cap = end - begin;
    l_diff.reserve(cap);  l_mrefl.reserve(cap);  l_mtrans.reserve(cap);
    l_drefl.reserve(cap); l_dtrans.reserve(cap); l_sub.reserve(cap);

    for (int i = begin; i < end; ++i) {
      const HitWorkItem& work = q_hits.items[i];
      const Material& mat = scene.materials[work.hit.mat_id];
      PathState& path = paths[work.path_idx];

      if (!path.active) continue;

      // ── Route based on material flags ──────────────────────────────────
      if (mat.isDelta) {
        // Delta materials
        if (mat.isTransmissive) {
          // Fresnel coin-flip: reflect vs transmit
          float F = fresnel_dielectric(path.ray.dir.z, mat.ior);
          float u = path.rng.next_float();
          if (u < F) {
            l_drefl.push_back(work);
          } else {
            l_dtrans.push_back(work);
          }
        } else {
          l_drefl.push_back(work);
        }
      } else if (mat.isConductor) {
        // Conductors: always microfacet reflection
        l_mrefl.push_back(work);
      } else if (mat.hasSubsurface) {
        // Subsurface scattering
        l_sub.push_back(work);
      } else if (mat.isTransmissive) {
        // Transmissive dielectric: Fresnel-based split
        float cos_o = std::abs(path.ray.dir.z);
        // For non-normal directions, use the actual angle
        Onb onb(work.hit.normal);
        Vec3 wo_local = onb.to_local(-path.ray.dir);
        float F = fresnel_dielectric(std::abs(wo_local.z), mat.ior);
        float u = path.rng.next_float();
        if (u < F) {
          l_mrefl.push_back(work);
        } else {
          l_mtrans.push_back(work);
        }
        (void)cos_o;
      } else {
        // Opaque dielectric: weight between diffuse and microfacet reflection
        float spec_weight = mat.F0.x; // scalar approximation
        float u = path.rng.next_float();
        if (u < spec_weight) {
          l_mrefl.push_back(work);
        } else {
          l_diff.push_back(work);
        }
      }
    }

    flush_local_to_queue(q_diffuse, l_diff);
    flush_local_to_queue(q_microfacet_refl, l_mrefl);
    flush_local_to_queue(q_microfacet_trans, l_mtrans);
    flush_local_to_queue(q_delta_refl, l_drefl);
    flush_local_to_queue(q_delta_trans, l_dtrans);
    flush_local_to_queue(q_subsurface, l_sub);
  });
}

// ═════════════════════════════════════════════════════════════════════════════
// Shade Kernels — each calls ONLY its queue's sample(). No branching.
// ═════════════════════════════════════════════════════════════════════════════

// Helper macro for the common shade-kernel pattern (non-delta)
#define SHADE_KERNEL_BODY(BSDF_NS, IS_DELTA_FLAG)                                         \
  int num_hits = q.size.load(std::memory_order_relaxed);                                   \
  if (num_hits == 0) return;                                                               \
  int grain = 16384;                                                                        \
  pool.parallel_for(num_hits, grain, [&](int begin, int end) {                             \
    thread_local std::vector<RayWorkItem> local_rays_next;                                 \
    local_rays_next.clear();                                                               \
    local_rays_next.reserve(end - begin);                                                  \
    for (int i = begin; i < end; ++i) {                                                    \
      const auto& work = q.items[i];                                                       \
      PathState& path = paths[work.path_idx];                                              \
      const HitRecord& rec = work.hit;                                                     \
      const Material& mat = scene.materials[rec.mat_id];                                   \
      Onb onb(rec.normal);                                                                 \
      Vec3 wo_local = onb.to_local(-path.ray.dir);                                        \
      BSDFSample sample;                                                                   \
      if (!BSDF_NS::sample(wo_local, mat, path.rng, sample)) {                            \
        path.active = false;                                                               \
        continue;                                                                          \
      }                                                                                    \
      if constexpr (IS_DELTA_FLAG) {                                                       \
        path.throughput *= sample.f;                                                       \
      } else {                                                                             \
        float cos_i = std::abs(sample.wi.z);                                               \
        if (sample.pdf < 1e-8f || cos_i < 1e-8f) { path.active = false; continue; }       \
        path.throughput *= sample.f * cos_i / sample.pdf;                                  \
      }                                                                                    \
      Vec3 wi_world = onb.to_world(sample.wi);                                            \
      Vec3 origin = offset_ray_origin(rec.pos, rec.geo_normal, wi_world, kRayEps);         \
      path.ray = Ray{origin, wi_world};                                                    \
      path.prev_bsdf_pdf_sa = sample.pdf;                                                  \
      path.prev_was_delta = IS_DELTA_FLAG;                                                 \
      path.depth++;                                                                        \
      local_rays_next.push_back({work.path_idx});                                          \
    }                                                                                      \
    flush_local_to_queue(q_rays_next, local_rays_next);                                    \
  });

void shade_diffuse(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene) {
  SHADE_KERNEL_BODY(diffuse_bsdf, false)
}

void shade_microfacet_refl(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene) {
  SHADE_KERNEL_BODY(microfacet_refl_bsdf, false)
}

void shade_microfacet_trans(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene) {
  SHADE_KERNEL_BODY(microfacet_trans_bsdf, false)
}

void shade_delta_refl(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene) {
  SHADE_KERNEL_BODY(delta_refl_bsdf, true)
}

void shade_delta_trans(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene) {
  SHADE_KERNEL_BODY(delta_trans_bsdf, true)
}

void shade_subsurface(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene) {
  SHADE_KERNEL_BODY(subsurface_bsdf, false)
}

#undef SHADE_KERNEL_BODY

// ═════════════════════════════════════════════════════════════════════════════
// Shadow ray kernel
// ═════════════════════════════════════════════════════════════════════════════

void shadow(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<ShadowWorkItem>& q_shadow, const Scene& scene) {
  int num_shadows = q_shadow.size.load(std::memory_order_relaxed);
  int grain = 16384;

  pool.parallel_for(num_shadows, grain, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      const auto& work = q_shadow.items[i];
      if (!scene.intersects(work.ray)) {
        paths[work.path_idx].radiance += work.contrib;
      }
    }
  });
}

// ═════════════════════════════════════════════════════════════════════════════
// Render Frame — full wavefront pipeline
// ═════════════════════════════════════════════════════════════════════════════

void WavefrontRenderer::render_frame(const Scene &scene, const Camera &camera, TripleSwapchain &swapchain) {
  spp_++;

  int num_paths = accumulation_buffer_.size();

  std::vector<PathState> paths(num_paths);

  WavefrontQueue<RayWorkItem> q_rays;
  WavefrontQueue<RayWorkItem> q_rays_next;
  WavefrontQueue<HitWorkItem> q_hits;
  WavefrontQueue<ShadowWorkItem> q_shadow;

  // 6 classification queues
  WavefrontQueue<HitWorkItem> q_diffuse;
  WavefrontQueue<HitWorkItem> q_microfacet_refl;
  WavefrontQueue<HitWorkItem> q_microfacet_trans;
  WavefrontQueue<HitWorkItem> q_delta_refl;
  WavefrontQueue<HitWorkItem> q_delta_trans;
  WavefrontQueue<HitWorkItem> q_subsurface;

  q_rays.reset(num_paths);
  q_rays_next.reset(num_paths);
  q_hits.reset(num_paths);
  q_shadow.reset(num_paths);

  q_diffuse.reset(num_paths);
  q_microfacet_refl.reset(num_paths);
  q_microfacet_trans.reset(num_paths);
  q_delta_refl.reset(num_paths);
  q_delta_trans.reset(num_paths);
  q_subsurface.reset(num_paths);

  // Raygen Kernel
  raygen(*pool_, paths, q_rays, camera, spp_, height_, width_);

#ifdef XN_DEBUG_QUEUES
  if (q_rays.size.load() > 0 && spp_ == 10) {
    std::fprintf(stderr, "After Raygen\n");
    std::fprintf(stderr, "Q_Rays: %d, Num Paths: %d\n", q_rays.size.load(), num_paths);
  }
#endif

  // Rendering Loop
  for (int bounce = 0; bounce < max_bounces_; ++bounce) {

#ifdef XN_DEBUG_QUEUES
    if (bounce % 2 == 0 && spp_ == 10) {
      std::fprintf(stderr, "===========\n");
      std::fprintf(stderr, "New Bounce\n");
      std::fprintf(stderr, "===========\n");
      std::fprintf(stderr, "Before Closest Hit. Bounce %d\n", bounce);
      std::fprintf(stderr, "Q_Rays: %d\n", q_rays.size.load());
      std::fprintf(stderr, "Q_Hits: %d\n", q_hits.size.load());
    }
#endif

    closest_hit(*pool_, paths, q_rays, q_hits, scene);

#ifdef XN_DEBUG_QUEUES
    if (bounce % 2 == 0 && spp_ == 10) {
      std::fprintf(stderr, "After Closest Hit. Bounce %d\n", bounce);
      std::fprintf(stderr, "Q_Rays: %d\n", q_rays.size.load());
      std::fprintf(stderr, "Q_Hits: %d\n", q_hits.size.load());
    }
#endif

    next_event_estimation(*pool_, paths, q_hits, q_shadow, scene, min_bounces_);

#ifdef XN_DEBUG_QUEUES
    if (bounce % 2 == 0 && spp_ == 10) {
      std::fprintf(stderr, "After NEE. Bounce %d\n", bounce);
      std::fprintf(stderr, "Q_Rays: %d\n", q_rays.size.load());
      std::fprintf(stderr, "Q_Hits: %d\n", q_hits.size.load());
      std::fprintf(stderr, "Q_Shadow: %d\n", q_shadow.size.load());
    }
#endif

    classification(*pool_, paths, q_hits,
                   q_diffuse, q_microfacet_refl, q_microfacet_trans,
                   q_delta_refl, q_delta_trans, q_subsurface, scene);

#ifdef XN_DEBUG_QUEUES
    if (bounce % 2 == 0 && spp_ == 10) {
      std::fprintf(stderr, "After Classification. Bounce %d\n", bounce);
      std::fprintf(stderr, "Q_Rays: %d\n", q_rays.size.load());
      std::fprintf(stderr, "Q_Hits: %d\n", q_hits.size.load());
      std::fprintf(stderr, "Q_Diffuse: %d\n", q_diffuse.size.load());
      std::fprintf(stderr, "Q_Microfacet_Refl: %d\n", q_microfacet_refl.size.load());
      std::fprintf(stderr, "Q_Microfacet_Trans: %d\n", q_microfacet_trans.size.load());
      std::fprintf(stderr, "Q_Delta_Refl: %d\n", q_delta_refl.size.load());
      std::fprintf(stderr, "Q_Delta_Trans: %d\n", q_delta_trans.size.load());
      std::fprintf(stderr, "Q_Subsurface: %d\n", q_subsurface.size.load());
    }
#endif

    shade_diffuse(*pool_, paths, q_diffuse, q_rays_next, scene);
    shade_microfacet_refl(*pool_, paths, q_microfacet_refl, q_rays_next, scene);
    shade_microfacet_trans(*pool_, paths, q_microfacet_trans, q_rays_next, scene);
    shade_delta_refl(*pool_, paths, q_delta_refl, q_rays_next, scene);
    shade_delta_trans(*pool_, paths, q_delta_trans, q_rays_next, scene);
    shade_subsurface(*pool_, paths, q_subsurface, q_rays_next, scene);
    shadow(*pool_, paths, q_shadow, scene);


#ifdef XN_DEBUG_QUEUES
    if (bounce % 2 == 0 && spp_ == 10) {
      std::fprintf(stderr, "After Shading and Shadow. Bounce %d\n", bounce);
      std::fprintf(stderr, "Q_Rays: %d\n", q_rays.size.load());
      std::fprintf(stderr, "Q_Rays_Next: %d\n", q_rays_next.size.load());
    }
#endif

    std::swap(q_rays.items, q_rays_next.items);
    q_rays.size.store(q_rays_next.size.load());
    q_rays_next.size.store(0);
  }

  for (int i = 0; i < num_paths; ++i) {
    accumulation_buffer_[paths[i].pixel_idx] += paths[i].radiance;
  }

  float* out = swapchain.get_write_buffer();
  float inv_spp = 1.0f / (float)spp_;
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int src_idx = y * width_ + x;
      int dst_idx = (height_ - 1 - y) * width_ + x;
      Vec3 c = accumulation_buffer_[src_idx] * inv_spp;
      out[dst_idx * 3 + 0] = c.x;
      out[dst_idx * 3 + 1] = c.y;
      out[dst_idx * 3 + 2] = c.z;
    }
  }
  swapchain.swap_writer();
}

} // namespace xn
