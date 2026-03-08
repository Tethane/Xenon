#pragma once
// render/wavefront.h — Wavefront rendering and progressive accumulation

#include <cstring>

#include "scene/scene.h"
#include "camera/camera.h"
#include "render/swapchain.h"
#include "render/thread_pool.h"
#include "render/wavefront_state.h"

// #define XN_DEBUG_QUEUES 1

namespace xn {

// Wavefront Renderer
// Unbiased Path Integrator supporting NEE and MIS using a staged wavefront pipeline
// Optimized for Parallelism and easy swapping from CPU to GPU
class WavefrontRenderer {
public:
    WavefrontRenderer(int width, int height);
    ~WavefrontRenderer();

    void render_frame(const Scene& scene, const Camera& camera, TripleSwapchain& swapchain);
    void render_frame_tiled(const Scene& scene, const Camera& camera, TripleSwapchain& swapchain);
    
    void reset_accumulation() { spp_ = 0; }
    int  get_spp() const { return spp_; }

    void set_bounces(int min_b, int max_b) { min_bounces_ = min_b; max_bounces_ = max_b; }

private:
    int width_, height_;
    int spp_ = 0;
    int min_bounces_ = 3;
    int max_bounces_ = 8;
    std::vector<Vec3> accumulation_buffer_;
    ThreadPool* pool_;
};

template <typename T>
static inline void flush_local_to_queue(WavefrontQueue<T>& q, const std::vector<T>& local) {
  if (local.empty()) return;
  int base = q.push_block((int) local.size());
  std::memcpy(q.items.data() + base, local.data(), local.size() * sizeof(T));
}

// Wavefront Pipeline Kernels

// Generates Camera Rays over the Entire Frame
void raygen(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<RayWorkItem>& q_rays, const Camera& camera, int spp, int height, int width);

// Filters Out Misses — Input: q_rays, Output: q_hits
void closest_hit(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<RayWorkItem>& q_rays, WavefrontQueue<HitWorkItem>& q_hits, const Scene& scene);

// Generate Shadow Rays at Hits and Adjust MIS weighting — Input: q_hits, Output: q_shadow
void next_event_estimation(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q_hits, WavefrontQueue<ShadowWorkItem>& q_shadow, const Scene& scene, int min_bounces);

// Classify hits into 6 queues based on material — Input: q_hits, Output: 6 per-lobe queues
void classification(ThreadPool& pool, std::vector<PathState>& paths,
    WavefrontQueue<HitWorkItem>& q_hits,
    WavefrontQueue<HitWorkItem>& q_diffuse,
    WavefrontQueue<HitWorkItem>& q_microfacet_refl,
    WavefrontQueue<HitWorkItem>& q_microfacet_trans,
    WavefrontQueue<HitWorkItem>& q_delta_refl,
    WavefrontQueue<HitWorkItem>& q_delta_trans,
    WavefrontQueue<HitWorkItem>& q_subsurface,
    const Scene& scene);

// Per-queue shading kernels — each calls only its queue's sample()
void shade_diffuse(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene);
void shade_microfacet_refl(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene);
void shade_microfacet_trans(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene);
void shade_delta_refl(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene);
void shade_delta_trans(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene);
void shade_subsurface(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene);

// Compute Shadow Ray Intersections — Input: q_shadow, Output: radiance accumulation
void shadow(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<ShadowWorkItem>& q_shadow, const Scene& scene);

} // namespace xn
