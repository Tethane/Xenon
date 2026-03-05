#pragma once
// render/wavefront.h — Wavefront rendering and progressive accumulation

#include "scene/scene.h"
#include "camera/camera.h"
#include "render/swapchain.h"
#include "render/thread_pool.h"
#include "render/wavefront_state.h"

#include <cstring>

namespace xn {

// Wavefront Renderer
// Unbiased Path Integrator supporting NEE and MIS using a staged wavefront pipeline
// Optimized for Parallelism and easy swapping from CPU to GPU
class WavefrontRenderer {
public:
    WavefrontRenderer(int width, int height);
    ~WavefrontRenderer();

    void render_frame(const Scene& scene, const Camera& camera, TripleSwapchain& swapchain);
    
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
// Input: None
// Output: Camera generated rays stored in q_rays
void raygen(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<RayWorkItem>& q_rays, const Camera& camera, int spp, int height, int width);

// Filters Out Misses
// Input: q_rays
// Output: q_hits
void closest_hit(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<RayWorkItem>& q_rays, WavefrontQueue<HitWorkItem>& q_hits, const Scene& scene);

// Generate Shadow Rays at Hits and Adjust MIS weighting
// Input: q_hits
// Output: q_shadow
void next_event_estimation(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q_hits, WavefrontQueue<ShadowWorkItem>& q_shadow, const Scene& scene, int min_bounces);

// Classify Rays Based on BSDF Sampling and NEE Compatibility
// Input: q_hits
// Output: q_hits_diffuse, q_hits_glossy_refl, q_hits_glossy_trans, q_hits_delta
void classification(ThreadPool& pool, WavefrontQueue<HitWorkItem>& q_hits, WavefrontQueue<HitWorkItem>& q_hit_diffuse, WavefrontQueue<HitWorkItem>& q_hit_glossy_refl, WavefrontQueue<HitWorkItem>& q_hit_glossy_trans, WavefrontQueue<HitWorkItem>& q_hit_delta, const Scene& scene);

// Shade Diffuse Hits and Sample BSDF:
// Input: q_hits_diffuse
// Output: q_rays_next
void shade_diffuse(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q_hits_diffuse, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene);

// Shade Glossy Reflection Hits and Sample BSDF:
// Input: q_hits_glossy_refl
// Output: q_rays_next
void shade_glossy_refl(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q_hits_glossy_refl, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene);


// Shade Glossy Transmittion Hits and Sample BSDF:
// Input: q_hits_glossy_trans
// Output: q_rays_next
void shade_glossy_trans(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q_hits_glossy_trans, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene);

// Shade Delta Hits and Produce Reflected Ray
// Input: q_hits_delta
// Output: q_rays_next
void shade_delta(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<HitWorkItem>& q_hits_delta, WavefrontQueue<RayWorkItem>& q_rays_next, const Scene& scene);

// Compute Shadow Ray Intersections
// Input: q_shadow
// Output: None
void shadow(ThreadPool& pool, std::vector<PathState>& paths, WavefrontQueue<ShadowWorkItem>& q_shadow, const Scene& scene);

} // namespace xn
