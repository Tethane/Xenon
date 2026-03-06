#pragma once
// render/wavefront_state.h — State for wavefront path tracing

#include "math/ray.h"
#include "camera/sampler.h"
#include <vector>
#include <cstdint>
#include <atomic>

namespace xn {

enum LobeType {
  LOBE_DIFFUSE = 0,
  LOBE_MICROFACET_REFL,
  LOBE_MICROFACET_TRANS,
  LOBE_DELTA_REFL,
  LOBE_DELTA_TRANS,
  LOBE_DIFFUSE_SUBSURFACE,
  LOBE_COUNT
};

// PathState — state of a single path in the wavefront
struct PathState {
  Ray   ray;
  Vec3  throughput = Vec3(1.f);
  Vec3  radiance   = Vec3(0.f);
  PCGState rng;        // Per-path RNG
  int   pixel_idx  = -1;
  int   depth      = 0;
  float prev_bsdf_pdf_sa = 0.f;
  bool  active     = true;
  bool  specular   = false; // For MIS
  bool  prev_was_delta = false;
};

struct RayWorkItem {
  int path_idx;
};

struct HitWorkItem {
  int path_idx;
  HitRecord hit;
};

struct ShadowWorkItem {
  int   path_idx;
  Ray   ray;
  float t_max;
  Vec3  contrib;
  int   origin_prim_id;
  int   light_prim_id;
};

// Typed, chunk-friendly queue
template<typename T>
struct WavefrontQueue {
  std::vector<T> items;
  std::atomic<int> size{0};

  void reset(int max_size) {
    if ((int)items.size() < max_size) items.resize(max_size);
    this->size.store(0);
  }

  void push(const T& item) {
    int idx = this->size.fetch_add(1, std::memory_order_relaxed);
    items[idx] = item;
  }

  // Support for block pushing (optional for now, but good for performance)
  int push_block(int n) {
    return this->size.fetch_add(n, std::memory_order_relaxed);
  }
};

} // namespace xn
