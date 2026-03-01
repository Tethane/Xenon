#pragma once
// render/wavefront_state.h — State for wavefront path tracing

#include "math/ray.h"
#include <vector>

namespace xn {

// PathState — state of a single path in the wavefront
struct PathState {
    Ray   ray;
    Vec3  throughput = Vec3(1.f);
    Vec3  radiance   = Vec3(0.f);
    int   pixel_idx  = -1;
    bool  active     = false;
    bool  specular   = false; // For MIS
};

// Simple queue for indices of active paths
struct WavefrontQueue {
    std::vector<int> indices;
    int size = 0;

    void reset(int max_size) {
        indices.resize(max_size);
        size = 0;
    }

    void push(int idx) {
        indices[size++] = idx;
    }
};

} // namespace xn
