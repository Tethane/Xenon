#pragma once
// render/integrator.h — Unbiased Path Tracing Integrator

#include "scene/scene.h"
#include "camera/sampler.h"

namespace xn {

class PathIntegrator {
public:
    PathIntegrator(int max_depth = 8) : max_depth_(max_depth) {}

    // Computes incoming radiance along a single ray
    Vec3 li(const Ray& ray, const Scene& scene, PCGState& rng) const;

private:
    int max_depth_;

    // Sample a point on an area light
    struct LightSample {
        Vec3 pos;
        Vec3 normal;
        Vec3 emission;
        float pdf;
    };
    
    static LightSample sample_area_light(const Scene& scene, const Light& l, float u1, float u2);
};

} // namespace xn
