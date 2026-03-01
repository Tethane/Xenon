#pragma once
// material/bsdf.h — Principled BSDF (subset of Blender's model)

#include "math/vec3.h"
#include "math/ray.h"
#include <cmath>
#include <algorithm>

namespace xn {

struct PrincipledBSDF {
    Vec3  albedo    = Vec3(0.5f);
    float metallic  = 0.0f;
    float roughness = 0.5f;
    float ior       = 1.5f;
    float specular  = 0.5f;
};

// ─────────────────────────────────────────────────────────────────────────────
// GGX / Trowbridge-Reitz Microfacet Model
// ─────────────────────────────────────────────────────────────────────────────

inline float ggx_d(Vec3 m, Vec3 n, float alpha) {
    float cos_theta = dot(m, n);
    if (cos_theta <= 0.f) return 0.f;
    float a2 = alpha * alpha;
    float d = (cos_theta * cos_theta) * (a2 - 1.f) + 1.f;
    return a2 / (kPi * d * d);
}

inline float ggx_g1(Vec3 v, Vec3 n, float alpha) {
    float cos_theta = std::abs(dot(v, n));
    float a2 = alpha * alpha;
    return 2.f * cos_theta / (cos_theta + std::sqrt(a2 + (1.f - a2) * cos_theta * cos_theta));
}

inline float ggx_g2(Vec3 l, Vec3 v, Vec3 n, float alpha) {
    return ggx_g1(l, n, alpha) * ggx_g1(v, n, alpha);
}

inline Vec3 fresnel_schlick(float cos_theta, Vec3 f0) {
    return f0 + (Vec3(1.f) - f0) * std::pow(std::max(0.f, 1.f - cos_theta), 5.f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sampling utilities
// ─────────────────────────────────────────────────────────────────────────────

inline Vec3 sample_cosine_hemisphere(float u1, float u2) {
    float r = std::sqrt(u1);
    float phi = 2.0f * kPi * u2;
    return { r * std::cos(phi), r * std::sin(phi), std::sqrt(std::max(0.0f, 1.0f - u1)) };
}

inline Vec3 sample_ggx(float u1, float u2, float alpha) {
    float phi = 2.f * kPi * u1;
    float cos_theta = std::sqrt((1.f - u2) / (1.f + (alpha * alpha - 1.f) * u2));
    float sin_theta = std::sqrt(std::max(0.f, 1.f - cos_theta * cos_theta));
    return { sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta };
}

} // namespace xn
