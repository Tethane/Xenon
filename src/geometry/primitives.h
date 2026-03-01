#pragma once
// geometry/primitives.h — Sphere and Plane analytic primitives

#include "math/ray.h"
#include "geometry/aabb.h"
#include <cmath>

namespace xn {

// ─────────────────────────────────────────────────────────────────────────────
// Möller–Trumbore ray-triangle intersection
// Returns true and fills t, u, v (barycentric) if hit in [ray.tmin, ray.tmax]
// ─────────────────────────────────────────────────────────────────────────────
[[nodiscard]] inline bool ray_triangle(
    const Ray& ray,
    Vec3 v0, Vec3 v1, Vec3 v2,
    float& t, float& u, float& v,
    float t_max = kInfinity)
{
    constexpr float kEpsilon = 1e-7f;
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 h  = cross(ray.dir, e2);
    float a = dot(e1, h);
    if (std::abs(a) < kEpsilon) return false; // parallel

    float inv_a = 1.f / a;
    Vec3  s   = ray.origin - v0;
    u = inv_a * dot(s, h);
    if (u < 0.f || u > 1.f) return false;

    Vec3  q = cross(s, e1);
    v = inv_a * dot(ray.dir, q);
    if (v < 0.f || u+v > 1.f) return false;

    t = inv_a * dot(e2, q);
    return t >= ray.tmin && t <= t_max;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sphere
// ─────────────────────────────────────────────────────────────────────────────
struct Sphere {
    Vec3  center;
    float radius;
    int   mat_id;

    [[nodiscard]] bool intersect(const Ray& ray, HitRecord& rec) const {
        Vec3  oc = ray.origin - center;
        float a  = dot(ray.dir, ray.dir);
        float hb = dot(oc, ray.dir);
        float c  = dot(oc, oc) - radius*radius;
        float disc = hb*hb - a*c;
        if (disc < 0.f) return false;

        float sq = std::sqrt(disc);
        float t  = (-hb - sq) / a;
        if (t < ray.tmin || t > ray.tmax) {
            t = (-hb + sq) / a;
            if (t < ray.tmin || t > ray.tmax) return false;
        }
        rec.t       = t;
        rec.pos     = ray.at(t);
        Vec3 outward = (rec.pos - center) / radius;
        rec.front_face = dot(outward, ray.dir) < 0.f;
        rec.normal  = rec.front_face ? outward : -outward;
        rec.geo_normal = rec.normal;
        rec.mat_id  = mat_id;
        rec.prim_id = -1;
        return true;
    }

    AABB aabb() const {
        Vec3 r(radius);
        return { center - r, center + r };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Infinite Plane (y = offset, facing +y)
// ─────────────────────────────────────────────────────────────────────────────
struct Plane {
    Vec3  normal;
    float offset;  // dot(normal, point) = offset
    int   mat_id;

    [[nodiscard]] bool intersect(const Ray& ray, HitRecord& rec) const {
        float denom = dot(normal, ray.dir);
        if (std::abs(denom) < 1e-7f) return false;
        float t = (offset - dot(normal, ray.origin)) / denom;
        if (t < ray.tmin || t > ray.tmax) return false;
        rec.t          = t;
        rec.pos        = ray.at(t);
        rec.front_face = denom < 0.f;
        rec.normal     = rec.front_face ? normal : -normal;
        rec.geo_normal = rec.normal;
        rec.mat_id     = mat_id;
        rec.prim_id    = -1;
        return true;
    }
};

} // namespace xn
