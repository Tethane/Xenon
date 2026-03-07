#pragma once
// geometry/primitives.h — Analytic geometric primitives
//
// All primitives produce a HitRecord identical in layout to triangle hits
// so the integrator/shading code needs zero special-casing.
//
// Intersection convention:
//   • rec.t must be initialised to the current best hit distance before calling.
//   • On success: rec.t is updated, all other fields are written.
//   • rec.prim_id: set by the caller (PrimBLAS) to the prim's index in PrimGroup.
//   • Normals: all outward-face convention (faceforward applied internally).
//   • UV: each primitive produces texture coordinates in [0,1]^2 where meaningful.

#include "math/ray.h"
#include "geometry/aabb.h"
#include <cmath>

namespace xn {

// ─────────────────────────────────────────────────────────────────────────────
// Möller–Trumbore ray-triangle intersection
// Returns true and fills t, u, v (barycentric) if hit in [ray.tmin, t_max]
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

    [[nodiscard]] bool intersect(const Ray& ray, HitRecord& rec) const noexcept {
        // Use the numerically robust form: a = dot(d,d), hb = dot(oc,d)
        Vec3  oc = ray.origin - center;
        float a  = dot(ray.dir, ray.dir);
        float hb = dot(oc, ray.dir);
        float c  = dot(oc, oc) - radius * radius;
        float disc = hb * hb - a * c;
        if (disc < 0.f) return false;

        float sq = std::sqrt(disc);
        float t  = (-hb - sq) / a;
        if (t < ray.tmin || t >= rec.t) {
            t = (-hb + sq) / a;
            if (t < ray.tmin || t >= rec.t) return false;
        }

        Vec3 outward = (ray.at(t) - center) / radius; // unit outward normal

        // Spherical UV: u = phi/(2pi), v = (pi-theta)/pi  (standard latitude-longitude)
        float phi   = std::atan2(-outward.z, outward.x) + kPi;
        float theta = std::acos(std::clamp(-outward.y, -1.f, 1.f));

        rec.t          = t;
        rec.pos        = ray.at(t);
        rec.geo_normal = outward;
        rec.normal     = outward;
        rec.front_face = dot(outward, ray.dir) < 0.f;
        if (!rec.front_face) {
            rec.geo_normal = -outward;
            rec.normal     = -outward;
        }
        rec.mat_id = mat_id;
        rec.u      = phi   * (0.5f * kInvPi);    // [0, 1]
        rec.v      = theta * kInvPi;              // [0, 1]
        // prim_id set by caller
        return true;
    }

    AABB aabb() const noexcept {
        Vec3 r(radius);
        return { center - r, center + r };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Infinite Plane  (dot(normal, p) = offset)
// ─────────────────────────────────────────────────────────────────────────────
struct Plane {
    Vec3  normal;    // must be unit length
    float offset;   // dot(normal, point_on_plane)
    int   mat_id;

    [[nodiscard]] bool intersect(const Ray& ray, HitRecord& rec) const noexcept {
        float denom = dot(normal, ray.dir);
        if (std::abs(denom) < 1e-7f) return false;
        float t = (offset - dot(normal, ray.origin)) / denom;
        if (t < ray.tmin || t >= rec.t) return false;

        Vec3 p = ray.at(t);
        Vec3 n = (denom < 0.f) ? normal : -normal; // face toward ray origin

        rec.t          = t;
        rec.pos        = p;
        rec.geo_normal = n;
        rec.normal     = n;
        rec.front_face = denom < 0.f;
        rec.mat_id     = mat_id;
        // Planar UV: arbitrary tiling in world XZ or XY depending on normal axis
        rec.u = p.x - std::floor(p.x);
        rec.v = p.z - std::floor(p.z);
        return true;
    }

    // Infinite plane has no useful finite AABB; return a large slab on the normal axis.
    // In practice, planes are special-cased in PrimBLAS to not split the tree further.
    AABB aabb() const noexcept {
        constexpr float kHuge = 1e4f;
        Vec3 abs_n = { std::abs(normal.x), std::abs(normal.y), std::abs(normal.z) };
        // The plane extends infinitely in the two non-dominant axes.
        Vec3 lo = Vec3(-kHuge) + normal * (offset - 0.01f);
        Vec3 hi = Vec3( kHuge) + normal * (offset + 0.01f);
        for (int i = 0; i < 3; ++i) {
            if (abs_n[i] > 0.9f) {
                // Dominant axis: thin slab across offset
                lo[i] = offset - 0.01f;
                hi[i] = offset + 0.01f;
            }
        }
        return { lo, hi };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Box  (axis-aligned, defined by center + half-extents)
// ─────────────────────────────────────────────────────────────────────────────
struct Box {
    Vec3 center;
    Vec3 half;    // half-extents along X, Y, Z
    int  mat_id;

    [[nodiscard]] bool intersect(const Ray& ray, HitRecord& rec) const noexcept {
        // Slab intersection; track which axis produced the entering face.
        const Vec3 mn = center - half;
        const Vec3 mx = center + half;

        float tmin = ray.tmin, tmax = rec.t;
        int   near_axis = 0;
        bool  near_min_face = false; // true = ray came from the mn face

        for (int i = 0; i < 3; ++i) {
            float invD = 1.f / ray.dir[i];
            float t0 = (mn[i] - ray.origin[i]) * invD;
            float t1 = (mx[i] - ray.origin[i]) * invD;
            bool  flip = invD < 0.f;
            if (flip) { float tmp = t0; t0 = t1; t1 = tmp; }
            if (t0 > tmin) {
                tmin      = t0;
                near_axis = i;
                near_min_face = !flip; // mn face if ray enters from the low side
            }
            tmax = std::min(tmax, t1);
            if (tmin > tmax) return false;
        }

        rec.t   = tmin;
        rec.pos = ray.at(tmin);
        rec.mat_id = mat_id;

        // Face normal along near_axis
        Vec3 n(0.f);
        n[near_axis] = near_min_face ? -1.f : 1.f; // outward: away from center
        if (dot(n, ray.dir) > 0.f) n = -n; // face toward ray
        rec.geo_normal = n;
        rec.normal     = n;
        rec.front_face = dot(n, ray.dir) < 0.f;

        // UV: project onto face
        int u_axis = (near_axis + 1) % 3;
        int v_axis = (near_axis + 2) % 3;
        rec.u = (rec.pos[u_axis] - mn[u_axis]) / (2.f * half[u_axis]);
        rec.v = (rec.pos[v_axis] - mn[v_axis]) / (2.f * half[v_axis]);
        return true;
    }

    AABB aabb() const noexcept {
        return { center - half, center + half };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Disk  (center + normal + radius)
// Intersection: ray-plane test followed by radius check.
// ─────────────────────────────────────────────────────────────────────────────
struct Disk {
    Vec3  center;
    Vec3  normal;  // unit outward normal
    float radius;
    int   mat_id;

    [[nodiscard]] bool intersect(const Ray& ray, HitRecord& rec) const noexcept {
        float denom = dot(normal, ray.dir);
        if (std::abs(denom) < 1e-7f) return false;
        float t = dot(center - ray.origin, normal) / denom;
        if (t < ray.tmin || t >= rec.t) return false;

        Vec3 p  = ray.at(t);
        Vec3 op = p - center;
        if (dot(op, op) > radius * radius) return false; // outside disk

        Vec3 n = (denom < 0.f) ? normal : -normal;
        rec.t          = t;
        rec.pos        = p;
        rec.geo_normal = n;
        rec.normal     = n;
        rec.front_face = denom < 0.f;
        rec.mat_id     = mat_id;

        // Polar UV: u = angle / (2π), v = radial distance / radius
        // Build a local 2D frame in the disk plane
        Vec3 ref  = (std::abs(normal.x) < 0.9f) ? Vec3(1,0,0) : Vec3(0,1,0);
        Vec3 tang = normalize(cross(normal, ref));
        Vec3 bita = cross(normal, tang);
        float px  = dot(op, tang);
        float py  = dot(op, bita);
        float phi = std::atan2(py, px);
        if (phi < 0.f) phi += 2.f * kPi;
        rec.u = phi * (0.5f * kInvPi);
        rec.v = std::sqrt(dot(op, op)) / radius;
        return true;
    }

    AABB aabb() const noexcept {
        // The disk fits inside a sphere of `radius` centered at `center`;
        // but that wastes space. Build a tighter box by projecting the disk.
        Vec3 e = {
            radius * std::sqrt(1.f - normal.x * normal.x),
            radius * std::sqrt(1.f - normal.y * normal.y),
            radius * std::sqrt(1.f - normal.z * normal.z)
        };
        // Pad by 1mm to avoid zero-thickness AABB on axis-aligned disks
        e = max3(e, Vec3(0.001f));
        return { center - e, center + e };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Quad  (parallelogram, defined by origin + two edge vectors)
//   Points on quad: P = origin + s*u_vec + t*v_vec,  s,t ∈ [0,1]
// ─────────────────────────────────────────────────────────────────────────────
struct Quad {
    Vec3 origin;  // one corner
    Vec3 u_vec;   // first edge vector
    Vec3 v_vec;   // second edge vector
    int  mat_id;

    [[nodiscard]] bool intersect(const Ray& ray, HitRecord& rec) const noexcept {
        Vec3 n_raw = cross(u_vec, v_vec); // not necessarily unit length
        float area_sq = dot(n_raw, n_raw);
        if (area_sq < 1e-14f) return false;

        float denom = dot(n_raw, ray.dir);
        if (std::abs(denom) < 1e-9f) return false;
        float t = dot(n_raw, origin - ray.origin) / denom;
        if (t < ray.tmin || t >= rec.t) return false;

        // Project hit point onto quad's local frame to find barycentric (s, tv)
        Vec3 p  = ray.at(t);
        Vec3 op = p - origin;

        // s = (p - origin) × v_vec · n̂ / |u × v|,  using triple products
        float s  = dot(cross(op, v_vec), n_raw) / area_sq;
        float tv = dot(cross(u_vec, op), n_raw) / area_sq;
        if (s < 0.f || s > 1.f || tv < 0.f || tv > 1.f) return false;

        Vec3 n = normalize(n_raw);
        if (denom > 0.f) n = -n; // face toward ray

        rec.t          = t;
        rec.pos        = p;
        rec.geo_normal = n;
        rec.normal     = n;
        rec.front_face = denom < 0.f;
        rec.mat_id     = mat_id;
        rec.u          = s;
        rec.v          = tv;
        return true;
    }

    AABB aabb() const noexcept {
        AABB b;
        b.expand(origin);
        b.expand(origin + u_vec);
        b.expand(origin + v_vec);
        b.expand(origin + u_vec + v_vec);
        // Pad 1mm on each axis to avoid degenerate AABB for planar quads
        b.mn = b.mn - Vec3(0.001f);
        b.mx = b.mx + Vec3(0.001f);
        return b;
    }
};

} // namespace xn
