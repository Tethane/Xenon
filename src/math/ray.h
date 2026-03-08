#pragma once
// math/ray.h — ray and hit record types; RayPacket4 for SIMD traversal

#include <cstdint>
#include <limits>

#include "vec3.h"
#include "simd.h"

namespace xn {

constexpr float kInfinity   = std::numeric_limits<float>::infinity();
constexpr float kRayEps     = 1e-4f;  // surface offset for bounce rays
constexpr float kShadowEps  = 5e-4f;  // surface offset for shadow rays
constexpr float kPi         = 3.14159265358979323846f;
constexpr float kInvPi      = 1.f / kPi;

// ─────────────────────────────────────────────────────────────────────────────
// Ray Origin Offset Helpers
// ─────────────────────────────────────────────────────────────────────────────

inline Vec3 offset_ray_origin(const Vec3& p, const Vec3& ng, const Vec3& dir, float eps) {
    Vec3 n = ng;
    if (dot(n, dir) < 0.f) n = -n;
    return p + n * eps;
}

inline Vec3 offset_ray_origin(const Vec3& p, const Vec3& ng, const Vec3& dir) {
    return offset_ray_origin(p, ng, dir, kRayEps);
}

// ─────────────────────────────────────────────────────────────────────────────
// Ray
// ─────────────────────────────────────────────────────────────────────────────
struct Ray {
    Vec3  origin;
    Vec3  dir;
    float tmin = kRayEps;
    float tmax = kInfinity;

    Vec3 at(float t) const { return origin + t * dir; }
};

// ─────────────────────────────────────────────────────────────────────────────
// HitRecord — written by intersection routines
// ─────────────────────────────────────────────────────────────────────────────
struct HitRecord {
    float  t        = kInfinity;
    Vec3   pos      = {};
    Vec3   normal   = {};       // always outward-facing
    Vec3   geo_normal = {};     // geometric (flat) normal
    int    mat_id   = -1;
    int    prim_id  = -1;       // triangle / primitive index
    int    instance_id = -1;    // mesh instance index
    float  u = 0, v = 0;       // barycentric / UV
    bool   front_face = true;

    bool valid() const { return t < kInfinity; }
};

// ─────────────────────────────────────────────────────────────────────────────
// RayPacket4 — 4 rays packed SoA for SIMD BVH traversal
// ─────────────────────────────────────────────────────────────────────────────
struct alignas(16) RayPacket4 {
    Float4 ox, oy, oz;           // origins
    Float4 dx, dy, dz;           // directions
    Float4 inv_dx, inv_dy, inv_dz; // precomputed reciprocals
    Float4 tmin, tmax;
    int    sign[3];              // sign bit of direction per axis

    static RayPacket4 from_rays(const Ray* rays, int count) {
        alignas(16) float ox[4]={}, oy[4]={}, oz[4]={};
        alignas(16) float dx[4]={}, dy[4]={}, dz[4]={};
        alignas(16) float tmin[4]={}, tmax[4]={};
        for (int i = 0; i < count; ++i) {
            ox[i]=rays[i].origin.x; oy[i]=rays[i].origin.y; oz[i]=rays[i].origin.z;
            dx[i]=rays[i].dir.x;    dy[i]=rays[i].dir.y;    dz[i]=rays[i].dir.z;
            tmin[i]=rays[i].tmin;   tmax[i]=rays[i].tmax;
        }
        RayPacket4 p;
        p.ox=Float4(_mm_load_ps(ox)); p.oy=Float4(_mm_load_ps(oy)); p.oz=Float4(_mm_load_ps(oz));
        p.dx=Float4(_mm_load_ps(dx)); p.dy=Float4(_mm_load_ps(dy)); p.dz=Float4(_mm_load_ps(dz));
        p.tmin=Float4(_mm_load_ps(tmin)); p.tmax=Float4(_mm_load_ps(tmax));
        p.inv_dx = rcp4(p.dx); p.inv_dy = rcp4(p.dy); p.inv_dz = rcp4(p.dz);
        p.sign[0] = (p.dx[0] < 0) ? 1 : 0;
        p.sign[1] = (p.dy[0] < 0) ? 1 : 0;
        p.sign[2] = (p.dz[0] < 0) ? 1 : 0;
        return p;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Utility: orthonormal basis from a normal
// ─────────────────────────────────────────────────────────────────────────────
struct Onb {
    Vec3 u, v, n; // n = shading normal

    explicit Onb(Vec3 normal) : n(normal) {
        // Duff et al. "Building an Orthonormal Basis, Revisited" (2017)
        float s = (n.z >= 0.f) ? 1.f : -1.f;
        float a = -1.f / (s + n.z);
        float b = n.x * n.y * a;
        u = Vec3(1.f + s*n.x*n.x*a, s*b, -s*n.x);
        v = Vec3(b, s + n.y*n.y*a, -n.y);
    }

    Vec3 to_world(Vec3 local) const {
        return local.x*u + local.y*v + local.z*n;
    }
    Vec3 to_local(Vec3 world) const {
        return {dot(world, u), dot(world, v), dot(world, n)};
    }
};

} // namespace xn
