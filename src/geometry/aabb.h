#pragma once
// geometry/aabb.h — Axis-Aligned Bounding Box with SIMD slab intersection

#include <algorithm>
#include <cstring>

#include "math/vec3.h"
#include "math/ray.h"
#include "math/simd.h"

namespace xn {

// ─────────────────────────────────────────────────────────────────────────────
// AABB — 24 bytes
// ─────────────────────────────────────────────────────────────────────────────
struct AABB {
    Vec3 mn, mx;

    AABB() : mn(kInfinity), mx(-kInfinity) {}
    AABB(Vec3 mn, Vec3 mx) : mn(mn), mx(mx) {}

    void expand(Vec3 p) {
        mn = min3(mn, p);
        mx = max3(mx, p);
    }
    void expand(const AABB& b) {
        mn = min3(mn, b.mn);
        mx = max3(mx, b.mx);
    }

    Vec3   center()  const { return (mn + mx) * 0.5f; }
    Vec3   extent()  const { return mx - mn; }
    float  surface_area() const {
        Vec3 e = extent();
        return 2.f * (e.x*e.y + e.y*e.z + e.z*e.x);
    }
    int    max_extent_axis() const { return max_axis(extent()); }
    bool   valid() const { return mn.x<=mx.x && mn.y<=mx.y && mn.z<=mx.z; }

    // Scalar slab intersection — returns true if hit in [tmin, tmax]
    bool intersect(const Ray& ray, float& t_hit_near, float& t_hit_far) const {
        // Use precomputed reciprocal only if available — general fallback here
        float tx0 = (mn.x - ray.origin.x) / ray.dir.x;
        float tx1 = (mx.x - ray.origin.x) / ray.dir.x;
        float ty0 = (mn.y - ray.origin.y) / ray.dir.y;
        float ty1 = (mx.y - ray.origin.y) / ray.dir.y;
        float tz0 = (mn.z - ray.origin.z) / ray.dir.z;
        float tz1 = (mx.z - ray.origin.z) / ray.dir.z;

        float tmin = std::max({std::min(tx0,tx1), std::min(ty0,ty1), std::min(tz0,tz1), ray.tmin});
        float tmax = std::min({std::max(tx0,tx1), std::max(ty0,ty1), std::max(tz0,tz1), ray.tmax});

        t_hit_near = tmin;
        t_hit_far  = tmax;
        return tmin <= tmax;
    }

    // Faster version using precomputed reciprocal direction (for BVH traversal)
    bool intersect_fast(Vec3 inv_dir, Vec3 origin, int sign[3], float tmin, float tmax) const {
        const Vec3* bounds[2] = {&mn, &mx};
        float txmin = (bounds[  sign[0]]->x - origin.x) * inv_dir.x;
        float txmax = (bounds[1-sign[0]]->x - origin.x) * inv_dir.x;
        float tymin = (bounds[  sign[1]]->y - origin.y) * inv_dir.y;
        float tymax = (bounds[1-sign[1]]->y - origin.y) * inv_dir.y;
        float tzmin = (bounds[  sign[2]]->z - origin.z) * inv_dir.z;
        float tzmax = (bounds[1-sign[2]]->z - origin.z) * inv_dir.z;

        tmin = std::max({tmin, txmin, tymin, tzmin});
        tmax = std::min({tmax, txmax, tymax, tzmax});
        return tmin <= tmax;
    }

    void get_corners(Vec3 corners[8]) const {
        corners[0] = {mn.x, mn.y, mn.z};
        corners[1] = {mx.x, mn.y, mn.z};
        corners[2] = {mn.x, mx.y, mn.z};
        corners[3] = {mx.x, mx.y, mn.z};
        corners[4] = {mn.x, mn.y, mx.z};
        corners[5] = {mx.x, mn.y, mx.z};
        corners[6] = {mn.x, mx.y, mx.z};
        corners[7] = {mx.x, mx.y, mx.z};
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// AABB slab test for 4 boxes simultaneously (SSE) — used in BVH traversal
// Returns a bitmask: bit i set = ray i hits box
// ─────────────────────────────────────────────────────────────────────────────
struct alignas(16) AABB4 {
    // SoA layout: [minx0..3] [miny0..3] [minz0..3] [maxx0..3] ...
    Float4 mn_x, mn_y, mn_z;
    Float4 mx_x, mx_y, mx_z;

    void set(int lane, const AABB& b) {
        // Write single lane — used during build only, not hot path
        alignas(16) float buf[4];
        auto set_lane = [&](Float4& f, float val) {
            _mm_store_ps(buf, f.v);
            buf[lane] = val;
            f.v = _mm_load_ps(buf);
        };
        set_lane(mn_x, b.mn.x); set_lane(mn_y, b.mn.y); set_lane(mn_z, b.mn.z);
        set_lane(mx_x, b.mx.x); set_lane(mx_y, b.mx.y); set_lane(mx_z, b.mx.z);
    }

    // Returns bitmask of hitting lanes
    int intersect4(const RayPacket4& rp) const {
        Float4 tx0 = (mn_x - rp.ox) * rp.inv_dx;
        Float4 tx1 = (mx_x - rp.ox) * rp.inv_dx;
        Float4 ty0 = (mn_y - rp.oy) * rp.inv_dy;
        Float4 ty1 = (mx_y - rp.oy) * rp.inv_dy;
        Float4 tz0 = (mn_z - rp.oz) * rp.inv_dz;
        Float4 tz1 = (mx_z - rp.oz) * rp.inv_dz;

        Float4 tmin = max4(max4(min4(tx0,tx1), min4(ty0,ty1)), max4(min4(tz0,tz1), rp.tmin));
        Float4 tmax = min4(min4(max4(tx0,tx1), max4(ty0,ty1)), min4(max4(tz0,tz1), rp.tmax));

        return (tmin <= tmax).movemask();
    }
};

} // namespace xn
