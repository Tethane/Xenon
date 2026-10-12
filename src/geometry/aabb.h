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

	void expand(const AABB &b) {
		if (!b.valid())
			return;
		mn = min3(mn, b.mn);
		mx = max3(mx, b.mx);
	}

	Vec3 center() const {
		Vec3 c = (mn + mx) * 0.5f;

		for (int i = 0; i < 3; ++i)
			if (!std::isfinite(c[i]))
				c[i] = 0.f;

		return c;
	}

	Vec3 extent() const {
		return mx - mn;
	}

	float surface_area() const {
		Vec3 e = extent();

		return 2.f * (e.x * e.y + e.y * e.z + e.z * e.x);
	}

	int max_extent_axis() const {
		return max_axis(extent());
	}

	bool valid() const {
		return mn.x <= mx.x && mn.y <= mx.y && mn.z <= mx.z;
	}

	// Scalar slab intersection — returns true if hit in [tmin, tmax]
	bool intersect(const Ray &ray, float &t_hit_near, float &t_hit_far) const {
		float tmin = ray.tmin, tmax = ray.tmax;

		for (int i = 0; i < 3; ++i) {
			if (ray.dir[i] == 0.f) {
				if (ray.origin[i] < mn[i] || ray.origin[i] > mx[i])
					return false;
				continue;
			}
			float a = (mn[i] - ray.origin[i]) / ray.dir[i];
			float b = (mx[i] - ray.origin[i]) / ray.dir[i];
			tmin = std::max(tmin, std::min(a, b));
			tmax = std::min(tmax, std::max(a, b));
		}
		t_hit_near = tmin;
		t_hit_far = tmax;

		return tmin <= tmax;
	}

	bool intersect_fast(Vec3 inv_dir, Vec3 origin, const int[3], float tmin, float tmax) const {
		for (int i = 0; i < 3; ++i) {
			if (std::isinf(inv_dir[i])) {
				if (origin[i] < mn[i] || origin[i] > mx[i])
					return false;
				continue;
			}
			float a = (mn[i] - origin[i]) * inv_dir[i];
			float b = (mx[i] - origin[i]) * inv_dir[i];
			tmin = std::max(tmin, std::min(a, b));
			tmax = std::min(tmax, std::max(a, b));
		}

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

	void set(int lane, const AABB &b) {
		// Write single lane — used during build only, not hot path
		alignas(16) float buf[4];
		auto set_lane = [&](Float4 &f, float val) {
			_mm_store_ps(buf, f.v);
			buf[lane] = val;
			f.v = _mm_load_ps(buf);
		};
		set_lane(mn_x, b.mn.x);
		set_lane(mn_y, b.mn.y);
		set_lane(mn_z, b.mn.z);
		set_lane(mx_x, b.mx.x);
		set_lane(mx_y, b.mx.y);
		set_lane(mx_z, b.mx.z);
	}

	// Returns bitmask of hitting lanes
	int intersect4(const RayPacket4 &rp) const {
		Float4 tx0 = (mn_x - rp.ox) * rp.inv_dx;
		Float4 tx1 = (mx_x - rp.ox) * rp.inv_dx;
		Float4 ty0 = (mn_y - rp.oy) * rp.inv_dy;
		Float4 ty1 = (mx_y - rp.oy) * rp.inv_dy;
		Float4 tz0 = (mn_z - rp.oz) * rp.inv_dz;
		Float4 tz1 = (mx_z - rp.oz) * rp.inv_dz;

		Float4 tmin = max4(max4(min4(tx0, tx1), min4(ty0, ty1)), max4(min4(tz0, tz1), rp.tmin));
		Float4 tmax = min4(min4(max4(tx0, tx1), max4(ty0, ty1)), min4(max4(tz0, tz1), rp.tmax));

		return (tmin <= tmax).movemask();
	}
};

} // namespace xn
