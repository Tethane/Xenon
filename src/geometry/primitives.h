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
//   • Normals: outward-facing; front_face records the incident side.
//   • UV: each primitive produces texture coordinates in [0,1]^2 where meaningful.

#include <cmath>

#include "math/ray.h"
#include "geometry/aabb.h"

namespace xn {

// ─────────────────────────────────────────────────────────────────────────────
// Möller–Trumbore ray-triangle intersection
// Returns true and fills t, u, v (barycentric) if hit in [ray.tmin, t_max]
// ─────────────────────────────────────────────────────────────────────────────
[[nodiscard]] inline bool ray_triangle(const Ray &ray, Vec3 v0, Vec3 v1, Vec3 v2, float &t,
									   float &u, float &v, float t_max = kInfinity) {
	auto triple = [](Vec3 a, Vec3 b, Vec3 c) -> double {
		return double(a.x) * (double(b.y) * c.z - double(b.z) * c.y) +
			   double(a.y) * (double(b.z) * c.x - double(b.x) * c.z) +
			   double(a.z) * (double(b.x) * c.y - double(b.y) * c.x);
	};
	Vec3 e1 = v1 - v0, e2 = v2 - v0, delta = ray.origin - v0;
	double det = triple(e1, ray.dir, e2);

	if (det == 0.0)
		return false;
	double bu = triple(delta, ray.dir, e2) / det;

	if (bu < 0.0 || bu > 1.0)
		return false;
	double bv = triple(ray.dir, delta, e1) / det;

	if (bv < 0.0 || bu + bv > 1.0)
		return false;
	double distance = triple(e2, delta, e1) / det;

	if (distance < ray.tmin || distance >= std::min(t_max, ray.tmax))
		return false;
	t = (float)distance;
	u = (float)bu;
	v = (float)bv;

	return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sphere
// ─────────────────────────────────────────────────────────────────────────────
struct Sphere {
	Vec3 center;
	float radius;
	int mat_id;

	[[nodiscard]] bool intersect(const Ray &ray, HitRecord &rec) const noexcept {
		Vec3 oc = ray.origin - center;
		double a = double(ray.dir.x) * ray.dir.x + double(ray.dir.y) * ray.dir.y +
				   double(ray.dir.z) * ray.dir.z;

		if (a == 0.0)
			return false;
		double hb = double(oc.x) * ray.dir.x + double(oc.y) * ray.dir.y + double(oc.z) * ray.dir.z;
		double c = double(oc.x) * oc.x + double(oc.y) * oc.y + double(oc.z) * oc.z -
				   double(radius) * radius;
		double disc = hb * hb - a * c;

		if (disc < 0.0)
			return false;
		double root = std::sqrt(disc);
		double q = -hb - (hb < 0.0 ? -root : root);
		double t0 = q / a, t1 = q == 0.0 ? t0 : c / q;

		if (t0 > t1) {
			double tmp = t0;
			t0 = t1;
			t1 = tmp;
		}
		double distance = t0;

		if (distance < ray.tmin)
			distance = t1;

		if (distance < ray.tmin || distance >= std::min(rec.t, ray.tmax))
			return false;
		float t = (float)distance;

		Vec3 outward = (ray.at(t) - center) / radius; // unit outward normal

		// Spherical UV: u = phi/(2pi), v = (pi-theta)/pi  (standard latitude-longitude)
		float phi = std::atan2(-outward.z, outward.x) + kPi;
		float theta = std::acos(std::clamp(-outward.y, -1.f, 1.f));

		rec.t = t;
		rec.pos = ray.at(t);
		rec.geo_normal = outward;
		rec.normal = outward;
		rec.front_face = dot(outward, ray.dir) < 0.f;
		rec.mat_id = mat_id;
		rec.u = phi * (0.5f * kInvPi); // [0, 1]
		rec.v = theta * kInvPi;		   // [0, 1]
		// prim_id set by caller
		return true;
	}

	AABB aabb() const noexcept {
		Vec3 r(radius);

		return {center - r, center + r};
	}
};

// ─────────────────────────────────────────────────────────────────────────────
// Infinite Plane  (dot(normal, p) = offset)
// ─────────────────────────────────────────────────────────────────────────────
struct Plane {
	Vec3 normal;  // must be unit length
	float offset; // dot(normal, point_on_plane)
	int mat_id;

	[[nodiscard]] bool intersect(const Ray &ray, HitRecord &rec) const noexcept {
		float denom = dot(normal, ray.dir);

		if (denom == 0.f)
			return false;
		float t = (offset - dot(normal, ray.origin)) / denom;

		if (t < ray.tmin || t >= std::min(rec.t, ray.tmax))
			return false;

		Vec3 p = ray.at(t);
		Vec3 n = normal;

		rec.t = t;
		rec.pos = p;
		rec.geo_normal = n;
		rec.normal = n;
		rec.front_face = denom < 0.f;
		rec.mat_id = mat_id;
		// Planar UV: arbitrary tiling in world XZ or XY depending on normal axis
		rec.u = p.x - std::floor(p.x);
		rec.v = p.z - std::floor(p.z);

		return true;
	}

	// Unbounded geometry must never be culled by a finite slab.
	AABB aabb() const noexcept {
		return {Vec3(-kInfinity), Vec3(kInfinity)};
	}
};

// ─────────────────────────────────────────────────────────────────────────────
// Box  (axis-aligned, defined by center + half-extents)
// ─────────────────────────────────────────────────────────────────────────────
struct Box {
	Vec3 center;
	Vec3 half; // half-extents along X, Y, Z
	int mat_id;

	[[nodiscard]] bool intersect(const Ray &ray, HitRecord &rec) const noexcept {
		const Vec3 mn = center - half;
		const Vec3 mx = center + half;
		float near_t = -kInfinity, far_t = kInfinity;
		int near_axis = 0, far_axis = 0;
		float near_sign = -1.f, far_sign = 1.f;

		for (int i = 0; i < 3; ++i) {
			if (ray.dir[i] == 0.f) {
				if (ray.origin[i] < mn[i] || ray.origin[i] > mx[i])
					return false;
				continue;
			}
			float t0 = (mn[i] - ray.origin[i]) / ray.dir[i];
			float t1 = (mx[i] - ray.origin[i]) / ray.dir[i];
			float sign = -1.f;

			if (t0 > t1) {
				std::swap(t0, t1);
				sign = 1.f;
			}

			if (t0 > near_t) {
				near_t = t0;
				near_axis = i;
				near_sign = sign;
			}

			if (t1 < far_t) {
				far_t = t1;
				far_axis = i;
				far_sign = -sign;
			}

			if (near_t > far_t)
				return false;
		}
		float t = near_t;

		if (t < ray.tmin) {
			t = far_t;
			near_axis = far_axis;
			near_sign = far_sign;
		}

		if (t < ray.tmin || t >= std::min(rec.t, ray.tmax))
			return false;
		rec.t = t;
		rec.pos = ray.at(t);
		rec.mat_id = mat_id;
		Vec3 n(0.f);
		n[near_axis] = near_sign;
		rec.geo_normal = n;
		rec.normal = n;
		rec.front_face = dot(n, ray.dir) < 0.f;

		// UV: project onto face
		int u_axis = (near_axis + 1) % 3;
		int v_axis = (near_axis + 2) % 3;
		rec.u = (rec.pos[u_axis] - mn[u_axis]) / (2.f * half[u_axis]);
		rec.v = (rec.pos[v_axis] - mn[v_axis]) / (2.f * half[v_axis]);

		return true;
	}

	AABB aabb() const noexcept {
		return {center - half, center + half};
	}
};

// ─────────────────────────────────────────────────────────────────────────────
// Disk  (center + normal + radius)
// Intersection: ray-plane test followed by radius check.
// ─────────────────────────────────────────────────────────────────────────────
struct Disk {
	Vec3 center;
	Vec3 normal; // unit outward normal
	float radius;
	int mat_id;

	[[nodiscard]] bool intersect(const Ray &ray, HitRecord &rec) const noexcept {
		float denom = dot(normal, ray.dir);

		if (denom == 0.f)
			return false;
		float t = dot(center - ray.origin, normal) / denom;

		if (t < ray.tmin || t >= std::min(rec.t, ray.tmax))
			return false;

		Vec3 p = ray.at(t);
		Vec3 op = p - center;

		if (dot(op, op) > radius * radius)
			return false; // outside disk

		Vec3 n = normal;
		rec.t = t;
		rec.pos = p;
		rec.geo_normal = n;
		rec.normal = n;
		rec.front_face = denom < 0.f;
		rec.mat_id = mat_id;

		// Polar UV: u = angle / (2π), v = radial distance / radius
		// Build a local 2D frame in the disk plane
		Vec3 ref = (std::abs(normal.x) < 0.9f) ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
		Vec3 tang = normalize(cross(normal, ref));
		Vec3 bita = cross(normal, tang);
		float px = dot(op, tang);
		float py = dot(op, bita);
		float phi = std::atan2(py, px);

		if (phi < 0.f)
			phi += 2.f * kPi;
		rec.u = phi * (0.5f * kInvPi);
		rec.v = std::sqrt(dot(op, op)) / radius;

		return true;
	}

	AABB aabb() const noexcept {
		// The disk fits inside a sphere of `radius` centered at `center`;
		// but that wastes space. Build a tighter box by projecting the disk.
		Vec3 e = {radius * std::sqrt(std::max(0.f, 1.f - normal.x * normal.x)),
				  radius * std::sqrt(std::max(0.f, 1.f - normal.y * normal.y)),
				  radius * std::sqrt(std::max(0.f, 1.f - normal.z * normal.z))};
		// Pad by 1mm to avoid zero-thickness AABB on axis-aligned disks
		e = max3(e, Vec3(0.001f));

		return {center - e, center + e};
	}
};

// ─────────────────────────────────────────────────────────────────────────────
// Quad  (parallelogram, defined by origin + two edge vectors)
//   Points on quad: P = origin + s*u_vec + t*v_vec,  s,t ∈ [0,1]
// ─────────────────────────────────────────────────────────────────────────────
struct Quad {
	Vec3 origin; // one corner
	Vec3 u_vec;	 // first edge vector
	Vec3 v_vec;	 // second edge vector
	int mat_id;

	[[nodiscard]] bool intersect(const Ray &ray, HitRecord &rec) const noexcept {
		Vec3 n_raw = cross(u_vec, v_vec); // not necessarily unit length
		float area_sq = dot(n_raw, n_raw);

		if (area_sq == 0.f)
			return false;

		float denom = dot(n_raw, ray.dir);

		if (denom == 0.f)
			return false;
		float t = dot(n_raw, origin - ray.origin) / denom;

		if (t < ray.tmin || t >= std::min(rec.t, ray.tmax))
			return false;

		// Project hit point onto quad's local frame to find barycentric (s, tv)
		Vec3 p = ray.at(t);
		Vec3 op = p - origin;

		// s = (p - origin) × v_vec · n̂ / |u × v|,  using triple products
		float s = dot(cross(op, v_vec), n_raw) / area_sq;
		float tv = dot(cross(u_vec, op), n_raw) / area_sq;

		if (s < 0.f || s > 1.f || tv < 0.f || tv > 1.f)
			return false;

		Vec3 n = normalize(n_raw);

		rec.t = t;
		rec.pos = p;
		rec.geo_normal = n;
		rec.normal = n;
		rec.front_face = denom < 0.f;
		rec.mat_id = mat_id;
		rec.u = s;
		rec.v = tv;

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
