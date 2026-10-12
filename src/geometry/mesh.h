#pragma once
// geometry/mesh.h — Triangle mesh with SoA layout for cache-efficient access

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "math/vec3.h"
#include "math/ray.h"
#include "geometry/aabb.h"
#include "geometry/primitives.h"

namespace xn {

// ─────────────────────────────────────────────────────────────────────────────
// TriangleMesh — SoA layout for maximum SIMD friendliness
//   All vertex data stored as parallel float arrays.
//   Triangles stored as index triplets (int32).
// ─────────────────────────────────────────────────────────────────────────────
struct TriangleMesh {
	// Vertex positions (SoA)
	std::vector<float> vx, vy, vz;
	// Vertex normals (SoA) — may be empty (computed on build)
	std::vector<float> nx, ny, nz;
	// Per-vertex UVs. OBJ vertex splitting preserves position/normal/UV seams.
	std::vector<float> tu, tv;
	std::vector<uint8_t> uv_valid;
	// Face (triangle) index buffer — 3 indices per triangle
	std::vector<int32_t> indices; // size = num_triangles * 3
	// Per-triangle material IDs
	std::vector<int32_t> mat_ids; // size = num_triangles

	int mat_id = 0;

	int num_vertices() const {
		return (int)vx.size();
	}

	int num_triangles() const {
		return (int)indices.size() / 3;
	}

	// Fetch triangle vertex positions by triangle index
	void get_triangle(int tri_idx, Vec3 &p0, Vec3 &p1, Vec3 &p2) const {
		int i0 = indices[tri_idx * 3 + 0];
		int i1 = indices[tri_idx * 3 + 1];
		int i2 = indices[tri_idx * 3 + 2];
		p0 = {vx[i0], vy[i0], vz[i0]};
		p1 = {vx[i1], vy[i1], vz[i1]};
		p2 = {vx[i2], vy[i2], vz[i2]};
	}

	// Fetch interpolated (smooth) normal via barycentrics
	Vec3 interpolate_normal(int tri_idx, float u, float v) const {
		int i0 = indices[tri_idx * 3 + 0];
		int i1 = indices[tri_idx * 3 + 1];
		int i2 = indices[tri_idx * 3 + 2];
		Vec3 n0 = {nx[i0], ny[i0], nz[i0]};
		Vec3 n1 = {nx[i1], ny[i1], nz[i1]};
		Vec3 n2 = {nx[i2], ny[i2], nz[i2]};

		return normalize(n0 * (1.f - u - v) + n1 * u + n2 * v);
	}

	// Triangle geometric normal (flat)
	Vec3 geo_normal(int tri_idx) const {
		Vec3 p0, p1, p2;
		get_triangle(tri_idx, p0, p1, p2);

		return normalize(cross(p1 - p0, p2 - p0));
	}

	AABB triangle_aabb(int tri_idx) const {
		Vec3 p0, p1, p2;
		get_triangle(tri_idx, p0, p1, p2);
		AABB b;
		b.expand(p0);
		b.expand(p1);
		b.expand(p2);

		return b;
	}

	// Compute AABB of whole mesh
	AABB compute_aabb() const {
		AABB b;

		for (int i = 0; i < num_vertices(); ++i)
			b.expand({vx[i], vy[i], vz[i]});

		return b;
	}

	// Generate smooth normals via area-weighted averaging
	void compute_smooth_normals();

	// Reserve space
	void reserve(int nv, int nt) {
		vx.reserve(nv);
		vy.reserve(nv);
		vz.reserve(nv);
		nx.reserve(nv);
		ny.reserve(nv);
		nz.reserve(nv);
		tu.reserve(nv);
		tv.reserve(nv);
		uv_valid.reserve(nv);
		indices.reserve(nt * 3);
		mat_ids.reserve(nt);
	}

	void add_vertex(Vec3 p) {
		vx.push_back(p.x);
		vy.push_back(p.y);
		vz.push_back(p.z);
		nx.push_back(0);
		ny.push_back(0);
		nz.push_back(0);
		tu.push_back(0);
		tv.push_back(0);
		uv_valid.push_back(0);
	}

	void set_uv(int vertex, float u, float v) {
		if (tu.size() != vx.size()) {
			tu.resize(vx.size());
			tv.resize(vx.size());
			uv_valid.resize(vx.size());
		}
		tu[vertex] = u;
		tv[vertex] = v;
		uv_valid[vertex] = 1;
	}

	bool triangle_has_uv(int tri) const {
		return uv_valid.size() == vx.size() && uv_valid[indices[3 * tri]] &&
			   uv_valid[indices[3 * tri + 1]] && uv_valid[indices[3 * tri + 2]];
	}

	void surface_uv(int tri, float u, float v, HitRecord &rec) const {
		rec.has_uv = triangle_has_uv(tri);
		rec.tex_u = rec.tex_v = 0;
		rec.dpdu = rec.dpdv = Vec3(0.f);

		if (!rec.has_uv)
			return;
		int a = indices[3 * tri], b = indices[3 * tri + 1], c = indices[3 * tri + 2];
		rec.tex_u = tu[a] * (1 - u - v) + tu[b] * u + tu[c] * v;
		rec.tex_v = tv[a] * (1 - u - v) + tv[b] * u + tv[c] * v;
		float du1 = tu[b] - tu[a], dv1 = tv[b] - tv[a], du2 = tu[c] - tu[a], dv2 = tv[c] - tv[a];
		float det = du1 * dv2 - dv1 * du2;

		if (std::abs(det) > 1e-12f) {
			Vec3 p0, p1, p2;
			get_triangle(tri, p0, p1, p2);
			rec.dpdu = ((p1 - p0) * dv2 - (p2 - p0) * dv1) / det;
			rec.dpdv = ((p2 - p0) * du1 - (p1 - p0) * du2) / det;
		}
	}

	void add_triangle(int i0, int i1, int i2, int mid = -1) {
		indices.push_back(i0);
		indices.push_back(i1);
		indices.push_back(i2);
		mat_ids.push_back(mid >= 0 ? mid : mat_id);
	}

	// Ray-triangle intersection against a specific triangle
	bool intersect_triangle(const Ray &ray, int tri_idx, HitRecord &rec) const {
		Vec3 p0, p1, p2;
		get_triangle(tri_idx, p0, p1, p2);
		float t, u, v;

		if (!ray_triangle(ray, p0, p1, p2, t, u, v, rec.t))
			return false;

		rec.t = t;
		rec.pos = ray.at(t);
		rec.u = u;
		rec.v = v;
		rec.mat_id = mat_ids[tri_idx];
		rec.prim_id = tri_idx;
		surface_uv(tri_idx, u, v, rec);

		// Geometric normal (stable, world space)
		Vec3 gn = normalize(cross(p1 - p0, p2 - p0));
		rec.geo_normal = gn;
		rec.front_face = dot(gn, ray.dir) < 0.f; // hitting the front side

		// Shading normal (stable)
		if (!nx.empty()) {
			rec.normal = interpolate_normal(tri_idx, u, v);
		} else {
			rec.normal = gn;
		}

		if (rec.normal.length_sq() == 0.f)
			rec.normal = gn;

		if (dot(rec.normal, gn) < 0.f)
			rec.normal = -rec.normal;

		return true;
	}

	// Transform mesh: Scale -> Rotate YXZ -> Translate
	void transform(Vec3 pos, float scale, Vec3 rot_deg);

	// Merge another mesh into this one
	void merge(const TriangleMesh &other);
};

// ─────────────────────────────────────────────────────────────────────────────
// OBJ Loader — loads positions, normals, and triangle faces
// Returns false on failure. Sets mesh.mat_id = mat_id argument.
// ─────────────────────────────────────────────────────────────────────────────
[[nodiscard]] bool load_obj(const std::string &path, TriangleMesh &mesh,
							const std::map<std::string, int> &mat_map, int default_mat_id = 0);

} // namespace xn
