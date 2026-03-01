#pragma once
// geometry/mesh.h — Triangle mesh with SoA layout for cache-efficient access

#include "math/vec3.h"
#include "math/ray.h"
#include "geometry/aabb.h"
#include "geometry/primitives.h"
#include <vector>
#include <string>
#include <cstdint>

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
    // Face (triangle) index buffer — 3 indices per triangle
    std::vector<int32_t> indices; // size = num_triangles * 3

    int mat_id = 0;

    int num_vertices()  const { return (int)vx.size(); }
    int num_triangles() const { return (int)indices.size() / 3; }

    // Fetch triangle vertex positions by triangle index
    void get_triangle(int tri_idx, Vec3& p0, Vec3& p1, Vec3& p2) const {
        int i0 = indices[tri_idx*3+0];
        int i1 = indices[tri_idx*3+1];
        int i2 = indices[tri_idx*3+2];
        p0 = {vx[i0], vy[i0], vz[i0]};
        p1 = {vx[i1], vy[i1], vz[i1]};
        p2 = {vx[i2], vy[i2], vz[i2]};
    }

    // Fetch interpolated (smooth) normal via barycentrics
    Vec3 interpolate_normal(int tri_idx, float u, float v) const {
        int i0 = indices[tri_idx*3+0];
        int i1 = indices[tri_idx*3+1];
        int i2 = indices[tri_idx*3+2];
        Vec3 n0 = {nx[i0], ny[i0], nz[i0]};
        Vec3 n1 = {nx[i1], ny[i1], nz[i1]};
        Vec3 n2 = {nx[i2], ny[i2], nz[i2]};
        return normalize(n0*(1.f-u-v) + n1*u + n2*v);
    }

    // Triangle geometric normal (flat)
    Vec3 geo_normal(int tri_idx) const {
        Vec3 p0, p1, p2;
        get_triangle(tri_idx, p0, p1, p2);
        return normalize(cross(p1-p0, p2-p0));
    }

    AABB triangle_aabb(int tri_idx) const {
        Vec3 p0, p1, p2;
        get_triangle(tri_idx, p0, p1, p2);
        AABB b;
        b.expand(p0); b.expand(p1); b.expand(p2);
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
        vx.reserve(nv); vy.reserve(nv); vz.reserve(nv);
        nx.reserve(nv); ny.reserve(nv); nz.reserve(nv);
        indices.reserve(nt*3);
    }

    void add_vertex(Vec3 p) {
        vx.push_back(p.x); vy.push_back(p.y); vz.push_back(p.z);
        nx.push_back(0);   ny.push_back(0);   nz.push_back(0);
    }

    void add_triangle(int i0, int i1, int i2) {
        indices.push_back(i0);
        indices.push_back(i1);
        indices.push_back(i2);
    }

    // Ray-triangle intersection against a specific triangle
    bool intersect_triangle(const Ray& ray, int tri_idx, HitRecord& rec) const {
        Vec3 p0, p1, p2;
        get_triangle(tri_idx, p0, p1, p2);
        float t, u, v;
        if (!ray_triangle(ray, p0, p1, p2, t, u, v, rec.t)) return false;

        rec.t      = t;
        rec.pos    = ray.at(t);
        rec.u      = u;
        rec.v      = v;
        rec.mat_id = mat_id;
        rec.prim_id = tri_idx;

        // Geometric normal
        Vec3 gn = normalize(cross(p1-p0, p2-p0));
        rec.geo_normal  = gn;
        rec.front_face  = dot(gn, ray.dir) < 0.f;

        // Shading normal: smooth if available
        if (!nx.empty()) {
            rec.normal = interpolate_normal(tri_idx, u, v);
            if (!rec.front_face) rec.normal = -rec.normal;
        } else {
            rec.normal = rec.front_face ? gn : -gn;
        }
        return true;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// OBJ Loader — loads positions, normals, and triangle faces
// Returns false on failure. Sets mesh.mat_id = mat_id argument.
// ─────────────────────────────────────────────────────────────────────────────
[[nodiscard]] bool load_obj(const std::string& path, TriangleMesh& mesh, int mat_id = 0);

} // namespace xn
