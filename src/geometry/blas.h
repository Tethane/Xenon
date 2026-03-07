#pragma once
// geometry/blas.h — Bottom-Level Acceleration Structure
//
// Binary BVH over one TriangleMesh, built with binned SAH (12 bins).
// Traversal hot path tests both BVH children simultaneously via two
// independent SSE computation chains — yields ~2× AABB throughput
// through out-of-order ILP with no extra data layout changes.
//
// Ray space: all intersection calls expect rays in BLAS-local space.
//            The TLAS is responsible for transforming world rays before calling.

#include "geometry/aabb.h"
#include "geometry/mesh.h"
#include <vector>
#include <cstdint>

namespace xn {

// ─── BLASNode — 32 bytes (fits two nodes per cache line) ─────────────────────
struct alignas(32) BLASNode {
    AABB     bbox;   // 24 bytes — local-space bounding box
    uint32_t child;  // internal: left child index (right = child+1); leaf: first prim slot
    uint32_t count;  // 0 → internal node; > 0 → leaf (count triangles)

    bool is_leaf() const noexcept { return count > 0; }
};
static_assert(sizeof(BLASNode) == 32, "BLASNode must be exactly 32 bytes");

// ─── BLAS ─────────────────────────────────────────────────────────────────────
class BLAS {
public:
    BLAS() = default;
    ~BLAS() = default;

    BLAS(const BLAS&) = delete;
    BLAS& operator=(const BLAS&) = delete;
    BLAS(BLAS&&) = default;
    BLAS& operator=(BLAS&&) = default;

    // Build binned-SAH BVH over all triangles in `mesh`.
    // Vertex positions are used as-is (caller applies any mesh transform first).
    void build(const TriangleMesh& mesh);

    // Root AABB in local space — TLAS uses this to compute world AABB.
    const AABB& root_aabb() const noexcept;

    // ── Closest-hit query ────────────────────────────────────────────────────
    // Returns true if any triangle is hit in [ray.tmin, rec.t).
    // On success updates rec; on miss rec is unchanged.
    // rec.t should be initialised to the ray's far limit before the first call
    // (e.g. ray.tmax).
    [[nodiscard]] bool intersect(const Ray& ray, HitRecord& rec) const;

    // ── Any-hit query (shadow / occlusion) ───────────────────────────────────
    // Returns true immediately on the first intersection found in [ray.tmin, ray.tmax].
    // Substantially faster than intersect() for shadow testing.
    [[nodiscard]] bool intersects(const Ray& ray) const;

private:
    const TriangleMesh* mesh_  = nullptr;
    std::vector<BLASNode>  nodes_;
    std::vector<uint32_t>  prims_;  // prims_[i] → mesh triangle index

    // Build helpers
    struct Prim { AABB bbox; Vec3 center; uint32_t tri_idx; };

    uint32_t alloc_node();
    void subdivide    (uint32_t node, uint32_t lo, uint32_t hi, std::vector<Prim>& ps);
    void make_leaf    (uint32_t node, uint32_t lo, uint32_t hi, std::vector<Prim>& ps);
    void refit_bounds (uint32_t node, uint32_t lo, uint32_t hi, const std::vector<Prim>& ps);
};

} // namespace xn
