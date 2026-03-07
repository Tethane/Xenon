#pragma once
// geometry/tlas.h — Top-Level Acceleration Structure
//
// A BVH over `Instance` objects.  Each instance holds:
//   • a pointer to a BLAS (shared; owned elsewhere)
//   • an AffineTransform (local ↔ world)
//   • a pre-computed world-space AABB (built from 8 transformed local AABB corners)
//
// Traversal:
//   Closest-hit — scalar BVH walk, ordered by child near-t.
//     For each leaf instance the ray is transformed to local space,
//     BLAS::intersect() is called, then position and normals are
//     mapped back to world space.
//   Any-hit    — same walk but exits immediately on first BLAS hit.
//
// HitRecord semantics are preserved exactly:
//   rec.t         — parametric distance along the WORLD ray (t_local == t_world)
//   rec.pos       — world-space position  (recomputed as world_ray.at(rec.t))
//   rec.normal    — shading normal, world space, normalised
//   rec.geo_normal — geometric normal, world space, normalised
//   rec.front_face — recomputed from world-space geo_normal · world ray.dir
//   rec.mat_id, rec.prim_id, rec.u, rec.v — unchanged from BLAS

#include "geometry/aabb.h"
#include "geometry/blas.h"
#include "geometry/transform.h"
#include <vector>
#include <cstdint>

namespace xn {

// ─── Instance ─────────────────────────────────────────────────────────────────
struct Instance {
    const BLAS*     blas       = nullptr;
    AffineTransform xform;          // default: identity
    AABB            world_aabb;     // pre-computed, updated by rebuild_world_aabb()
    int             instance_id = 0;

    // (Re)compute world_aabb from the BLAS root AABB and current transform.
    // Call this after changing xform.
    void rebuild_world_aabb() {
        if (blas) world_aabb = xform.world_aabb(blas->root_aabb());
    }
};

// ─── TLASNode — 32 bytes ──────────────────────────────────────────────────────
struct alignas(32) TLASNode {
    AABB     bbox;    // world-space
    uint32_t child;   // internal: left child (right = child+1); leaf: first instance slot
    uint32_t count;   // 0 → internal; > 0 → leaf instance count
    bool is_leaf() const noexcept { return count > 0; }
};
static_assert(sizeof(TLASNode) == 32);

// ─── TLAS ─────────────────────────────────────────────────────────────────────
class TLAS {
public:
    TLAS() = default;

    // Build a BVH over the supplied instances using binned SAH.
    // `instances` are copied (BLAS pointers not owned).
    void build(std::vector<Instance> instances);

    // Add a single instance and rebuild (convenience; prefer batch build).
    void add(Instance inst);
    void rebuild();

    // Closest-hit query in world space.
    [[nodiscard]] bool intersect(const Ray& world_ray, HitRecord& rec) const;

    // Any-hit query in world space — fast shadow test.
    [[nodiscard]] bool intersects(const Ray& world_ray) const;

private:
    std::vector<Instance>  instances_;
    std::vector<TLASNode>  nodes_;
    std::vector<uint32_t>  inst_indices_;  // indirection into instances_

    struct Prim { AABB bbox; Vec3 center; uint32_t inst_idx; };

    uint32_t alloc_node();
    void subdivide    (uint32_t node, uint32_t lo, uint32_t hi, std::vector<Prim>& ps);
    void make_leaf    (uint32_t node, uint32_t lo, uint32_t hi, std::vector<Prim>& ps);
    void refit_bounds (uint32_t node, uint32_t lo, uint32_t hi, const std::vector<Prim>& ps);

    // Apply transform fixup to a local HitRecord to produce world-space values.
    static void fixup_hit(HitRecord& rec, const Ray& world_ray,
                          const AffineTransform& xform) noexcept;
};

} // namespace xn
