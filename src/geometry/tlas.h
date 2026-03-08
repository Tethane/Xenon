#pragma once
// geometry/tlas.h — Top-Level Acceleration Structure
//
// A BVH over `Instance` objects.  Each instance holds:
//   • a GeomHandle (MESH_BLAS or PRIM_BLAS — zero-overhead tagged union)
//   • an AffineTransform (local ↔ world)
//   • a pre-computed world-space AABB (built from 8 transformed local AABB corners)
//
// Traversal:
//   Closest-hit — scalar BVH walk, ordered by child near-t.
//     For each leaf instance the ray is transformed to local space,
//     GeomHandle::intersect() is called, then position and normals are
//     mapped back to world space.
//   Any-hit    — same walk but exits immediately on first geometry hit.
//
// HitRecord semantics are preserved exactly:
//   rec.t         — parametric distance along the WORLD ray (t_local == t_world)
//   rec.pos       — world-space position  (recomputed as world_ray.at(rec.t))
//   rec.normal    — shading normal, world space, normalised
//   rec.geo_normal — geometric normal, world space, normalised
//   rec.front_face — recomputed from world-space geo_normal · world ray.dir
//   rec.mat_id, rec.prim_id, rec.u, rec.v — unchanged from geometry
//   rec.instance_id — set to Instance::instance_id on every hit

#include "geometry/aabb.h"
#include "geometry/blas.h"
#include "geometry/prim_blas.h"
#include "geometry/transform.h"
#include <vector>
#include <cstdint>

namespace xn {

// ─── GeomHandle — zero-overhead tagged pointer to BLAS or PrimBLAS ────────────
//
// Design rationale: both BLAS and PrimBLAS expose the same three-function API
// (intersect, intersects, root_aabb). Rather than a vtable, we store a type tag
// and a union of raw pointers.  The branch is a single integer comparison on a
// register value — branch-predictor friendly and zero heap overhead.
struct GeomHandle {
    enum Type : uint8_t { MESH_BLAS = 0, PRIM_BLAS = 1 } type = MESH_BLAS;
    union {
        const BLAS*     mesh;
        const PrimBLAS* prim;
    };

    // Construct from a mesh BLAS pointer
    static GeomHandle from_mesh(const BLAS* b) noexcept {
        GeomHandle h; h.type = MESH_BLAS; h.mesh = b; return h;
    }
    // Construct from a prim BLAS pointer
    static GeomHandle from_prim(const PrimBLAS* p) noexcept {
        GeomHandle h; h.type = PRIM_BLAS; h.prim = p; return h;
    }

    bool valid() const noexcept { return type == MESH_BLAS ? (mesh != nullptr) : (prim != nullptr); }

    AABB root_aabb() const noexcept {
        if (type == MESH_BLAS) return mesh ? mesh->root_aabb() : AABB{};
        return prim ? prim->root_aabb() : AABB{};
    }

    [[nodiscard]] bool intersect(const Ray& r, HitRecord& rec) const {
        if (type == MESH_BLAS) return mesh && mesh->intersect(r, rec);
        return prim && prim->intersect(r, rec);
    }

    [[nodiscard]] bool intersects(const Ray& r) const {
        if (type == MESH_BLAS) return mesh && mesh->intersects(r);
        return prim && prim->intersects(r);
    }
};

// ─── Instance ─────────────────────────────────────────────────────────────────
struct Instance {
    GeomHandle      geom;               // geometry (mesh or prim BLAS)
    AffineTransform xform;              // default: identity
    AABB            world_aabb;         // pre-computed, updated by rebuild_world_aabb()
    int             instance_id = 0;

    // (Re)compute world_aabb from the geometry root AABB and current transform.
    // Call this after changing xform.
    void rebuild_world_aabb() {
        world_aabb = xform.world_aabb(geom.root_aabb());
    }
};

// ─── TLASNode — 32 bytes ──────────────────────────────────────────────────────
struct alignas(32) TLASNode {
    AABB     bbox;    // world-space
    uint32_t child;   // internal: left child (right = child+1); leaf: first instance slot
    uint32_t count;   // 0 → internal; >0 → leaf instance count
    bool is_leaf() const noexcept { return count > 0; }
};
static_assert(sizeof(TLASNode) == 32);

// ─── TLAS ─────────────────────────────────────────────────────────────────────
class TLAS {
public:
    TLAS() = default;

    // Build a BVH over the supplied instances using binned SAH.
    // `instances` are moved in (BLAS pointers not owned).
    void build(std::vector<Instance> instances);

    // Add a single instance and rebuild (convenience; prefer batch build).
    void add(Instance inst);
    void rebuild();

    // Closest-hit query in world space.
    [[nodiscard]] bool intersect(const Ray& world_ray, HitRecord& rec) const;

    // Any-hit query in world space — fast shadow test.
    [[nodiscard]] bool intersects(const Ray& world_ray) const;

    // ── GPU upload accessors ─────────────────────────────────────────────────
    const std::vector<TLASNode>&  nodes()        const noexcept { return nodes_; }
    const std::vector<uint32_t>&  inst_indices() const noexcept { return inst_indices_; }
    const std::vector<Instance>&  instances()    const noexcept { return instances_; }

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
