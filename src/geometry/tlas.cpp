// geometry/tlas.cpp — TLAS build (binned SAH) and traversal
//
// Traversal is scalar: the TLAS typically holds a small number of instances
// so SIMD per-TLAS-node gains are negligible.  The real inner loop is inside
// BLAS::intersect(), which already uses the SSE 2-child test.
//
// HitRecord correctness notes (see tlas.h header for full explanation):
//   • t is invariant under our transform (unnormalised local ray direction).
//   • pos  is recomputed as world_ray.at(rec.t) — cheaper than transforming.
//   • normal / geo_normal: apply (A⁻ᵀ)·n via world_to_local.transform_normal.
//   • front_face: re-derived from world-space dot product after normal fixup.

#include "geometry/tlas.h"
#include <algorithm>
#include <cassert>

namespace xn {

// ─── Build helpers ────────────────────────────────────────────────────────────
uint32_t TLAS::alloc_node() {
    nodes_.emplace_back();
    return static_cast<uint32_t>(nodes_.size() - 1);
}

void TLAS::refit_bounds(uint32_t node, uint32_t lo, uint32_t hi,
                        const std::vector<Prim>& ps) {
    AABB b;
    for (uint32_t i = lo; i < hi; ++i) b.expand(ps[i].bbox);
    nodes_[node].bbox = b;
}

void TLAS::make_leaf(uint32_t node, uint32_t lo, uint32_t hi,
                     std::vector<Prim>& ps) {
    nodes_[node].child = lo;
    nodes_[node].count = hi - lo;
    for (uint32_t i = lo; i < hi; ++i)
        inst_indices_[i] = ps[i].inst_idx;
}

void TLAS::subdivide(uint32_t node, uint32_t lo, uint32_t hi,
                     std::vector<Prim>& ps) {
    refit_bounds(node, lo, hi, ps);
    const uint32_t count = hi - lo;

    // ── Leaf: a single instance per leaf keeps traversal simple ─────────────
    if (count <= 1) { make_leaf(node, lo, hi, ps); return; }

    // ── Choose split axis ────────────────────────────────────────────────────
    AABB centBounds;
    for (uint32_t i = lo; i < hi; ++i) centBounds.expand(ps[i].center);
    const int   axis        = centBounds.max_extent_axis();
    const float axisExtent  = centBounds.extent()[axis];

    if (axisExtent == 0.f) { make_leaf(node, lo, hi, ps); return; }

    // ── Binned SAH ───────────────────────────────────────────────────────────
    constexpr int kBins = 8;  // fewer bins — TLAS has fewer prims
    struct Bin { AABB bounds; int count = 0; };
    Bin bins[kBins];

    const float bscale = static_cast<float>(kBins) / axisExtent;
    for (uint32_t i = lo; i < hi; ++i) {
        int b = static_cast<int>((ps[i].center[axis] - centBounds.mn[axis]) * bscale);
        if (b >= kBins) b = kBins - 1;
        bins[b].count++;
        bins[b].bounds.expand(ps[i].bbox);
    }

    float lArea[kBins-1], rArea[kBins-1];
    int   lCnt [kBins-1], rCnt [kBins-1];
    AABB  lb; int ls = 0;
    for (int i = 0; i < kBins-1; ++i) {
        ls += bins[i].count; lCnt[i] = ls;
        lb.expand(bins[i].bounds); lArea[i] = lb.surface_area();
    }
    AABB  rb; int rs = 0;
    for (int i = kBins-1; i > 0; --i) {
        rs += bins[i].count; rCnt[i-1] = rs;
        rb.expand(bins[i].bounds); rArea[i-1] = rb.surface_area();
    }

    const float parentArea = nodes_[node].bbox.surface_area();
    float bestCost = kInfinity; int bestBin = -1;
    for (int i = 0; i < kBins-1; ++i) {
        const float c = (lArea[i]*lCnt[i] + rArea[i]*rCnt[i]) / parentArea;
        if (c < bestCost) { bestCost = c; bestBin = i; }
    }

    if (bestBin < 0) { make_leaf(node, lo, hi, ps); return; }

    // ── Partition ────────────────────────────────────────────────────────────
    auto it = std::partition(ps.begin() + lo, ps.begin() + hi,
        [&](const Prim& p) {
            int b = static_cast<int>((p.center[axis] - centBounds.mn[axis]) * bscale);
            if (b >= kBins) b = kBins - 1;
            return b <= bestBin;
        });
    uint32_t mid = static_cast<uint32_t>(std::distance(ps.begin(), it));

    if (mid == lo || mid == hi) {
        mid = lo + count / 2;
        std::nth_element(ps.begin()+lo, ps.begin()+mid, ps.begin()+hi,
            [axis](const Prim& a, const Prim& b){ return a.center[axis] < b.center[axis]; });
    }

    const uint32_t left = alloc_node();
    alloc_node();  // right = left+1
    nodes_[node].child = left;
    nodes_[node].count = 0;

    subdivide(left,     lo,  mid, ps);
    subdivide(left + 1, mid, hi,  ps);
}

// ─── Public: build ────────────────────────────────────────────────────────────
void TLAS::build(std::vector<Instance> instances) {
    instances_ = std::move(instances);
    rebuild();
}

void TLAS::add(Instance inst) {
    instances_.push_back(std::move(inst));
}

void TLAS::rebuild() {
    nodes_.clear();
    inst_indices_.clear();

    const uint32_t n = static_cast<uint32_t>(instances_.size());
    if (n == 0) return;

    inst_indices_.resize(n);
    nodes_.reserve(n * 2);

    std::vector<Prim> ps(n);
    for (uint32_t i = 0; i < n; ++i) {
        instances_[i].rebuild_world_aabb();
        ps[i].bbox     = instances_[i].world_aabb;
        ps[i].center   = instances_[i].world_aabb.center();
        ps[i].inst_idx = i;
    }

    alloc_node();  // root = 0
    subdivide(0, 0, n, ps);
}

// ─── HitRecord world-space fixup ─────────────────────────────────────────────
// Called after BLAS::intersect() returns a local-space hit.
//
// t:         already correct (t_local == t_world — see transform.h comments).
// pos:       recomputed as world_ray.at(rec.t) — no transform needed.
// normals:   apply (A⁻ᵀ)·n  via world_to_local.transform_normal(), then normalise.
//            For TRS with uniform scale this reduces to the pure rotation.
// front_face: re-evaluated in world space after normal is fixed up.
// mat_id / prim_id / u / v: untouched.
void TLAS::fixup_hit(HitRecord& rec, const Ray& world_ray,
                     const AffineTransform& xform) noexcept {
    // Position — just use parametric form; avoids a redundant transform_point
    rec.pos        = world_ray.at(rec.t);

    // Normals — (A⁻ᵀ)·n, renormalised
    rec.normal     = xform.normal_to_world(rec.normal);
    rec.geo_normal = xform.normal_to_world(rec.geo_normal);

    // Re-derive front_face from the world-space geometry normal.
    // This is correct even if the instance has an orientation-reversing transform.
    rec.front_face = dot(rec.geo_normal, world_ray.dir) < 0.f;
}

// ─── Traversal: closest-hit ───────────────────────────────────────────────────
// Scalar TLAS walk + full BLAS descent per candidate instance.
bool TLAS::intersect(const Ray& world_ray, HitRecord& rec) const {
    if (nodes_.empty()) return false;

    // Precompute for TLAS AABB tests
    const Vec3 inv_dir = {
        1.f / world_ray.dir.x,
        1.f / world_ray.dir.y,
        1.f / world_ray.dir.z
    };
    const int sign[3] = {
        world_ray.dir.x < 0,
        world_ray.dir.y < 0,
        world_ray.dir.z < 0
    };

    uint32_t stack[64];
    int      ptr  = 0;
    stack[ptr++]  = 0;
    bool     hit  = false;

    while (ptr > 0) {
        const uint32_t   idx  = stack[--ptr];
        const TLASNode&  node = nodes_[idx];

        // TLAS AABB test — use existing scalar fast path (instances are few,
        // scalar vs SSE difference is negligible at this level)
        if (!node.bbox.intersect_fast(inv_dir, world_ray.origin,
                                      const_cast<int*>(sign),
                                      world_ray.tmin, rec.t))
            continue;

        if (node.is_leaf()) {
            // ── Per-instance intersection ────────────────────────────────────
            for (uint32_t i = 0; i < node.count; ++i) {
                const Instance& inst = instances_[inst_indices_[node.child + i]];
                if (!inst.geom.valid()) continue;

                // Transform ray to instance local space (t preserved by design)
                const Ray local_ray = inst.xform.to_local(world_ray);

                // Forward local rec.t so geometry culls with the current best distance.
                HitRecord local_rec = rec;
                if (inst.geom.intersect(local_ray, local_rec)) {
                    // Fix up world-space fields before accepting
                    fixup_hit(local_rec, world_ray, inst.xform);
                    local_rec.instance_id = inst.instance_id;
                    rec = local_rec;
                    hit = true;
                }
            }
        } else {
            // ── Ordered child push (near child processed first) ──────────────
            //    Use a simple scalar near-t estimate via AABB centre projection
            //    — full slab test already done above; ordering improves culling.
            const uint32_t left  = node.child;
            const uint32_t right = left + 1;

            float tl, tr;
            // Quick distance estimate: dot((bbox.center - origin), inv_dir) on split axis
            const int ax = node.bbox.max_extent_axis();
            tl = (nodes_[left ].bbox.center()[ax] - world_ray.origin[ax]) * inv_dir[ax];
            tr = (nodes_[right].bbox.center()[ax] - world_ray.origin[ax]) * inv_dir[ax];

            if (tl < tr) {
                stack[ptr++] = right;
                stack[ptr++] = left;
            } else {
                stack[ptr++] = left;
                stack[ptr++] = right;
            }
        }
    }
    return hit;
}

// ─── Traversal: any-hit ───────────────────────────────────────────────────────
bool TLAS::intersects(const Ray& world_ray) const {
    if (nodes_.empty()) return false;

    const Vec3 inv_dir = {
        1.f / world_ray.dir.x,
        1.f / world_ray.dir.y,
        1.f / world_ray.dir.z
    };
    const int sign[3] = {
        world_ray.dir.x < 0,
        world_ray.dir.y < 0,
        world_ray.dir.z < 0
    };

    uint32_t stack[64];
    int      ptr  = 0;
    stack[ptr++]  = 0;

    while (ptr > 0) {
        const uint32_t  idx  = stack[--ptr];
        const TLASNode& node = nodes_[idx];

        if (!node.bbox.intersect_fast(inv_dir, world_ray.origin,
                                      const_cast<int*>(sign),
                                      world_ray.tmin, world_ray.tmax))
            continue;

        if (node.is_leaf()) {
            for (uint32_t i = 0; i < node.count; ++i) {
                const Instance& inst = instances_[inst_indices_[node.child + i]];
                if (!inst.geom.valid()) continue;
                const Ray local_ray = inst.xform.to_local(world_ray);
                if (inst.geom.intersects(local_ray))
                    return true;    // any-hit — done
            }
        } else {
            // No ordering needed for any-hit
            stack[ptr++] = node.child;
            stack[ptr++] = node.child + 1;
        }
    }
    return false;
}

} // namespace xn
