// geometry/blas.cpp — BLAS build (binned SAH) and traversal
//
// SIMD policy:
//   aabb_pair_sse() tests both BVH children in one call by running two
//   independent __m128 computation chains.  The chains share vInv/vOrg loads
//   but are otherwise fully independent, so an OOO processor issues them
//   in parallel — giving ~2× AABB throughput with no wider tree or extra
//   memory layout changes.  All other code remains scalar.

#include "geometry/blas.h"
#include <algorithm>
#include <cassert>
#include <immintrin.h>   // SSE2 (always available on x86-64)

namespace xn {

// ─── SSE 2-child AABB slab test ──────────────────────────────────────────────
// Tests AABBs `a` and `b` against the same ray simultaneously.
// inv    : precomputed 1/ray.dir (component-wise)
// org    : ray.origin
// tmin   : ray.tmin (or 0)
// tmax   : current best hit distance (rec.t for closest-hit, ray.tmax for any-hit)
// tn[2]  : filled with the near-t for each child (used for traversal ordering)
// returns: bitmask — bit 0 = child a hit, bit 1 = child b hit
static int aabb_pair_sse(
        const AABB& a, const AABB& b,
        const Vec3& inv, const Vec3& org,
        float tmin, float tmax,
        float tn[2]) noexcept
{
    // Load shared ray data once
    const __m128 vInv = _mm_set_ps(0.f, inv.z, inv.y, inv.x);
    const __m128 vOrg = _mm_set_ps(0.f, org.z, org.y, org.x);

    // ── Child A slab test ────────────────────────────────────────────────────
    const __m128 aLo  = _mm_set_ps(0.f, a.mn.z, a.mn.y, a.mn.x);
    const __m128 aHi  = _mm_set_ps(0.f, a.mx.z, a.mx.y, a.mx.x);
    const __m128 tA0  = _mm_mul_ps(_mm_sub_ps(aLo, vOrg), vInv);   // (lo-org)*inv
    const __m128 tA1  = _mm_mul_ps(_mm_sub_ps(aHi, vOrg), vInv);   // (hi-org)*inv
    const __m128 taLo = _mm_min_ps(tA0, tA1);   // per-axis slab near
    const __m128 taHi = _mm_max_ps(tA0, tA1);   // per-axis slab far

    // ── Child B slab test  (data-independent → executes in parallel via ILP) ─
    const __m128 bLo  = _mm_set_ps(0.f, b.mn.z, b.mn.y, b.mn.x);
    const __m128 bHi  = _mm_set_ps(0.f, b.mx.z, b.mx.y, b.mx.x);
    const __m128 tB0  = _mm_mul_ps(_mm_sub_ps(bLo, vOrg), vInv);
    const __m128 tB1  = _mm_mul_ps(_mm_sub_ps(bHi, vOrg), vInv);
    const __m128 tbLo = _mm_min_ps(tB0, tB1);
    const __m128 tbHi = _mm_max_ps(tB0, tB1);

    // ── Horizontal reduce: max(x,y,z) for near; min(x,y,z) for far ──────────
    // _MM_SHUFFLE(3,3,0,1) swaps lanes 0↔1, keeping lane 3 quiet (= 0 padding).
    // After two passes the result lands in lane 0.
    auto hmax3 = [](const __m128 v) noexcept -> __m128 {
        const __m128 h = _mm_max_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(3,3,0,1)));
        return _mm_max_ps(h, _mm_shuffle_ps(v, v, _MM_SHUFFLE(3,3,3,2)));
    };
    auto hmin3 = [](const __m128 v) noexcept -> __m128 {
        const __m128 h = _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(3,3,0,1)));
        return _mm_min_ps(h, _mm_shuffle_ps(v, v, _MM_SHUFFLE(3,3,3,2)));
    };

    const __m128 vTmin = _mm_set1_ps(tmin);
    const __m128 vTmax = _mm_set1_ps(tmax);

    // Clamp to ray interval
    const float tNearA = _mm_cvtss_f32(_mm_max_ps(hmax3(taLo), vTmin));
    const float tFarA  = _mm_cvtss_f32(_mm_min_ps(hmin3(taHi), vTmax));
    const float tNearB = _mm_cvtss_f32(_mm_max_ps(hmax3(tbLo), vTmin));
    const float tFarB  = _mm_cvtss_f32(_mm_min_ps(hmin3(tbHi), vTmax));

    tn[0] = tNearA; tn[1] = tNearB;
    return (tNearA <= tFarA ? 1 : 0) | (tNearB <= tFarB ? 2 : 0);
}

// ─── Build helpers ────────────────────────────────────────────────────────────
uint32_t BLAS::alloc_node() {
    nodes_.emplace_back();
    return static_cast<uint32_t>(nodes_.size() - 1);
}

void BLAS::refit_bounds(uint32_t node, uint32_t lo, uint32_t hi,
                        const std::vector<Prim>& ps) {
    AABB b;
    for (uint32_t i = lo; i < hi; ++i) b.expand(ps[i].bbox);
    nodes_[node].bbox = b;
}

void BLAS::make_leaf(uint32_t node, uint32_t lo, uint32_t hi,
                     std::vector<Prim>& ps) {
    nodes_[node].child = lo;
    nodes_[node].count = hi - lo;
    for (uint32_t i = lo; i < hi; ++i)
        prims_[i] = ps[i].tri_idx;
}

void BLAS::subdivide(uint32_t node, uint32_t lo, uint32_t hi,
                     std::vector<Prim>& ps) {
    refit_bounds(node, lo, hi, ps);
    const uint32_t count = hi - lo;

    // ── Leaf threshold ───────────────────────────────────────────────────────
    if (count <= 2) { make_leaf(node, lo, hi, ps); return; }

    // ── Choose split axis via centroid extent ────────────────────────────────
    AABB centBounds;
    for (uint32_t i = lo; i < hi; ++i) centBounds.expand(ps[i].center);
    const int axis = centBounds.max_extent_axis();

    // Degenerate: all centroids coincide — force leaf
    const float axisExtent = centBounds.extent()[axis];
    if (axisExtent == 0.f) { make_leaf(node, lo, hi, ps); return; }

    // ── Binned SAH ───────────────────────────────────────────────────────────
    constexpr int kBins = 12;
    struct Bin { AABB bounds; int count = 0; };
    Bin bins[kBins];

    const float scale = static_cast<float>(kBins) / axisExtent;
    for (uint32_t i = lo; i < hi; ++i) {
        int b = static_cast<int>((ps[i].center[axis] - centBounds.mn[axis]) * scale);
        if (b >= kBins) b = kBins - 1;
        bins[b].count++;
        bins[b].bounds.expand(ps[i].bbox);
    }

    float leftArea[kBins-1], rightArea[kBins-1];
    int   leftCnt [kBins-1], rightCnt [kBins-1];

    AABB  lb; int ls = 0;
    for (int i = 0; i < kBins-1; ++i) {
        ls += bins[i].count; leftCnt[i] = ls;
        lb.expand(bins[i].bounds); leftArea[i] = lb.surface_area();
    }
    AABB  rb; int rs = 0;
    for (int i = kBins-1; i > 0; --i) {
        rs += bins[i].count; rightCnt[i-1] = rs;
        rb.expand(bins[i].bounds); rightArea[i-1] = rb.surface_area();
    }

    const float parentArea = nodes_[node].bbox.surface_area();
    const float leafCost   = static_cast<float>(count);   // relative cost of a leaf here
    float bestCost = kInfinity; int bestBin = -1;
    for (int i = 0; i < kBins-1; ++i) {
        const float c = (leftArea[i]*leftCnt[i] + rightArea[i]*rightCnt[i]) / parentArea;
        if (c < bestCost) { bestCost = c; bestBin = i; }
    }

    // If leaf is cheaper than any split, collapse to leaf
    if (bestCost >= leafCost || bestBin < 0) {
        make_leaf(node, lo, hi, ps);
        return;
    }

    // ── Partition prims around best split ────────────────────────────────────
    auto it = std::partition(ps.begin() + lo, ps.begin() + hi,
        [&](const Prim& p) {
            int b = static_cast<int>((p.center[axis] - centBounds.mn[axis]) * scale);
            if (b >= kBins) b = kBins - 1;
            return b <= bestBin;
        });
    uint32_t mid = static_cast<uint32_t>(std::distance(ps.begin(), it));

    // Guard against degenerate partition (all prims on one side)
    if (mid == lo || mid == hi) {
        mid = lo + count / 2;
        std::nth_element(ps.begin()+lo, ps.begin()+mid, ps.begin()+hi,
            [axis](const Prim& a, const Prim& b){ return a.center[axis] < b.center[axis]; });
    }

    // Allocate left+right children (must be contiguous: right = left+1)
    const uint32_t left = alloc_node();
    alloc_node(); // right = left+1
    nodes_[node].child = left;
    nodes_[node].count = 0;   // mark as internal

    // Note: `nodes_` may have been reallocated — access only via index from here.
    subdivide(left,     lo,  mid, ps);
    subdivide(left + 1, mid, hi,  ps);
}

// ─── Public: build ────────────────────────────────────────────────────────────
void BLAS::build(const TriangleMesh& mesh) {
    mesh_ = &mesh;
    nodes_.clear();
    prims_.clear();

    const uint32_t nTris = static_cast<uint32_t>(mesh.num_triangles());
    if (nTris == 0) return;

    prims_.resize(nTris);
    nodes_.reserve(nTris * 2);

    std::vector<Prim> ps(nTris);
    for (uint32_t i = 0; i < nTris; ++i) {
        ps[i].bbox    = mesh.triangle_aabb(i);
        ps[i].center  = ps[i].bbox.center();
        ps[i].tri_idx = i;
    }

    alloc_node(); // root = 0
    subdivide(0, 0, nTris, ps);
}

const AABB& BLAS::root_aabb() const noexcept {
    static const AABB empty;
    return nodes_.empty() ? empty : nodes_[0].bbox;
}

// ─── Traversal: closest-hit ───────────────────────────────────────────────────
bool BLAS::intersect(const Ray& ray, HitRecord& rec) const {
    if (nodes_.empty()) return false;

    // Precompute reciprocal direction for slab tests
    const Vec3 inv_dir = {
        1.f / ray.dir.x, 1.f / ray.dir.y, 1.f / ray.dir.z
    };

    uint32_t stack[64];
    int      ptr  = 0;
    stack[ptr++]  = 0;
    bool     hit  = false;

    while (ptr > 0) {
        const uint32_t  idx  = stack[--ptr];
        const BLASNode& node = nodes_[idx];

        // ── Leaf: intersect triangles ─────────────────────────────────────────
        if (node.is_leaf()) {
            for (uint32_t i = 0; i < node.count; ++i) {
                if (mesh_->intersect_triangle(ray, prims_[node.child + i], rec))
                    hit = true;
                // rec.t is updated on each closer hit → later AABB culls use it
            }
            continue;
        }

        // ── Internal: SSE 2-child test then ordered push ──────────────────────
        const uint32_t  left  = node.child;
        const uint32_t  right = left + 1;
        float tn[2];
        const int mask = aabb_pair_sse(
            nodes_[left].bbox, nodes_[right].bbox,
            inv_dir, ray.origin,
            ray.tmin, rec.t,    // rec.t shrinks as we find closer hits → tighter culling
            tn);

        if (mask == 0) continue;  // neither child hit

        if (mask == 3) {
            // Both hit — push farther child first so nearer is processed next
            if (tn[0] < tn[1]) {
                stack[ptr++] = right;
                stack[ptr++] = left;
            } else {
                stack[ptr++] = left;
                stack[ptr++] = right;
            }
        } else {
            // Only one child hit
            stack[ptr++] = (mask & 1) ? left : right;
        }
    }
    return hit;
}

// ─── Traversal: any-hit ───────────────────────────────────────────────────────
// No ordering needed — we exit on the very first geometry hit.
bool BLAS::intersects(const Ray& ray) const {
    if (nodes_.empty()) return false;

    const Vec3 inv_dir = {
        1.f / ray.dir.x, 1.f / ray.dir.y, 1.f / ray.dir.z
    };

    uint32_t stack[64];
    int      ptr  = 0;
    stack[ptr++]  = 0;

    while (ptr > 0) {
        const uint32_t  idx  = stack[--ptr];
        const BLASNode& node = nodes_[idx];

        if (node.is_leaf()) {
            HitRecord tmp; tmp.t = ray.tmax;
            for (uint32_t i = 0; i < node.count; ++i) {
                if (mesh_->intersect_triangle(ray, prims_[node.child + i], tmp))
                    return true;   // early exit — no need to find closest
            }
            continue;
        }

        const uint32_t left  = node.child;
        const uint32_t right = left + 1;
        float tn[2];
        const int mask = aabb_pair_sse(
            nodes_[left].bbox, nodes_[right].bbox,
            inv_dir, ray.origin,
            ray.tmin, ray.tmax,   // fixed tmax — we just need any hit
            tn);

        // Push both hit children (unordered — any-hit doesn't care about order)
        if (mask & 2) stack[ptr++] = right;
        if (mask & 1) stack[ptr++] = left;
    }
    return false;
}

} // namespace xn
