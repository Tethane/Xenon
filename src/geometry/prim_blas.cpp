// geometry/prim_blas.cpp — PrimBLAS build and traversal
//
// Implementation mirrors blas.cpp closely: binned SAH with 12 bins,
// SSE-free scalar traversal (the TLAS is the outer loop; PrimBLAS
// typically holds few objects so SIMD overhead is not justified).

#include "geometry/prim_blas.h"
#include <algorithm>
#include <cassert>

namespace xn {

// ─── Build helpers ────────────────────────────────────────────────────────────
uint32_t PrimBLAS::alloc_node() {
    nodes_.emplace_back();
    return static_cast<uint32_t>(nodes_.size() - 1);
}

void PrimBLAS::refit_bounds(uint32_t node, uint32_t lo, uint32_t hi,
                             const std::vector<PrimEntry>& ps) {
    AABB b;
    for (uint32_t i = lo; i < hi; ++i) b.expand(ps[i].bbox);
    nodes_[node].bbox = b;
}

void PrimBLAS::make_leaf(uint32_t node, uint32_t lo, uint32_t hi,
                          std::vector<PrimEntry>& ps) {
    nodes_[node].child = lo;
    nodes_[node].count = hi - lo;
    for (uint32_t i = lo; i < hi; ++i)
        prims_[i] = ps[i].idx;
}

void PrimBLAS::subdivide(uint32_t node, uint32_t lo, uint32_t hi,
                          std::vector<PrimEntry>& ps) {
    refit_bounds(node, lo, hi, ps);
    const uint32_t count = hi - lo;

    if (count <= 2) { make_leaf(node, lo, hi, ps); return; }

    // ── Choose split axis ─────────────────────────────────────────────────────
    AABB centBounds;
    for (uint32_t i = lo; i < hi; ++i) centBounds.expand(ps[i].center);
    const int axis = centBounds.max_extent_axis();
    const float axisExtent = centBounds.extent()[axis];
    if (axisExtent == 0.f) { make_leaf(node, lo, hi, ps); return; }

    // ── Binned SAH (12 bins) ──────────────────────────────────────────────────
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

    float lArea[kBins-1], rArea[kBins-1];
    int   lCnt [kBins-1], rCnt [kBins-1];

    AABB lb; int ls = 0;
    for (int i = 0; i < kBins-1; ++i) {
        ls += bins[i].count; lCnt[i] = ls;
        lb.expand(bins[i].bounds); lArea[i] = lb.surface_area();
    }
    AABB rb; int rs = 0;
    for (int i = kBins-1; i > 0; --i) {
        rs += bins[i].count; rCnt[i-1] = rs;
        rb.expand(bins[i].bounds); rArea[i-1] = rb.surface_area();
    }

    const float parentArea = nodes_[node].bbox.surface_area();
    const float leafCost = static_cast<float>(count);
    float bestCost = kInfinity; int bestBin = -1;
    for (int i = 0; i < kBins-1; ++i) {
        const float c = (lArea[i]*lCnt[i] + rArea[i]*rCnt[i]) / parentArea;
        if (c < bestCost) { bestCost = c; bestBin = i; }
    }

    if (bestCost >= leafCost || bestBin < 0) {
        make_leaf(node, lo, hi, ps);
        return;
    }

    // ── Partition ─────────────────────────────────────────────────────────────
    auto it = std::partition(ps.begin() + lo, ps.begin() + hi,
        [&](const PrimEntry& p) {
            int b = static_cast<int>((p.center[axis] - centBounds.mn[axis]) * scale);
            if (b >= kBins) b = kBins - 1;
            return b <= bestBin;
        });
    uint32_t mid = static_cast<uint32_t>(std::distance(ps.begin(), it));

    if (mid == lo || mid == hi) {
        mid = lo + count / 2;
        std::nth_element(ps.begin()+lo, ps.begin()+mid, ps.begin()+hi,
            [axis](const PrimEntry& a, const PrimEntry& b){
                return a.center[axis] < b.center[axis]; });
    }

    const uint32_t left = alloc_node();
    alloc_node(); // right = left+1
    nodes_[node].child = left;
    nodes_[node].count = 0; // internal

    subdivide(left,     lo,  mid, ps);
    subdivide(left + 1, mid, hi,  ps);
}

// ─── Public: build ─────────────────────────────────────────────────────────────
void PrimBLAS::build(const PrimGroup& group) {
    group_ = &group;
    nodes_.clear();
    prims_.clear();

    const uint32_t n = static_cast<uint32_t>(group.size());
    if (n == 0) return;

    prims_.resize(n);
    nodes_.reserve(n * 2);

    std::vector<PrimEntry> ps(n);
    for (uint32_t i = 0; i < n; ++i) {
        ps[i].bbox   = prim_aabb(group.prims[i]);
        ps[i].center = ps[i].bbox.center();
        ps[i].idx    = i;
    }

    alloc_node(); // root = 0
    subdivide(0, 0, n, ps);
}

const AABB& PrimBLAS::root_aabb() const noexcept {
    static const AABB empty;
    return nodes_.empty() ? empty : nodes_[0].bbox;
}

// ─── Traversal: closest-hit ────────────────────────────────────────────────────
bool PrimBLAS::intersect(const Ray& ray, HitRecord& rec) const {
    if (nodes_.empty() || !group_) return false;

    const Vec3 inv_dir = {
        1.f / ray.dir.x, 1.f / ray.dir.y, 1.f / ray.dir.z
    };

    uint32_t stack[64];
    int      ptr = 0;
    stack[ptr++] = 0;
    bool hit = false;

    while (ptr > 0) {
        const uint32_t      idx  = stack[--ptr];
        const PrimBLASNode& node = nodes_[idx];

        // AABB cull — use rec.t as the current best hit distance
        float tnear, tfar;
        if (!node.bbox.intersect(ray, tnear, tfar) || tnear > rec.t) continue;

        if (node.is_leaf()) {
            for (uint32_t i = 0; i < node.count; ++i) {
                uint32_t pi = prims_[node.child + i];
                if (prim_intersect(group_->prims[pi], ray, rec, static_cast<int>(pi)))
                    hit = true;
            }
            continue;
        }

        // Ordered push: test both children, push farther first
        const uint32_t left  = node.child;
        const uint32_t right = left + 1;

        float tnl, tfl, tnr, tfr;
        bool hl = nodes_[left ].bbox.intersect(ray, tnl, tfl) && tnl <= rec.t;
        bool hr = nodes_[right].bbox.intersect(ray, tnr, tfr) && tnr <= rec.t;

        if (hl && hr) {
            if (tnl < tnr) {
                stack[ptr++] = right;
                stack[ptr++] = left;
            } else {
                stack[ptr++] = left;
                stack[ptr++] = right;
            }
        } else if (hl) {
            stack[ptr++] = left;
        } else if (hr) {
            stack[ptr++] = right;
        }
    }
    return hit;
}

// ─── Traversal: any-hit ────────────────────────────────────────────────────────
bool PrimBLAS::intersects(const Ray& ray) const {
    if (nodes_.empty() || !group_) return false;

    uint32_t stack[64];
    int      ptr = 0;
    stack[ptr++] = 0;

    // Reuse a scratch rec so prim_intersect can check rec.t guard
    HitRecord scratch;
    scratch.t = ray.tmax;

    while (ptr > 0) {
        const uint32_t      idx  = stack[--ptr];
        const PrimBLASNode& node = nodes_[idx];

        float tnear, tfar;
        if (!node.bbox.intersect(ray, tnear, tfar)) continue;

        if (node.is_leaf()) {
            for (uint32_t i = 0; i < node.count; ++i) {
                uint32_t pi = prims_[node.child + i];
                scratch.t = ray.tmax; // reset for each prim to avoid cross-prim masking
                if (prim_intersect(group_->prims[pi], ray, scratch, static_cast<int>(pi)))
                    return true;
            }
            continue;
        }

        stack[ptr++] = node.child;
        stack[ptr++] = node.child + 1;
    }
    return false;
}

} // namespace xn
