#pragma once
// geometry/blas.h — CPU-optimized Bottom-Level Acceleration Structure (BLAS)

#include "geometry/aabb.h"
#include "geometry/mesh.h"
#include <vector>
#include <cstdint>

namespace xn {

// ─────────────────────────────────────────────────────────────────────────────
// BLASNode — 32 bytes (packed to fit in half a cache line)
// Layout:
// [min.x, min.y, min.z, left_child/tri_offset]  (16 bytes)
// [max.x, max.y, max.z, tri_count]             (16 bytes)
// ─────────────────────────────────────────────────────────────────────────────
struct alignas(32) BLASNode {
    float min[3];
    uint32_t left_child_or_offset; 
    float max[3];
    uint32_t tri_count; // tri_count > 0 means leaf

    bool is_leaf() const { return tri_count > 0; }
    
    void set_bounds(const AABB& bbox) {
        min[0] = bbox.mn.x; min[1] = bbox.mn.y; min[2] = bbox.mn.z;
        max[0] = bbox.mx.x; max[1] = bbox.mx.y; max[2] = bbox.mx.z;
    }

    AABB get_bounds() const {
        return AABB({min[0], min[1], min[2]}, {max[0], max[1], max[2]});
    }
};

class BLAS {
public:
    BLAS() = default;

    struct BuildItem {
        AABB bbox;
        Vec3 center;
        uint32_t tri_idx;
    };

    // Build BLAS over a mesh using binned SAH
    void build(const TriangleMesh& mesh);

    // Closest hit test
    bool intersect(const Ray& ray, HitRecord& rec) const;

    // Shadow ray test (any-hit)
    bool intersect_shadow(const Ray& ray) const;

    // Diagnostics
    void print_stats() const;

    const AABB& bounds() const { return nodes_.empty() ? empty_bounds_ : root_bounds_; }

private:
    const TriangleMesh* mesh_ = nullptr;
    std::vector<BLASNode> nodes_;
    std::vector<uint32_t> tri_indices_;
    AABB root_bounds_;
    static inline AABB empty_bounds_;

};

} // namespace xn
