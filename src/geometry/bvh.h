#pragma once
// geometry/bvh.h — Cache-efficient BVH with SAH construction

#include "geometry/aabb.h"
#include "geometry/mesh.h"
#include <vector>
#include <memory>

namespace xn {

// ─────────────────────────────────────────────────────────────────────────────
// BVHNode — 32 bytes (packed to fit in half a cache line)
// ─────────────────────────────────────────────────────────────────────────────
struct alignas(32) BVHNode {
    AABB bbox;
    union {
        uint32_t left_child;    // if > 0, index to left child (right is left+1)
        uint32_t tri_offset;   // if leaf, index into redirected triangle array
    };
    uint32_t tri_count;         // 0 if internal node

    bool is_leaf() const { return tri_count > 0; }
};

// ─────────────────────────────────────────────────────────────────────────────
// BVH — Acceleration structure for TriangleMesh
// ─────────────────────────────────────────────────────────────────────────────
class BVH {
public:
    BVH() = default;
    
    // Build BVH over a mesh using SAH
    void build(const TriangleMesh& mesh);

    // Closest hit test
    bool intersect(const Ray& ray, HitRecord& rec) const;

    // Visibility test (early out)
    bool intersects(const Ray& ray) const;
    bool intersects(const Ray& ray, int& hit_prim_id) const;

    // Packet traversal (4 rays)
    void intersect4(const RayPacket4& rp, HitRecord recs[4], int active_mask) const;

private:
    const TriangleMesh* mesh_ = nullptr;
    std::vector<BVHNode> nodes_;
    std::vector<uint32_t> tri_indices_; // indirection to mesh triangles

    struct BuildItem {
        AABB bbox;
        Vec3 center;
        uint32_t tri_idx;
    };

    uint32_t subdivide(uint32_t node_idx, uint32_t start, uint32_t end, std::vector<BuildItem>& items);
    void update_node_bounds(uint32_t node_idx, uint32_t start, uint32_t end, const std::vector<BuildItem>& items);
};

} // namespace xn
