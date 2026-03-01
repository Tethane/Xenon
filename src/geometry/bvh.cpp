// geometry/bvh.cpp — BVH build (SAH) and traversal implementation

#include "geometry/bvh.h"
#include <algorithm>
#include <stack>

namespace xn {

void BVH::build(const TriangleMesh& mesh) {
    mesh_ = &mesh;
    uint32_t num_tris = mesh.num_triangles();
    if (num_tris == 0) return;

    std::vector<BuildItem> items(num_tris);
    for (uint32_t i = 0; i < num_tris; ++i) {
        items[i].bbox = mesh.triangle_aabb(i);
        items[i].center = items[i].bbox.center();
        items[i].tri_idx = i;
    }

    nodes_.clear();
    nodes_.reserve(num_tris * 2);
    nodes_.emplace_back(); // Root

    tri_indices_.resize(num_tris);
    subdivide(0, 0, num_tris, items);
}

uint32_t BVH::subdivide(uint32_t node_idx, uint32_t start, uint32_t end, std::vector<BuildItem>& items) {
    update_node_bounds(node_idx, start, end, items);

    uint32_t count = end - start;
    if (count <= 2) { // Leaf threshold
        nodes_[node_idx].tri_offset = start;
        nodes_[node_idx].tri_count = count;
        for (uint32_t i = 0; i < count; ++i) {
            tri_indices_[start + i] = items[start + i].tri_idx;
        }
        return node_idx;
    }

    // SAH Partitioning
    int best_axis = -1;
    float best_cost = kInfinity;

    AABB cent_bounds;
    for (uint32_t i = start; i < end; ++i) cent_bounds.expand(items[i].center);
    int axis = cent_bounds.max_extent_axis();

    // Binning for SAH
    constexpr int kBins = 12;
    struct Bin {
        AABB bounds;
        int tri_count = 0;
    } bins[kBins];

    float scale = (float)kBins / cent_bounds.extent()[axis];
    if (cent_bounds.extent()[axis] == 0) scale = 0;

    for (uint32_t i = start; i < end; ++i) {
        int b = (int)((items[i].center[axis] - cent_bounds.mn[axis]) * scale);
        if (b >= kBins) b = kBins - 1;
        bins[b].tri_count++;
        bins[b].bounds.expand(items[i].bbox);
    }

    float left_area[kBins - 1], right_area[kBins - 1];
    int left_cnt[kBins - 1], right_cnt[kBins - 1];
    AABB left_box, right_box;
    int left_sum = 0, right_sum = 0;

    for (int i = 0; i < kBins - 1; ++i) {
        left_sum += bins[i].tri_count;
        left_cnt[i] = left_sum;
        left_box.expand(bins[i].bounds);
        left_area[i] = left_box.surface_area();
    }
    for (int i = kBins - 1; i > 0; --i) {
        right_sum += bins[i].tri_count;
        right_cnt[i - 1] = right_sum;
        right_box.expand(bins[i].bounds);
        right_area[i - 1] = right_box.surface_area();
    }

    float parent_area = nodes_[node_idx].bbox.surface_area();
    best_axis = axis;
    int best_bin = -1;
    for (int i = 0; i < kBins - 1; ++i) {
        float cost = (left_area[i] * left_cnt[i] + right_area[i] * right_cnt[i]) / parent_area;
        if (cost < best_cost) {
            best_cost = cost;
            best_bin = i;
        }
    }

    // If leaf is cheaper, stop
    if (best_cost >= (float)count) {
        nodes_[node_idx].tri_offset = start;
        nodes_[node_idx].tri_count = count;
        for (uint32_t i = 0; i < count; ++i) {
            tri_indices_[start + i] = items[start + i].tri_idx;
        }
        return node_idx;
    }

    // Partition
    auto it = std::partition(items.begin() + start, items.begin() + end, [&](const BuildItem& item) {
        int b = (int)((item.center[axis] - cent_bounds.mn[axis]) * scale);
        if (b >= kBins) b = kBins - 1;
        return b <= best_bin;
    });
    uint32_t mid = (uint32_t)std::distance(items.begin(), it);

    if (mid == start || mid == end) { // Fallback if SAH fails to split
        mid = start + count / 2;
    }

    uint32_t left_idx = (uint32_t)nodes_.size();
    nodes_.emplace_back();
    nodes_.emplace_back();
    nodes_[node_idx].left_child = left_idx;
    nodes_[node_idx].tri_count = 0;

    subdivide(left_idx, start, mid, items);
    subdivide(left_idx + 1, mid, end, items);

    return node_idx;
}

void BVH::update_node_bounds(uint32_t node_idx, uint32_t start, uint32_t end, const std::vector<BuildItem>& items) {
    AABB b;
    for (uint32_t i = start; i < end; ++i) b.expand(items[i].bbox);
    nodes_[node_idx].bbox = b;
}

bool BVH::intersect(const Ray& ray, HitRecord& rec) const {
    if (nodes_.empty()) return false;
    
    uint32_t stack[64];
    uint32_t ptr = 0;
    stack[ptr++] = 0;

    bool hit = false;
    float inv_dx = 1.f / ray.dir.x, inv_dy = 1.f / ray.dir.y, inv_dz = 1.f / ray.dir.z;
    Vec3 inv_dir(inv_dx, inv_dy, inv_dz);
    int sign[3] = { ray.dir.x < 0, ray.dir.y < 0, ray.dir.z < 0 };

    while (ptr > 0) {
        uint32_t idx = stack[--ptr];
        const BVHNode& node = nodes_[idx];

        if (!node.bbox.intersect_fast(inv_dir, ray.origin, sign, ray.tmin, rec.t)) continue;

        if (node.is_leaf()) {
            for (uint32_t i = 0; i < node.tri_count; ++i) {
                if (mesh_->intersect_triangle(ray, tri_indices_[node.tri_offset + i], rec)) {
                    hit = true;
                }
            }
        } else {
            // Near-far traversal heuristic
            if (sign[max_axis(node.bbox.extent())]) {
                stack[ptr++] = node.left_child;
                stack[ptr++] = node.left_child + 1;
            } else {
                stack[ptr++] = node.left_child + 1;
                stack[ptr++] = node.left_child;
            }
        }
    }
    return hit;
}

bool BVH::intersects(const Ray& ray) const {
    if (nodes_.empty()) return false;
    uint32_t stack[64];
    uint32_t ptr = 0;
    stack[ptr++] = 0;

    Vec3 inv_dir(1.f / ray.dir.x, 1.f / ray.dir.y, 1.f / ray.dir.z);
    int sign[3] = { ray.dir.x < 0, ray.dir.y < 0, ray.dir.z < 0 };

    while (ptr > 0) {
        uint32_t idx = stack[--ptr];
        const BVHNode& node = nodes_[idx];
        if (!node.bbox.intersect_fast(inv_dir, ray.origin, sign, ray.tmin, ray.tmax)) continue;
        if (node.is_leaf()) {
            HitRecord dummy; dummy.t = ray.tmax;
            for (uint32_t i = 0; i < node.tri_count; ++i) {
                if (mesh_->intersect_triangle(ray, tri_indices_[node.tri_offset + i], dummy)) return true;
            }
        } else {
            stack[ptr++] = node.left_child;
            stack[ptr++] = node.left_child + 1;
        }
    }
    return false;
}

// SIMD 4-ray packet traversal - placeholder for now, will implement if time permits or strictly needed for MVP
// The user asked for SIMD but also "Wavefront Rendering" which often uses single rays gathered into packets.
// I'll start with scalar for correctness and add SIMD packet traversal in Phase 7.
void BVH::intersect4(const RayPacket4& rp, HitRecord recs[4], int active_mask) const {
    (void)rp; (void)recs; (void)active_mask;
}

} // namespace xn
