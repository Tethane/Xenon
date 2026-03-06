// geometry/blas.cpp — BLAS implementation with binned SAH
#include "geometry/blas.h"
#include <algorithm>
#include <iostream>

namespace xn {

// Fixed recursive builder that allocates children contiguously
void build_recursive(uint32_t node_idx, uint32_t& node_ptr, uint32_t start, uint32_t end, 
                     std::vector<BLASNode>& nodes, std::vector<uint32_t>& tri_indices, 
                     std::vector<BLAS::BuildItem>& items) {
    
    AABB node_bbox;
    for (uint32_t i = start; i < end; ++i) node_bbox.expand(items[i].bbox);
    nodes[node_idx].set_bounds(node_bbox);

    uint32_t count = end - start;
    if (count <= 2) { // Leaf threshold
        nodes[node_idx].left_child_or_offset = start;
        nodes[node_idx].tri_count = count;
        for (uint32_t i = 0; i < count; ++i) {
            tri_indices[start + i] = items[start + i].tri_idx;
        }
        return;
    }

    // Binned SAH logic
    AABB cent_bounds;
    for (uint32_t i = start; i < end; ++i) cent_bounds.expand(items[i].center);
    int axis = cent_bounds.max_extent_axis();
    float cent_extent = cent_bounds.extent()[axis];

    bool should_split = true;
    int best_bin = -1;
    constexpr int kBins = 32;
    float scale = (float)kBins / cent_extent;

    if (cent_extent == 0) {
        should_split = false;
    } else {
        struct Bin {
            AABB bounds;
            uint32_t tri_count = 0;
        } bins[kBins];

        for (uint32_t i = start; i < end; ++i) {
            int b = (int)((items[i].center[axis] - cent_bounds.mn[axis]) * scale);
            if (b >= kBins) b = kBins - 1;
            bins[b].tri_count++;
            bins[b].bounds.expand(items[i].bbox);
        }

        float left_area[kBins - 1], right_area[kBins - 1];
        uint32_t left_cnt[kBins - 1], right_cnt[kBins - 1];
        AABB left_box, right_box;
        uint32_t left_sum = 0, right_sum = 0;

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

        float parent_area = node_bbox.surface_area();
        float best_cost = (float)count; // Leaf cost
        for (int i = 0; i < kBins - 1; ++i) {
            float cost = 1.0f + (left_area[i] * left_cnt[i] + right_area[i] * right_cnt[i]) / parent_area;
            if (cost < best_cost) {
                best_cost = cost;
                best_bin = i;
            }
        }
        if (best_bin == -1) should_split = false;
    }

    if (!should_split) {
        nodes[node_idx].left_child_or_offset = start;
        nodes[node_idx].tri_count = count;
        for (uint32_t i = 0; i < count; ++i) {
            tri_indices[start + i] = items[start + i].tri_idx;
        }
        return;
    }

    // Partition
    auto it = std::partition(items.begin() + start, items.begin() + end, [&](const BLAS::BuildItem& item) {
        int b = (int)((item.center[axis] - cent_bounds.mn[axis]) * scale);
        if (b >= kBins) b = kBins - 1;
        return b <= best_bin;
    });
    uint32_t mid = (uint32_t)std::distance(items.begin(), it);
    if (mid == start || mid == end) mid = start + count / 2;

    uint32_t left_child = node_ptr;
    node_ptr += 2;
    if (node_ptr > nodes.size()) nodes.resize(node_ptr);

    nodes[node_idx].left_child_or_offset = left_child;
    nodes[node_idx].tri_count = 0;

    build_recursive(left_child, node_ptr, start, mid, nodes, tri_indices, items);
    build_recursive(left_child + 1, node_ptr, mid, end, nodes, tri_indices, items);
}

void BLAS::build(const TriangleMesh& mesh) {
    mesh_ = &mesh;
    uint32_t num_tris = mesh.num_triangles();
    if (num_tris == 0) return;

    std::vector<BuildItem> items(num_tris);
    AABB mesh_bbox;
    for (uint32_t i = 0; i < num_tris; ++i) {
        items[i].bbox = mesh.triangle_aabb(i);
        items[i].center = items[i].bbox.center();
        items[i].tri_idx = i;
        mesh_bbox.expand(items[i].bbox);
    }
    root_bounds_ = mesh_bbox;

    nodes_.clear();
    nodes_.emplace_back(); // Root
    uint32_t node_ptr = 1;
    tri_indices_.resize(num_tris);
    
    build_recursive(0, node_ptr, 0, num_tris, nodes_, tri_indices_, items);
}

bool BLAS::intersect(const Ray& ray, HitRecord& rec) const {
    if (nodes_.empty()) return false;
    
    uint32_t stack[64];
    uint32_t ptr = 0;
    stack[ptr++] = 0;

    bool hit = false;
    Vec3 inv_dir(1.f / ray.dir.x, 1.f / ray.dir.y, 1.f / ray.dir.z);
    int sign[3] = { ray.dir.x < 0, ray.dir.y < 0, ray.dir.z < 0 };

    while (ptr > 0) {
        uint32_t idx = stack[--ptr];
        const BLASNode& node = nodes_[idx];

        if (!node.get_bounds().intersect_fast(inv_dir, ray.origin, sign, ray.tmin, rec.t)) continue;

        if (node.is_leaf()) {
            for (uint32_t i = 0; i < node.tri_count; ++i) {
                if (mesh_->intersect_triangle(ray, tri_indices_[node.left_child_or_offset + i], rec)) {
                    hit = true;
                }
            }
        } else {
            uint32_t left = node.left_child_or_offset;
            uint32_t right = left + 1;
            
            // Near-child first traversal heuristic
            Vec3 extent = node.get_bounds().extent();
            if (sign[max_axis(extent)]) {
                stack[ptr++] = left;
                stack[ptr++] = right;
            } else {
                stack[ptr++] = right;
                stack[ptr++] = left;
            }
        }
    }
    return hit;
}

bool BLAS::intersect_shadow(const Ray& ray) const {
    if (nodes_.empty()) return false;
    
    uint32_t stack[64];
    uint32_t ptr = 0;
    stack[ptr++] = 0;

    Vec3 inv_dir(1.f / ray.dir.x, 1.f / ray.dir.y, 1.f / ray.dir.z);
    int sign[3] = { ray.dir.x < 0, ray.dir.y < 0, ray.dir.z < 0 };

    while (ptr > 0) {
        uint32_t idx = stack[--ptr];
        const BLASNode& node = nodes_[idx];

        if (!node.get_bounds().intersect_fast(inv_dir, ray.origin, sign, ray.tmin, ray.tmax)) continue;

        if (node.is_leaf()) {
            HitRecord dummy_rec;
            dummy_rec.t = ray.tmax;
            for (uint32_t i = 0; i < node.tri_count; ++i) {
                if (mesh_->intersect_triangle(ray, tri_indices_[node.left_child_or_offset + i], dummy_rec)) {
                    return true;
                }
            }
        } else {
            stack[ptr++] = node.left_child_or_offset;
            stack[ptr++] = node.left_child_or_offset + 1;
        }
    }
    return false;
}

} // namespace xn
