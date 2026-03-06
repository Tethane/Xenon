// geometry/tlas.cpp — TLAS implementation
#include "geometry/tlas.h"
#include <algorithm>

namespace xn {

// Recursive builder helper for TLAS
void build_tlas_recursive(uint32_t node_idx, uint32_t& node_ptr, uint32_t start, uint32_t end, 
                          std::vector<TLASNode>& nodes, std::vector<uint32_t>& inst_indices, 
                          std::vector<TLAS::BuildItem>& items) {
    AABB node_bbox;
    for (uint32_t i = start; i < end; ++i) node_bbox.expand(items[i].bbox);
    nodes[node_idx].set_bounds(node_bbox);

    uint32_t count = end - start;
    if (count <= 1) { // Reach instances
        nodes[node_idx].left_child_or_instance = start;
        nodes[node_idx].instance_count = count;
        for (uint32_t i = 0; i < count; ++i) {
            inst_indices[start + i] = items[start + i].inst_idx;
        }
        return;
    }

    AABB cent_bounds;
    for (uint32_t i = start; i < end; ++i) cent_bounds.expand(items[i].center);
    int axis = cent_bounds.max_extent_axis();
    float cent_extent = cent_bounds.extent()[axis];

    // Simple median split for TLAS as instance count is usually small
    // But let's use SAH if count > 4
    bool use_sah = count > 4 && cent_extent > 0;
    int best_bin = -1;
    constexpr int kBins = 16;
    float scale = (float)kBins / cent_extent;

    if (use_sah) {
        struct Bin { AABB bounds; uint32_t count = 0; } bins[kBins];
        for (uint32_t i = start; i < end; ++i) {
            int b = (int)((items[i].center[axis] - cent_bounds.mn[axis]) * scale);
            if (b >= kBins) b = kBins - 1;
            bins[b].count++;
            bins[b].bounds.expand(items[i].bbox);
        }
        float left_area[kBins-1], right_area[kBins-1];
        uint32_t left_cnt[kBins-1], right_cnt[kBins-1];
        AABB lb, rb; uint32_t ls=0, rs=0;
        for (int i=0; i<kBins-1; ++i) {
            ls += bins[i].count; left_cnt[i] = ls;
            lb.expand(bins[i].bounds); left_area[i] = lb.surface_area();
        }
        for (int i=kBins-1; i>0; --i) {
            rs += bins[i].count; right_cnt[i-1] = rs;
            rb.expand(bins[i].bounds); right_area[i-1] = rb.surface_area();
        }
        float parent_area = node_bbox.surface_area();
        float best_cost = (float)count;
        for (int i=0; i<kBins-1; ++i) {
            float cost = 1.0f + (left_area[i]*left_cnt[i] + right_area[i]*right_cnt[i])/parent_area;
            if (cost < best_cost) { best_cost = cost; best_bin = i; }
        }
    }

    uint32_t mid;
    if (best_bin != -1) {
        auto it = std::partition(items.begin() + start, items.begin() + end, [&](const TLAS::BuildItem& item) {
            int b = (int)((item.center[axis] - cent_bounds.mn[axis]) * scale);
            if (b >= kBins) b = kBins - 1;
            return b <= best_bin;
        });
        mid = (uint32_t)std::distance(items.begin(), it);
    } else {
        mid = start + count / 2;
        std::nth_element(items.begin() + start, items.begin() + mid, items.begin() + end, [&](const TLAS::BuildItem& a, const TLAS::BuildItem& b) {
            return a.center[axis] < b.center[axis];
        });
    }

    uint32_t left = node_ptr;
    node_ptr += 2;
    nodes.resize(std::max((size_t)node_ptr, nodes.size()));
    nodes[node_idx].left_child_or_instance = left;
    nodes[node_idx].instance_count = 0;

    build_tlas_recursive(left, node_ptr, start, mid, nodes, inst_indices, items);
    build_tlas_recursive(left + 1, node_ptr, mid, end, nodes, inst_indices, items);
}

void TLAS::build(const std::vector<Instance>& instances) {
    instances_ = &instances;
    uint32_t num = (uint32_t)instances.size();
    if (num == 0) return;

    std::vector<BuildItem> items(num);
    for (uint32_t i = 0; i < num; ++i) {
        items[i].bbox = instances[i].world_bounds;
        items[i].center = items[i].bbox.center();
        items[i].inst_idx = i;
    }

    nodes_.clear();
    nodes_.emplace_back();
    uint32_t node_ptr = 1;
    instance_indices_.resize(num);
    build_tlas_recursive(0, node_ptr, 0, num, nodes_, instance_indices_, items);
}

bool TLAS::intersect(const Ray& ray, const std::vector<BLAS>& blases, HitRecord& rec) const {
    if (nodes_.empty()) return false;
    
    uint32_t stack[64];
    uint32_t ptr = 0;
    stack[ptr++] = 0;

    bool hit = false;
    Vec3 inv_dir(1.f / ray.dir.x, 1.f / ray.dir.y, 1.f / ray.dir.z);
    int sign[3] = { ray.dir.x < 0, ray.dir.y < 0, ray.dir.z < 0 };

    while (ptr > 0) {
        uint32_t idx = stack[--ptr];
        const TLASNode& node = nodes_[idx];

        if (!node.get_bounds().intersect_fast(inv_dir, ray.origin, sign, ray.tmin, rec.t)) continue;

        if (node.is_leaf()) {
            for (uint32_t i = 0; i < node.instance_count; ++i) {
                const Instance& inst = (*instances_)[instance_indices_[node.left_child_or_instance + i]];
                
                // Transform ray to object space
                Ray local_ray;
                local_ray.origin = inst.inv_transform.transform_point(ray.origin);
                local_ray.dir = inst.inv_transform.transform_dir(ray.dir);
                local_ray.tmin = ray.tmin;
                local_ray.tmax = rec.t;

                HitRecord local_rec = rec;
                if (blases[inst.blas_id].intersect(local_ray, local_rec)) {
                    hit = true;
                    // Copy results back and transform normals
                    float old_t = rec.t;
                    rec = local_rec;
                    rec.t = local_rec.t; // Already in ray parametric space if transform is affine
                    // Note: if scaling is non-uniform, normals need careful handling.
                    // Assuming uniform scaling or affine for now.
                    rec.pos = inst.transform.transform_point(local_rec.pos);
                    rec.normal = normalize(inst.transform.transform_dir(local_rec.normal));
                    rec.geo_normal = normalize(inst.transform.transform_dir(local_rec.geo_normal));
                    if (inst.material_id != (uint32_t)-1) {
                        rec.mat_id = inst.material_id;
                    } else {
                        rec.mat_id = local_rec.mat_id;
                    }
                    rec.instance_id = inst.instance_id;
                }
            }
        } else {
            uint32_t left = node.left_child_or_instance;
            stack[ptr++] = left;
            stack[ptr++] = left + 1;
        }
    }
    return hit;
}

bool TLAS::intersect_shadow(const Ray& ray, const std::vector<BLAS>& blases) const {
    if (nodes_.empty()) return false;
    
    uint32_t stack[64];
    uint32_t ptr = 0;
    stack[ptr++] = 0;

    Vec3 inv_dir(1.f / ray.dir.x, 1.f / ray.dir.y, 1.f / ray.dir.z);
    int sign[3] = { ray.dir.x < 0, ray.dir.y < 0, ray.dir.z < 0 };

    while (ptr > 0) {
        uint32_t idx = stack[--ptr];
        const TLASNode& node = nodes_[idx];

        if (!node.get_bounds().intersect_fast(inv_dir, ray.origin, sign, ray.tmin, ray.tmax)) continue;

        if (node.is_leaf()) {
            for (uint32_t i = 0; i < node.instance_count; ++i) {
                const Instance& inst = (*instances_)[instance_indices_[node.left_child_or_instance + i]];
                Ray local_ray;
                local_ray.origin = inst.inv_transform.transform_point(ray.origin);
                local_ray.dir = inst.inv_transform.transform_dir(ray.dir);
                local_ray.tmin = ray.tmin;
                local_ray.tmax = ray.tmax;

                if (blases[inst.blas_id].intersect_shadow(local_ray)) return true;
            }
        } else {
            stack[ptr++] = node.left_child_or_instance;
            stack[ptr++] = node.left_child_or_instance + 1;
        }
    }
    return false;
}

} // namespace xn
