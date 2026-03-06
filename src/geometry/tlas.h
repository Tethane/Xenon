#pragma once
// geometry/tlas.h — Top-Level Acceleration Structure for mesh instances

#include "geometry/blas.h"
#include "math/mat4.h"
#include <vector>
#include <memory>

namespace xn {

struct Instance {
    uint32_t blas_id;    // Index into Scene's BLAS array
    uint32_t material_id; 
    uint32_t instance_id;
    Mat4 transform;
    Mat4 inv_transform;
    AABB world_bounds;

    Instance() = default;
    Instance(uint32_t blas_id, uint32_t mat_id, uint32_t inst_id, const Mat4& xform, const AABB& local_bounds)
        : blas_id(blas_id), material_id(mat_id), instance_id(inst_id), transform(xform) {
        inv_transform = transform.inverse();
        
        // Transform local AABB to world AABB
        world_bounds = AABB();
        Vec3 corners[8];
        local_bounds.get_corners(corners);
        for (int i = 0; i < 8; ++i) {
            world_bounds.expand(transform.transform_point(corners[i]));
        }
    }
};

struct TLASNode {
    float min[3];
    uint32_t left_child_or_instance; 
    float max[3];
    uint32_t instance_count; // 1 for leaf in TLAS normally

    bool is_leaf() const { return instance_count > 0; }
    
    void set_bounds(const AABB& bbox) {
        min[0] = bbox.mn.x; min[1] = bbox.mn.y; min[2] = bbox.mn.z;
        max[0] = bbox.mx.x; max[1] = bbox.mx.y; max[2] = bbox.mx.z;
    }

    AABB get_bounds() const {
        return AABB({min[0], min[1], min[2]}, {max[0], max[1], max[2]});
    }
};

class TLAS {
public:
    TLAS() = default;

    struct BuildItem {
        AABB bbox;
        Vec3 center;
        uint32_t inst_idx;
    };

    // Build TLAS over a set of instances
    void build(const std::vector<Instance>& instances);

    // Closest hit test (enters BLAS)
    bool intersect(const Ray& ray, const std::vector<BLAS>& blases, HitRecord& rec) const;

    // Shadow ray test
    bool intersect_shadow(const Ray& ray, const std::vector<BLAS>& blases) const;

private:
    std::vector<TLASNode> nodes_;
    std::vector<uint32_t> instance_indices_;
    const std::vector<Instance>* instances_ = nullptr;
};

} // namespace xn
