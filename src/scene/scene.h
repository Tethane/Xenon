#pragma once
// scene/scene.h — Container for materials, meshes, and lights

#include "geometry/mesh.h"
#include "geometry/bvh.h"
#include "material/material.h"
#include <vector>
#include <memory>

namespace xn {

struct Light {
  uint32_t mesh_id;
  uint32_t tri_idx;
  Vec3     emission;
  float    area;
  // For area light sampling: mesh, etc.
};

struct Scene {
    std::vector<TriangleMesh> meshes;
    std::vector<Material> materials;
    std::vector<BVH> bvhs;
    std::vector<Light> lights;

    // Master BVH — for now we'll just have one BVH per mesh or one for everything
    BVH world_bvh;

    void build_world_bvh() {
        // For the MVP, we assume a single triangle mesh for simplicity or 
        // merging all meshes into one.
        // Let's assume we have one main TriangleMesh for the whole scene.
    }

    // Closest object hit
    bool intersect(const Ray& ray, HitRecord& rec) const {
        return world_bvh.intersect(ray, rec);
    }

    // Visibility (shadow) ray
    bool intersects(const Ray& ray) const {
        return world_bvh.intersects(ray);
    }
    bool intersects(const Ray& ray, int& hit_prim_id) const {
        return world_bvh.intersects(ray, hit_prim_id);
    }
    
    // Pick a light to sample
    const Light& sample_light(float u_pick, float& pdf) const {
        if (lights.empty()) {
            pdf = 0.f;
            static Light null_light;
            return null_light;
        }
        int idx = (int)(u_pick * (float)lights.size());
        idx = std::clamp(idx, 0, (int)lights.size() - 1);
        pdf = 1.f / (float)lights.size();
        return lights[idx];
    }
};

} // namespace xn
