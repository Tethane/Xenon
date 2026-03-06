#pragma once
// scene/scene.h — Container for materials, meshes, and lights

#include "geometry/mesh.h"
#include "geometry/blas.h"
#include "geometry/tlas.h"
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
    std::vector<BLAS> blases;
    std::vector<Instance> instances;
    std::vector<Light> lights;

    // Master TLAS over instances
    TLAS world_tlas;

    void build_acceleration_structures() {
        blases.clear();
        blases.resize(meshes.size());
        for (size_t i = 0; i < meshes.size(); ++i) {
            blases[i].build(meshes[i]);
        }
        world_tlas.build(instances);
    }

    // Closest object hit
    bool intersect(const Ray& ray, HitRecord& rec) const {
        return world_tlas.intersect(ray, blases, rec);
    }

    // Visibility (shadow) ray
    bool intersects(const Ray& ray) const {
        return world_tlas.intersect_shadow(ray, blases);
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
