#pragma once
// scene/scene.h — Container for materials, meshes, lights, and TLAS/BLAS

#include "geometry/mesh.h"
#include "geometry/blas.h"
#include "geometry/tlas.h"
#include "material/material.h"
#include <vector>
#include <memory>

namespace xn {

struct Light {
    uint32_t mesh_id;   // index into scene.meshes
    uint32_t tri_idx;   // triangle index within that mesh
    Vec3     emission;
    float    area;
};

struct Scene {
    // ── Geometry ─────────────────────────────────────────────────────────────
    // One TriangleMesh per loaded object.  Each mesh is stored in world space
    // (transforms are baked into vertex data at load time).
    // blas_list[i] is the bottom-level BVH for meshes[i].
    // Stable pointers are required: unique_ptr avoids invalidation on push_back.
    std::vector<TriangleMesh>       meshes;
    std::vector<std::unique_ptr<BLAS>> blas_list;

    // Top-level BVH over all instances.
    TLAS tlas;

    // ── Scene data ────────────────────────────────────────────────────────────
    std::vector<Material> materials;
    std::vector<Light>    lights;

    // ── Build ─────────────────────────────────────────────────────────────────
    // Call once after all meshes have been loaded and transformed.
    // Builds one BLAS per mesh, then assembles the TLAS.
    void build_acceleration() {
        blas_list.clear();
        blas_list.reserve(meshes.size());

        std::vector<Instance> instances;
        instances.reserve(meshes.size());

        for (size_t i = 0; i < meshes.size(); ++i) {
            auto blas = std::make_unique<BLAS>();
            blas->build(meshes[i]);

            // Mesh vertices are already in world space — identity instance transform.
            Instance inst;
            inst.blas        = blas.get();
            inst.xform       = AffineTransform{};   // identity
            inst.instance_id = static_cast<int>(i);
            inst.rebuild_world_aabb();

            instances.push_back(std::move(inst));
            blas_list.push_back(std::move(blas));
        }

        tlas.build(std::move(instances));

        std::printf("[Scene] Built %zu BLAS + TLAS over %zu meshes\n",
                    blas_list.size(), meshes.size());
    }

    // ── Queries ───────────────────────────────────────────────────────────────
    // All three signatures are preserved exactly from the old scene.

    // Closest-hit — fills rec on success.
    bool intersect(const Ray& ray, HitRecord& rec) const {
        return tlas.intersect(ray, rec);
    }

    // Pure visibility test — fast any-hit, no HitRecord.
    bool intersects(const Ray& ray) const {
        return tlas.intersects(ray);
    }

    // Visibility test that also returns the hit primitive id.
    // Uses a full intersect internally (TLAS any-hit doesn't expose prim_id
    // directly, but this path is only called by code that needs the id).
    bool intersects(const Ray& ray, int& hit_prim_id) const {
        HitRecord rec;
        rec.t = ray.tmax;
        if (!tlas.intersect(ray, rec)) return false;
        hit_prim_id = rec.prim_id;
        return true;
    }

    // ── Light sampling ────────────────────────────────────────────────────────
    const Light& sample_light(float u_pick, float& pdf) const {
        if (lights.empty()) {
            pdf = 0.f;
            static const Light null_light{};
            return null_light;
        }
        int idx = std::clamp(static_cast<int>(u_pick * static_cast<float>(lights.size())),
                             0, static_cast<int>(lights.size()) - 1);
        pdf = 1.f / static_cast<float>(lights.size());
        return lights[idx];
    }
};

} // namespace xn