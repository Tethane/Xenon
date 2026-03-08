#pragma once
// scene/scene.h — Container for materials, geometry, lights, BVH, and environment

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "geometry/mesh.h"
#include "geometry/blas.h"
#include "geometry/prim_blas.h"
#include "geometry/tlas.h"
#include "scene/environment.h"
#include "material/material.h"

namespace xn {

// ─── Area light (triangle) ────────────────────────────────────────────────────
struct Light {
    uint32_t mesh_id;   // index into scene.meshes
    uint32_t tri_idx;   // triangle index within that mesh
    Vec3     emission;
    float    area;
};

// ─── Directional / Sun light ──────────────────────────────────────────────────
//
// Represents a distant light source at infinity (e.g. the sun).
//   direction  — unit vector pointing FROM the scene TOWARD the light source.
//                In a y-up scene, a setting sun at 30° elevation from the west
//                might be: direction = normalize({-1, tan(30°), 0}).
//   color      — spectral emission color, typically warm white for a sun.
//   intensity  — energy scale (lux or arbitrary units consistent with area lights).
//
// Shadow convention: shadow rays are cast in `direction` with tmax = kInfinity.
// There is no geometry behind a directional light, so any scene occluder will
// block it correctly.
//
// MIS: treated as a near-delta source (very high PDF). For the common case of a
// perfectly sharp sun, `pdf_sa` is set to a large sentinel (1e10f) so the
// power-2 MIS weight drives bsdf-sample contribution to near-zero, which is
// correct. See integrator for full derivation.
struct DirectionalLight {
    Vec3  direction  = {0.f, 1.f, 0.f}; // pointing toward the sun (unit vector, y-up default)
    Vec3  color      = {1.0f, 0.95f, 0.8f};
    float intensity  = 5.f;

    Vec3 emission() const noexcept { return color * intensity; }
};

// ─── Scene ────────────────────────────────────────────────────────────────────
struct Scene {
    // ── Geometry ──────────────────────────────────────────────────────────────
    // One TriangleMesh per loaded object.  Each mesh is stored in world space
    // (transforms are baked into vertex data at load time).
    // blas_list[i] is the bottom-level BVH for meshes[i].
    // Stable pointers are required: unique_ptr avoids invalidation on push_back.
    std::vector<TriangleMesh>        meshes;
    std::vector<std::unique_ptr<BLAS>> blas_list;

    // All analytic primitives (spheres, boxes, disks, quads, planes).
    // One PrimBLAS is built over this group, then added as an Instance to the TLAS.
    PrimGroup                        all_prims;
    std::unique_ptr<PrimBLAS>        prim_blas;

    // Top-level BVH over all instances.
    TLAS tlas;

    // ── Scene data ────────────────────────────────────────────────────────────
    std::vector<Material>        materials;
    std::vector<Light>           lights;
    std::vector<DirectionalLight> dir_lights;
    Environment                  sky;               // environment / sky model

    // ── Build ─────────────────────────────────────────────────────────────────
    // Call once after all meshes and primitives have been added.
    // Builds one BLAS per mesh + a PrimBLAS (if any primitives exist),
    // then assembles the TLAS.
    void build_acceleration() {
        blas_list.clear();
        blas_list.reserve(meshes.size());

        std::vector<Instance> instances;
        instances.reserve(meshes.size() + 1); // +1 for optional prim_blas

        int id = 0;

        for (size_t i = 0; i < meshes.size(); ++i) {
            auto blas = std::make_unique<BLAS>();
            blas->build(meshes[i]);

            Instance inst;
            inst.geom        = GeomHandle::from_mesh(blas.get());
            inst.xform       = AffineTransform{};   // identity (world-space meshes)
            inst.instance_id = id++;
            inst.rebuild_world_aabb();

            instances.push_back(std::move(inst));
            blas_list.push_back(std::move(blas));
        }

        // Add PrimBLAS instance if analytic primitives are present
        if (!all_prims.empty()) {
            prim_blas = std::make_unique<PrimBLAS>();
            prim_blas->build(all_prims);

            Instance inst;
            inst.geom        = GeomHandle::from_prim(prim_blas.get());
            inst.xform       = AffineTransform{};   // identity
            inst.instance_id = id++;
            inst.rebuild_world_aabb();

            instances.push_back(std::move(inst));
        }

        tlas.build(std::move(instances));

        std::printf("[Scene] Built %zu mesh BLAS + %s prim BLAS + TLAS\n",
                    blas_list.size(),
                    prim_blas ? "1" : "0");
    }

    // ── Queries ───────────────────────────────────────────────────────────────
    bool intersect(const Ray& ray, HitRecord& rec) const {
        return tlas.intersect(ray, rec);
    }
    bool intersects(const Ray& ray) const {
        return tlas.intersects(ray);
    }
    bool intersects(const Ray& ray, int& hit_prim_id) const {
        HitRecord rec;
        rec.t = ray.tmax;
        if (!tlas.intersect(ray, rec)) return false;
        hit_prim_id = rec.prim_id;
        return true;
    }

    // ── Area light sampling ───────────────────────────────────────────────────
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

    // ── Directional light sampling ────────────────────────────────────────────
    // Returns the index of the sampled directional light (uniform selection).
    // pdf is the selection probability (1 / N).
    int sample_dir_light(float u_pick, float& pdf) const {
        if (dir_lights.empty()) { pdf = 0.f; return -1; }
        int idx = std::clamp(static_cast<int>(u_pick * static_cast<float>(dir_lights.size())),
                             0, static_cast<int>(dir_lights.size()) - 1);
        pdf = 1.f / static_cast<float>(dir_lights.size());
        return idx;
    }
};

} // namespace xn