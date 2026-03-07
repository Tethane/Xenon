#include "scene/scene_file.h"
#include "geometry/mesh.h"
#include "material/material.h"
#include <fstream>
#include <sstream>
#include <cstdio>
#include <map>
#include <filesystem>

namespace xn {

// ─────────────────────────────────────────────────────────────────────────────
// Resolve a relative path against the scene file's directory
// ─────────────────────────────────────────────────────────────────────────────
static std::string resolve_relative(const std::string& scene_path, const std::string& rel) {
    namespace fs = std::filesystem;
    // 1. Try as-is (relative to CWD)
    if (fs::exists(rel)) return rel;
    // 2. Try relative to scene file
    fs::path scene_dir = fs::path(scene_path).parent_path();
    fs::path resolved  = scene_dir / rel;
    if (fs::exists(resolved)) return resolved.string();
    // Fallback to original behavior if both fail (though it will likely fail later)
    return rel;
}

// ─────────────────────────────────────────────────────────────────────────────
// Strip surrounding quotes from a token: "foo" → foo
// ─────────────────────────────────────────────────────────────────────────────
static std::string strip_quotes(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}

bool load_scene(const std::string& path, Scene& scene, Camera& camera, SceneConfig& config) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "Failed to open scene: %s\n", path.c_str());
        return false;
    }

    std::string line;
    std::map<std::string, int> mat_map;

    // Default camera params
    Vec3  eye(0, 5, 20), target(0, 5, 0), up(0, 1, 0);
    float fov = 40.f;

    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string cmd;
        ss >> cmd;
        if (cmd.empty() || cmd[0] == '#') continue;

        // ── config ───────────────────────────────────────────────────────────
        if (cmd == "config") {
            ss >> config.width >> config.height >> config.samples;

        // ── camera ───────────────────────────────────────────────────────────
        } else if (cmd == "camera") {
            ss >> eye.x    >> eye.y    >> eye.z
               >> target.x >> target.y >> target.z
               >> fov;

        // ── matfile (external .mat file) ──────────────────────────────────────
        } else if (cmd == "matfile") {
            std::string mat_path;
            ss >> mat_path;
            mat_path = strip_quotes(mat_path);

            std::string resolved = resolve_relative(path, mat_path);
            Material m = load_material(resolved);

            mat_map[m.name] = static_cast<int>(scene.materials.size());
            scene.materials.push_back(std::move(m));

        // ── material (legacy inline) ──────────────────────────────────────────
        } else if (cmd == "material") {
            std::string name;
            Vec3  albedo;
            float metallic, roughness;
            ss >> name >> albedo.x >> albedo.y >> albedo.z >> metallic >> roughness;

            float spec = 0.5f, ior = 1.5f, trans = 0.0f, trans_rough = -1.0f;
            if (ss >> spec >> ior >> trans >> trans_rough) {

            };

            Material m = material_from_legacy(name, albedo, metallic, roughness,
                                              spec, ior, trans, trans_rough);
            mat_map[name] = static_cast<int>(scene.materials.size());
            scene.materials.push_back(std::move(m));

        // ── mesh ──────────────────────────────────────────────────────────────
        // Each mesh command produces one entry in scene.meshes.
        // We no longer merge meshes together — each gets its own BLAS so:
        //   • large scenes stay cache-friendly (one BVH per object, not one giant BVH)
        //   • Light::mesh_id correctly identifies the source mesh for light sampling
        } else if (cmd == "mesh") {
            std::string obj_path;
            ss >> obj_path;

            // Resolve OBJ path relative to the scene file directory so that
            // scenes are portable (previously this path was used verbatim).
            // obj_path = resolve_relative(path, strip_quotes(obj_path));

            TriangleMesh m;
            if (!load_obj(obj_path, m, mat_map)) {
                std::fprintf(stderr, "[Error] Failed to load mesh: %s\n", obj_path.c_str());
                continue;
            }

            // Optional transform: px py pz scale rx ry rz
            Vec3  pos(0.f), rot_deg(0.f);
            float scale = 1.f;
            if (ss >> pos.x >> pos.y >> pos.z >> scale >> rot_deg.x >> rot_deg.y >> rot_deg.z) {
                m.transform(pos, scale, rot_deg);
            }

            scene.meshes.push_back(std::move(m));

        // ── light ─────────────────────────────────────────────────────────────
        // mesh_id now refers to the index into scene.meshes (1:1 with source objects).
        } else if (cmd == "light") {
            Light lt{};
            ss >> lt.mesh_id >> lt.tri_idx
               >> lt.emission.x >> lt.emission.y >> lt.emission.z
               >> lt.area;
            scene.lights.push_back(lt);
        }
    }

    camera.look_at(eye, target, up, fov,
                   static_cast<float>(config.width) / static_cast<float>(config.height));

    // Build one BLAS per mesh, then assemble the TLAS.
    scene.build_acceleration();

    return true;
}

} // namespace xn