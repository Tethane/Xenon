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
    fs::path scene_dir = fs::path(scene_path).parent_path();
    fs::path resolved  = scene_dir / rel;
    return resolved.string();
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

  // Simplified parser: one keyword per line
  std::string line;
  std::map<std::string, int> mat_map;
  
  // Default Camera Params
  Vec3 eye(0, 5, 20), target(0, 5, 0), up(0, 1, 0);
  float fov = 40.f;

  while (std::getline(f, line)) {
    std::stringstream ss(line);
    std::string cmd;
    ss >> cmd;

    if (cmd.empty() || cmd[0] == '#')
        continue;

    if (cmd == "config") {
        ss >> config.width >> config.height >> config.samples;
    } else if (cmd == "camera") {
        ss >> eye.x >> eye.y >> eye.z >> target.x >> target.y >> target.z >> fov;
    } else if (cmd == "matfile") {
        // ── New .mat file reference ──────────────────────────────────────
        std::string mat_path;
        ss >> mat_path;
        mat_path = strip_quotes(mat_path);

        // Resolve relative to scene directory
        std::string resolved = resolve_relative(path, mat_path);
        Material m = load_material(resolved);

        mat_map[m.name] = (int)scene.materials.size();
        scene.materials.push_back(std::move(m));
    } else if (cmd == "material") {
        // ── Legacy inline material (backward compat) ─────────────────────
        std::string name;
        Vec3 albedo;
        float metallic, roughness;
        ss >> name >> albedo.x >> albedo.y >> albedo.z >> metallic >> roughness;

        float spec = 0.5f, ior = 1.5f, trans = 0.0f, trans_rough = -1.0f;
        if (ss >> spec >> ior >> trans >> trans_rough) {
            // All extended fields present
        }

        Material m = material_from_legacy(name, albedo, metallic, roughness,
                                          spec, ior, trans, trans_rough);
        mat_map[name] = (int)scene.materials.size();
        scene.materials.push_back(std::move(m));
    } else if (cmd == "mesh") {
        std::string obj_path;
        ss >> obj_path;
        TriangleMesh m;
        if (load_obj(obj_path, m, mat_map)) {
            Vec3 p(0, 0, 0), r(0, 0, 0);
            float s = 1.0f;
            // Parse optional transform: px py pz scale rx ry rz
            if (ss >> p.x >> p.y >> p.z >> s >> r.x >> r.y >> r.z) {
                m.transform(p, s, r);
            }
            
            if (scene.meshes.empty()) {
                scene.meshes.push_back(std::move(m));
            } else {
                scene.meshes[0].merge(m);
            }
        } else {
            std::fprintf(stderr, "[Error] Failed to load mesh: %s\n", obj_path.c_str());
        }
    } else if (cmd == "light") {
        uint32_t mesh_id;
        uint32_t tri_idx;
        Vec3 emission;
        float area; // Perceived surface area of emission
        ss >> mesh_id >> tri_idx >> emission.x >> emission.y >> emission.z >> area;
        scene.lights.push_back({mesh_id, tri_idx, emission, area});
    }
  }

  camera.look_at(eye, target, up, fov, (float)config.width / config.height);
  
  // Build BVH
  if (!scene.meshes.empty()) {
      scene.world_bvh.build(scene.meshes[0]);
  }

  return true;
}

} // namespace xn
