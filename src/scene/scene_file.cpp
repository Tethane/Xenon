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

  // Mesh cache for sharing geometry within this scene load
  std::map<std::string, uint32_t> mesh_cache;
  int current_mat_id = -1;

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

        current_mat_id = (int)scene.materials.size();
        mat_map[m.name] = current_mat_id;
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
        current_mat_id = (int)scene.materials.size();
        mat_map[name] = current_mat_id;
        scene.materials.push_back(std::move(m));
    } else if (cmd == "mesh") {
        std::string obj_path;
        ss >> obj_path;
        
        // Resolve absolute path to use as key for mesh sharing
        std::string resolved_path = resolve_relative(path, obj_path);
        
        uint32_t mesh_id;
        
        if (mesh_cache.find(resolved_path) == mesh_cache.end()) {
            TriangleMesh m;
            if (load_obj(resolved_path, m, mat_map, current_mat_id)) {
                mesh_id = (uint32_t)scene.meshes.size();
                scene.meshes.push_back(std::move(m));
                mesh_cache[resolved_path] = mesh_id;
            } else {
                std::fprintf(stderr, "[Error] Failed to load mesh: %s\n", obj_path.c_str());
                continue;
            }
        } else {
            mesh_id = mesh_cache[resolved_path];
        }

        Vec3 p(0, 0, 0), r(0, 0, 0);
        float s = 1.0f;
        Mat4 xform = Mat4::identity();
        
        if (ss >> p.x >> p.y >> p.z >> s >> r.x >> r.y >> r.z) {
            // Transform: Translation * Rotation * Scale
            // Note: TriangleMesh::transform used Scale -> Rotate -> Translate
            // Let's match that order for consistency or use a proper mat4 composition.
            xform = Mat4::translate(p) * 
                    Mat4::rotate({0,0,1}, r.z * M_PI / 180.f) *
                    Mat4::rotate({0,1,0}, r.y * M_PI / 180.f) *
                    Mat4::rotate({1,0,0}, r.x * M_PI / 180.f) *
                    Mat4::scale({s, s, s});
        }
        
        uint32_t inst_id = (uint32_t)scene.instances.size();
        // Use -1 for material override to signify "use mesh/obj materials"
        scene.instances.emplace_back(mesh_id, (uint32_t)-1, inst_id, xform, scene.meshes[mesh_id].compute_aabb());

    } else if (cmd == "light") {
        uint32_t mesh_id;
        uint32_t tri_idx;
        Vec3 emission;
        float area;
        ss >> mesh_id >> tri_idx >> emission.x >> emission.y >> emission.z >> area;
        scene.lights.push_back({mesh_id, tri_idx, emission, area});
    }
  }

  camera.look_at(eye, target, up, fov, (float)config.width / config.height);
  
  // Build TLAS/BLAS
  scene.build_acceleration_structures();

  return true;
}

} // namespace xn
