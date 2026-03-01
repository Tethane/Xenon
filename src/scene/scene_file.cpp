#include "scene/scene_file.h"
#include "geometry/mesh.h"
#include <fstream>
#include <sstream>
#include <cstdio>
#include <map>

namespace xn {

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
        if (cmd == "config") {
            ss >> config.width >> config.height >> config.samples;
        } else if (cmd == "camera") {
            ss >> eye.x >> eye.y >> eye.z >> target.x >> target.y >> target.z >> fov;
        } else if (cmd == "material") {
            std::string name;
            PrincipledBSDF m;
            ss >> name >> m.albedo.x >> m.albedo.y >> m.albedo.z >> m.metallic >> m.roughness;
            mat_map[name] = (int)scene.materials.size();
            scene.materials.push_back(m);
        } else if (cmd == "mesh") {
            std::string obj_path, mat_name;
            ss >> obj_path >> mat_name;
            TriangleMesh m;
            if (load_obj(obj_path, m, mat_map[mat_name])) {
                scene.meshes.push_back(std::move(m));
            }
        } else if (cmd == "light") {
            int tri_idx;
            Vec3 emission;
            float area;
            ss >> tri_idx >> emission.x >> emission.y >> emission.z >> area;
            scene.lights.push_back({(uint32_t)tri_idx, emission, area});
        }
    }

    camera.look_at(eye, target, up, fov, (float)config.width / config.height);
    
    // For MVP: merge all meshes into one big mesh for the BVH
    if (!scene.meshes.empty()) {
        // Merge strategy or single mesh for world_bvh
        scene.world_bvh.build(scene.meshes[0]);
    }

    return true;
}

} // namespace xn
