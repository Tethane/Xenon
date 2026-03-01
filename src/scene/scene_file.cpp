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
            int tri_idx;
            Vec3 emission;
            float area;
            ss >> tri_idx >> emission.x >> emission.y >> emission.z >> area;
            scene.lights.push_back({(uint32_t)tri_idx, emission, area});
        }
    }

    camera.look_at(eye, target, up, fov, (float)config.width / config.height);
    
    if (!scene.meshes.empty()) {
        scene.world_bvh.build(scene.meshes[0]);
    }

    return true;
}

} // namespace xn
