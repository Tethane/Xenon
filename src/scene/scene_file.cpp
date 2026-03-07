#include "scene/scene_file.h"
#include "geometry/mesh.h"
#include "geometry/primitives.h"
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
    if (fs::exists(rel)) return rel;
    fs::path scene_dir = fs::path(scene_path).parent_path();
    fs::path resolved  = scene_dir / rel;
    if (fs::exists(resolved)) return resolved.string();
    return rel;
}

// ─────────────────────────────────────────────────────────────────────────────
// Strip surrounding quotes: "foo" → foo
// ─────────────────────────────────────────────────────────────────────────────
static std::string strip_quotes(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: read an int material-id from a name token (returns -1 on failure)
// ─────────────────────────────────────────────────────────────────────────────
static int resolve_mat(const std::string& name, const std::map<std::string, int>& mat_map) {
    auto it = mat_map.find(name);
    if (it == mat_map.end()) {
        std::fprintf(stderr, "[Warning] Unknown material '%s', using default\n", name.c_str());
        return 0;
    }
    return it->second;
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

        // ── config ────────────────────────────────────────────────────────────
        if (cmd == "config") {
            ss >> config.width >> config.height >> config.samples;

        // ── camera ────────────────────────────────────────────────────────────
        } else if (cmd == "camera") {
            ss >> eye.x    >> eye.y    >> eye.z
               >> target.x >> target.y >> target.z
               >> fov;

        // ── matfile (external .mat file) ─────────────────────────────────────
        } else if (cmd == "matfile") {
            std::string mat_path;
            ss >> mat_path;
            mat_path = strip_quotes(mat_path);
            mat_path = "materials/" + mat_path;

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
            ss >> spec >> ior >> trans >> trans_rough;

            Material m = material_from_legacy(name, albedo, metallic, roughness,
                                              spec, ior, trans, trans_rough);
            mat_map[name] = static_cast<int>(scene.materials.size());
            scene.materials.push_back(std::move(m));

        // ── mesh ──────────────────────────────────────────────────────────────
        } else if (cmd == "mesh") {
            std::string obj_path;
            ss >> obj_path;

            TriangleMesh m;
            if (!load_obj(obj_path, m, mat_map)) {
                std::fprintf(stderr, "[Error] Failed to load mesh: %s\n", obj_path.c_str());
                continue;
            }

            Vec3  pos(0.f), rot_deg(0.f);
            float scale = 1.f;
            if (ss >> pos.x >> pos.y >> pos.z >> scale >> rot_deg.x >> rot_deg.y >> rot_deg.z) {
                m.transform(pos, scale, rot_deg);
            }

            scene.meshes.push_back(std::move(m));

        // ── light (area light — triangle emitter) ─────────────────────────────
        } else if (cmd == "light") {
            Light lt{};
            ss >> lt.mesh_id >> lt.tri_idx
               >> lt.emission.x >> lt.emission.y >> lt.emission.z
               >> lt.area;
            scene.lights.push_back(lt);

        // ── sunlight (directional light) ──────────────────────────────────────
        // Syntax: sunlight <dx> <dy> <dz> <r> <g> <b> <intensity>
        //   (dx, dy, dz) is the direction FROM the scene TOWARD the sun.
        //   Normalized automatically at scene build time.
        } else if (cmd == "sunlight") {
            DirectionalLight dl;
            ss >> dl.direction.x >> dl.direction.y >> dl.direction.z
               >> dl.color.x     >> dl.color.y     >> dl.color.z
               >> dl.intensity;

            // Normalize direction (robust to imprecise scene file values)
            float len = dl.direction.length();
            if (len > 1e-6f) dl.direction = dl.direction / len;
            else               dl.direction = Vec3(0.f, 1.f, 0.f); // fallback: zenith

            scene.dir_lights.push_back(dl);
            std::printf("[Scene] Added directional light: dir=(%.2f,%.2f,%.2f) intensity=%.1f\n",
                        dl.direction.x, dl.direction.y, dl.direction.z, dl.intensity);

        // ── sky (environment / gradient sky) ─────────────────────────────────
        // Syntax: sky <z_r> <z_g> <z_b>  <h_r> <h_g> <h_b>  [<g_r> <g_g> <g_b>]  [intensity] [sharpness]
        } else if (cmd == "sky") {
            Vec3 zenith, horizon;
            ss >> zenith.x  >> zenith.y  >> zenith.z
               >> horizon.x >> horizon.y >> horizon.z;
            scene.sky.zenith_color  = zenith;
            scene.sky.horizon_color = horizon;

            // Optional ground color
            Vec3 ground(0.1f, 0.08f, 0.05f);
            float intensity = 1.f, sharpness = 3.f;
            if (ss >> ground.x >> ground.y >> ground.z) {
                scene.sky.ground_color = ground;
                ss >> intensity >> sharpness;
                scene.sky.intensity          = intensity;
                scene.sky.horizon_sharpness  = sharpness;
            }

            std::printf("[Scene] Sky set: zenith=(%.2f,%.2f,%.2f) horizon=(%.2f,%.2f,%.2f)\n",
                        zenith.x, zenith.y, zenith.z, horizon.x, horizon.y, horizon.z);

        // ── sphere ────────────────────────────────────────────────────────────
        // Syntax: sphere <cx> <cy> <cz> <radius> <mat_name>
        } else if (cmd == "sphere") {
            Sphere s;
            std::string mat_name;
            ss >> s.center.x >> s.center.y >> s.center.z >> s.radius >> mat_name;
            s.mat_id = resolve_mat(mat_name, mat_map);
            scene.all_prims.add(s);

        // ── box ───────────────────────────────────────────────────────────────
        // Syntax: box <cx> <cy> <cz> <hx> <hy> <hz> <mat_name>
        } else if (cmd == "box") {
            Box b;
            std::string mat_name;
            ss >> b.center.x >> b.center.y >> b.center.z
               >> b.half.x   >> b.half.y   >> b.half.z
               >> mat_name;
            b.mat_id = resolve_mat(mat_name, mat_map);
            scene.all_prims.add(b);

        // ── disk ──────────────────────────────────────────────────────────────
        // Syntax: disk <cx> <cy> <cz> <nx> <ny> <nz> <radius> <mat_name>
        } else if (cmd == "disk") {
            Disk d;
            std::string mat_name;
            ss >> d.center.x >> d.center.y >> d.center.z
               >> d.normal.x >> d.normal.y >> d.normal.z
               >> d.radius >> mat_name;
            float nlen = d.normal.length();
            if (nlen > 1e-6f) d.normal = d.normal / nlen;
            d.mat_id = resolve_mat(mat_name, mat_map);
            scene.all_prims.add(d);

        // ── quad ──────────────────────────────────────────────────────────────
        // Syntax: quad <ox> <oy> <oz> <ux> <uy> <uz> <vx> <vy> <vz> <mat_name>
        } else if (cmd == "quad") {
            Quad q;
            std::string mat_name;
            ss >> q.origin.x >> q.origin.y >> q.origin.z
               >> q.u_vec.x  >> q.u_vec.y  >> q.u_vec.z
               >> q.v_vec.x  >> q.v_vec.y  >> q.v_vec.z
               >> mat_name;
            q.mat_id = resolve_mat(mat_name, mat_map);
            scene.all_prims.add(q);

        // ── plane ─────────────────────────────────────────────────────────────
        // Syntax: plane <nx> <ny> <nz> <offset> <mat_name>
        //   where offset = dot(normal, any_point_on_plane)
        } else if (cmd == "plane") {
            Plane p;
            std::string mat_name;
            ss >> p.normal.x >> p.normal.y >> p.normal.z >> p.offset >> mat_name;
            float nlen = p.normal.length();
            if (nlen > 1e-6f) p.normal = p.normal / nlen;
            p.mat_id = resolve_mat(mat_name, mat_map);
            scene.all_prims.add(p);
        }
    }

    camera.look_at(eye, target, up, fov,
                   static_cast<float>(config.width) / static_cast<float>(config.height));

    // Build one BLAS per mesh + optional PrimBLAS, then assemble the TLAS.
    scene.build_acceleration();

    return true;
}

} // namespace xn