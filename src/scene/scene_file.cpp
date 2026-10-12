// scene/scene_file.cpp — .xenon scene file parser

#include <cstdio>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

#include "scene/scene_file.h"
#include "geometry/mesh.h"
#include "geometry/primitives.h"
#include "material/material.h"

namespace xn {

// ─────────────────────────────────────────────────────────────────────────────
// Resolve a relative path against the scene file's directory
// ─────────────────────────────────────────────────────────────────────────────
static std::string resolve_relative(const std::string &scene_path, const std::string &rel) {
	namespace fs = std::filesystem;

	return (fs::path(scene_path).parent_path() / rel).lexically_normal().string();
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: read an int material-id from a name token
// ─────────────────────────────────────────────────────────────────────────────
static int resolve_mat(const std::string &name, const std::map<std::string, int> &mat_map) {
	auto it = mat_map.find(name);

	if (it == mat_map.end()) {
		throw std::runtime_error("Unknown material '" + name + "'");
	}

	return it->second;
}

static bool load_scene_impl(const std::string &path, Scene &scene, Camera &camera,
							SceneConfig &config, int &line_num) {
	auto load_start = std::chrono::steady_clock::now();
	std::ifstream f(path);

	if (!f.is_open()) {
		std::fprintf(stderr, "Failed to open scene: %s\n", path.c_str());

		return false;
	}

	std::string line;
	std::map<std::string, int> mat_map;

	// Default camera params
	Vec3 eye(0, 5, 20), target(0, 5, 0), up(0, 1, 0);
	float fov = 40.f, aperture = 0.f, focus = 1.f;
	std::string camera_medium;

	while (std::getline(f, line)) {
		++line_num;
		std::stringstream ss(line);
		std::string cmd;
		ss >> cmd;

		if (cmd.empty() || cmd[0] == '#')
			continue;
		auto require_fields = [&]() {
			if (ss.fail())
				throw std::runtime_error("Invalid fields for " + cmd);
		};
		auto has_more = [&]() {
			if (ss.eof())
				return false;
			ss >> std::ws;

			return !ss.eof() && ss.peek() != '#';
		};

		// ── config ────────────────────────────────────────────────────────────
		if (cmd == "version") {
			ss >> config.version;

			if (config.version != 2)
				throw std::runtime_error("Unsupported scene version (supported: 2)");
		} else if (cmd == "config") {
			ss >> config.width >> config.height >> config.samples;

			// ── camera ────────────────────────────────────────────────────────────
		} else if (cmd == "camera") {
			ss >> eye.x >> eye.y >> eye.z >> target.x >> target.y >> target.z >> fov;

		} else if (cmd == "camera_up") {
			ss >> up.x >> up.y >> up.z;
		} else if (cmd == "camera_medium") {
			ss >> std::quoted(camera_medium);
		} else if (cmd == "lens") {
			ss >> aperture >> focus;

			// ── matfile (external .mat file) ─────────────────────────────────────
		} else if (cmd == "matfile") {
			std::string mat_path;
			ss >> std::quoted(mat_path);
			require_fields();

			if (config.version < 2)
				mat_path = "materials/" + mat_path;

			std::string resolved = resolve_relative(path, mat_path);
			Material m = load_material(resolved);
			mat_map[m.name] = static_cast<int>(scene.materials.size());
			scene.materials.push_back(std::move(m));

			// ── material (legacy inline) ──────────────────────────────────────────
		} else if (cmd == "material") {
			std::string name;
			Vec3 albedo;
			float metallic, roughness;
			ss >> name >> albedo.x >> albedo.y >> albedo.z >> metallic >> roughness;
			require_fields();

			float spec = 0.5f, ior = 1.5f, trans = 0.0f, trans_rough = -1.0f;

			for (float *value : {&spec, &ior, &trans, &trans_rough}) {
				if (!has_more())
					break;
				ss >> *value;
				require_fields();
			}

			Material m = material_from_legacy(name, albedo, metallic, roughness, spec, ior, trans,
											  trans_rough);
			mat_map[name] = static_cast<int>(scene.materials.size());
			scene.materials.push_back(std::move(m));

			// ── mesh ──────────────────────────────────────────────────────────────
		} else if (cmd == "instances_explicit") {
			if (config.version < 2)
				throw std::runtime_error("Explicit instances require scene version 2");
			scene.explicit_instances = true;
		} else if (cmd == "instance") {
			if (config.version < 2)
				throw std::runtime_error("Explicit instances require scene version 2");
			size_t mesh_id;
			Mat4x3 matrix{};
			ss >> mesh_id;

			for (int row = 0; row < 3; ++row)
				for (int col = 0; col < 4; ++col)
					ss >> matrix.cols[col][row];
			require_fields();
			scene.explicit_instances = true;
			scene.mesh_placements.push_back({mesh_id, AffineTransform::from_matrix(matrix)});
		} else if (cmd == "mesh") {
			std::string obj_path;
			ss >> std::quoted(obj_path);
			require_fields();
			obj_path = resolve_relative(path, obj_path);

			TriangleMesh m;

			if (!load_obj(obj_path, m, mat_map)) {
				std::fprintf(stderr, "[Error] Failed to load mesh: %s\n", obj_path.c_str());
				throw std::runtime_error("Failed to load mesh: " + obj_path);
			}

			Vec3 pos(0.f), rot_deg(0.f);
			float scale = 1.f;

			if (has_more()) {
				ss >> pos.x >> pos.y >> pos.z >> scale >> rot_deg.x >> rot_deg.y >> rot_deg.z;
				require_fields();

				if (scale == 0.f)
					throw std::runtime_error("Mesh scale must be nonzero");
				m.transform(pos, scale, rot_deg);
			}

			scene.meshes.push_back(std::move(m));

			// ── light (area light — triangle emitter) ─────────────────────────────
		} else if (cmd == "light") {
			Light lt{};
			lt.material_emission = false;
			ss >> lt.mesh_id >> lt.tri_idx >> lt.emission.x >> lt.emission.y >> lt.emission.z >>
				lt.area;

			if (lt.emission.x < 0.f || lt.emission.y < 0.f || lt.emission.z < 0.f)
				throw std::runtime_error("Light emission must be nonnegative");
			scene.lights.push_back(lt);

			// ── sunlight (directional light) ──────────────────────────────────────
			// Syntax: sunlight <dx> <dy> <dz> <r> <g> <b> <intensity>
			//   (dx, dy, dz) is the direction FROM the scene TOWARD the sun.
			//   Normalized automatically at scene build time.
		} else if (cmd == "pointlight" || cmd == "spotlight") {
			PointLight light;
			ss >> light.position.x >> light.position.y >> light.position.z;

			if (cmd == "spotlight")
				ss >> light.direction.x >> light.direction.y >> light.direction.z;
			ss >> light.intensity.x >> light.intensity.y >> light.intensity.z;
			require_fields();

			for (int i = 0; i < 3; ++i)
				if (!std::isfinite(light.position[i]) || !std::isfinite(light.intensity[i]) ||
					light.intensity[i] < 0.f)
					throw std::runtime_error("Invalid point/spot position or intensity");

			if (cmd == "spotlight") {
				float inner, outer;
				ss >> inner >> outer;
				require_fields();

				if (!std::isfinite(inner) || !std::isfinite(outer) || inner < 0.f ||
					outer < inner || outer >= 90.f || !std::isfinite(light.direction.length_sq()) ||
					light.direction.length_sq() == 0.f)
					throw std::runtime_error("Invalid spotlight axis or half-cone angles");
				light.direction = normalize(light.direction);
				light.cos_inner = std::cos(inner * kPi / 180.f);
				light.cos_outer = std::cos(outer * kPi / 180.f);
			}
			scene.point_lights.push_back(light);
		} else if (cmd == "sunlight") {
			DirectionalLight dl;
			ss >> dl.direction.x >> dl.direction.y >> dl.direction.z >> dl.color.x >> dl.color.y >>
				dl.color.z >> dl.intensity;

			// Normalize direction (robust to imprecise scene file values)
			require_fields();
			float len = dl.direction.length();

			if (!(len > 0.f) || dl.intensity < 0.f || dl.color.x < 0.f || dl.color.y < 0.f ||
				dl.color.z < 0.f)
				throw std::runtime_error("Invalid sunlight direction or radiance");
			dl.direction = dl.direction / len;

			scene.dir_lights.push_back(dl);
			std::printf("[Scene] Added directional light: dir=(%.2f,%.2f,%.2f) intensity=%.1f\n",
						dl.direction.x, dl.direction.y, dl.direction.z, dl.intensity);

			// ── sky (environment / gradient sky) ─────────────────────────────────
			// Syntax: sky <z_r> <z_g> <z_b>  <h_r> <h_g> <h_b>  [<g_r> <g_g> <g_b>]  [intensity]
			// [sharpness]
		} else if (cmd == "environment") {
			std::string texture_path;
			float intensity, rotation;
			ss >> std::quoted(texture_path) >> intensity >> rotation;
			require_fields();

			if (!std::isfinite(intensity) || intensity < 0 || !std::isfinite(rotation))
				throw std::runtime_error("Invalid environment intensity or rotation");
			scene.sky.set_image(load_texture(resolve_relative(path, texture_path), false),
								rotation * kPi / 180.f);
			scene.sky.intensity = intensity;
		} else if (cmd == "sky") {
			scene.sky.image.reset();
			scene.sky.cdf.clear();
			Vec3 zenith, horizon;
			ss >> zenith.x >> zenith.y >> zenith.z >> horizon.x >> horizon.y >> horizon.z;
			require_fields();
			scene.sky.zenith_color = zenith;
			scene.sky.horizon_color = horizon;

			// Optional ground color
			Vec3 ground(0.1f, 0.08f, 0.05f);
			float intensity = 1.f, sharpness = 3.f;

			if (has_more()) {
				ss >> ground.x >> ground.y >> ground.z;
				require_fields();
				scene.sky.ground_color = ground;

				if (has_more()) {
					ss >> intensity;
					require_fields();
				}

				if (has_more()) {
					ss >> sharpness;
					require_fields();
				}
				scene.sky.intensity = intensity;
				scene.sky.horizon_sharpness = sharpness;
			}

			for (float c : {zenith.x, zenith.y, zenith.z, horizon.x, horizon.y, horizon.z, ground.x,
							ground.y, ground.z, intensity})
				if (c < 0.f)
					throw std::runtime_error("Sky radiance must be nonnegative");

			if (!(sharpness > 0.f))
				throw std::runtime_error("Sky sharpness must be positive");
			std::printf("[Scene] Sky set: zenith=(%.2f,%.2f,%.2f) horizon=(%.2f,%.2f,%.2f)\n",
						zenith.x, zenith.y, zenith.z, horizon.x, horizon.y, horizon.z);

			// ── sphere ────────────────────────────────────────────────────────────
			// Syntax: sphere <cx> <cy> <cz> <radius> <mat_name>
		} else if (cmd == "volume_box") {
			HomogeneousMedium m;
			ss >> m.minimum.x >> m.minimum.y >> m.minimum.z >> m.maximum.x >> m.maximum.y >>
				m.maximum.z >> m.sigma_a.x >> m.sigma_a.y >> m.sigma_a.z >> m.sigma_s.x >>
				m.sigma_s.y >> m.sigma_s.z >> m.g;
			require_fields();

			for (int i = 0; i < 3; ++i) {
				if (!std::isfinite(m.minimum[i]) || !std::isfinite(m.maximum[i]) ||
					!(m.maximum[i] > m.minimum[i]) || !std::isfinite(m.sigma_a[i]) ||
					m.sigma_a[i] < 0.f || !std::isfinite(m.sigma_s[i]) || m.sigma_s[i] < 0.f)
					throw std::runtime_error("Invalid volume box bounds or coefficients");
			}

			if (!std::isfinite(m.g) || std::abs(m.g) >= .999f)
				throw std::runtime_error("Volume anisotropy must be in (-.999,.999)");

			for (const auto &other : scene.media) {
				bool overlaps = true;

				for (int i = 0; i < 3; ++i)
					overlaps &= m.minimum[i] < other.maximum[i] && m.maximum[i] > other.minimum[i];

				if (overlaps)
					throw std::runtime_error("Overlapping volume interiors are unsupported");
			}
			scene.media.push_back(m);
		} else if (cmd == "sphere") {
			Sphere s;
			std::string mat_name;
			ss >> s.center.x >> s.center.y >> s.center.z >> s.radius >> mat_name;
			require_fields();

			if (!(s.radius > 0.f))
				throw std::runtime_error("Sphere radius must be positive");
			s.mat_id = resolve_mat(mat_name, mat_map);
			scene.all_prims.add(s);

			// ── box ───────────────────────────────────────────────────────────────
			// Syntax: box <cx> <cy> <cz> <hx> <hy> <hz> <mat_name>
		} else if (cmd == "box") {
			Box b;
			std::string mat_name;
			ss >> b.center.x >> b.center.y >> b.center.z >> b.half.x >> b.half.y >> b.half.z >>
				mat_name;
			require_fields();

			if (!(b.half.x > 0.f && b.half.y > 0.f && b.half.z > 0.f))
				throw std::runtime_error("Box extents must be positive");
			b.mat_id = resolve_mat(mat_name, mat_map);
			scene.all_prims.add(b);

			// ── disk ──────────────────────────────────────────────────────────────
			// Syntax: disk <cx> <cy> <cz> <nx> <ny> <nz> <radius> <mat_name>
		} else if (cmd == "disk") {
			Disk d;
			std::string mat_name;
			ss >> d.center.x >> d.center.y >> d.center.z >> d.normal.x >> d.normal.y >>
				d.normal.z >> d.radius >> mat_name;
			require_fields();
			float nlen = d.normal.length();

			if (!(nlen > 0.f && d.radius > 0.f))
				throw std::runtime_error("Invalid disk normal or radius");
			d.normal = d.normal / nlen;
			d.mat_id = resolve_mat(mat_name, mat_map);
			scene.all_prims.add(d);

			// ── quad ──────────────────────────────────────────────────────────────
			// Syntax: quad <ox> <oy> <oz> <ux> <uy> <uz> <vx> <vy> <vz> <mat_name>
		} else if (cmd == "quad") {
			Quad q;
			std::string mat_name;
			ss >> q.origin.x >> q.origin.y >> q.origin.z >> q.u_vec.x >> q.u_vec.y >> q.u_vec.z >>
				q.v_vec.x >> q.v_vec.y >> q.v_vec.z >> mat_name;
			require_fields();

			if (cross(q.u_vec, q.v_vec).length_sq() == 0.f)
				throw std::runtime_error("Degenerate quad");
			q.mat_id = resolve_mat(mat_name, mat_map);
			scene.all_prims.add(q);

			// ── plane ─────────────────────────────────────────────────────────────
			// Syntax: plane <nx> <ny> <nz> <offset> <mat_name>
			//   where offset = dot(normal, any_point_on_plane)
		} else if (cmd == "plane") {
			Plane p;
			std::string mat_name;
			ss >> p.normal.x >> p.normal.y >> p.normal.z >> p.offset >> mat_name;
			require_fields();
			float nlen = p.normal.length();

			if (!(nlen > 0.f))
				throw std::runtime_error("Invalid plane normal");
			p.normal = p.normal / nlen;
			p.offset /= nlen;
			p.mat_id = resolve_mat(mat_name, mat_map);
			scene.all_prims.add(p);
		} else {
			throw std::runtime_error("Unknown scene command '" + cmd + "'");
		}
		require_fields();

		if (has_more())
			throw std::runtime_error("Unexpected fields for " + cmd);
	}

	if (f.bad())
		throw std::runtime_error("Failed to read scene");

	if (config.width <= 0 || config.height <= 0 || config.samples <= 0 ||
		static_cast<int64_t>(config.width) * config.height > std::numeric_limits<int>::max() / 3)
		throw std::runtime_error("Invalid image dimensions or sample count");

	if (fov <= 0.f || fov >= 180.f || (target - eye).length_sq() < 1e-12f ||
		cross(target - eye, up).length_sq() < 1e-12f)
		throw std::runtime_error("Invalid camera");

	for (const auto &mesh : scene.meshes) {
		for (int tri = 0; tri < mesh.num_triangles(); ++tri) {
			int mat_id = mesh.mat_ids[tri];

			if (mat_id < 0 || static_cast<size_t>(mat_id) >= scene.materials.size())
				throw std::runtime_error("Mesh references an invalid material");

			if (scene.materials[mat_id].has_textures() && !mesh.triangle_has_uv(tri))
				throw std::runtime_error("Textured mesh triangle lacks UV coordinates: " +
										 std::to_string(tri));
		}
	}

	for (const auto &prim : scene.all_prims.prims)
		std::visit(
			[&](const auto &p) {
				if (scene.materials[p.mat_id].has_textures())
					throw std::runtime_error(
						"Image textures on analytic primitives are unsupported; export a UV mesh");
			},
			prim);

	for (const auto &light : scene.lights) {
		if (light.mesh_id >= scene.meshes.size() ||
			light.tri_idx >= static_cast<uint32_t>(scene.meshes[light.mesh_id].num_triangles()) ||
			!(light.area > 0.f))
			throw std::runtime_error("Invalid light reference or area (mesh " +
									 std::to_string(light.mesh_id) + ", triangle " +
									 std::to_string(light.tri_idx) + ")");
	}

	camera.look_at(eye, target, up, fov,
				   static_cast<float>(config.width) / static_cast<float>(config.height));

	// Build one BLAS per mesh + optional PrimBLAS, then assemble the TLAS.
	camera.set_lens(aperture, focus);

	if (!camera_medium.empty()) {
		int id = resolve_mat(camera_medium, mat_map);

		if (scene.materials[id].transmission <= 0)
			throw std::runtime_error("Camera medium must name a transmissive material");
		camera.set_medium(id, scene.materials[id].absorption);
	}
	auto build_start = std::chrono::steady_clock::now();
	config.load_seconds = std::chrono::duration<double>(build_start - load_start).count();
	scene.build_acceleration();
	config.acceleration_seconds =
		std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start).count();

	return true;
}

bool load_scene(const std::string &path, Scene &scene, Camera &camera, SceneConfig &config) {
	int line_num = 0;

	try {
		return load_scene_impl(path, scene, camera, config, line_num);
	} catch (const std::exception &e) {
		std::fprintf(stderr, "[Scene] %s:%d: %s\n", path.c_str(), line_num, e.what());

		return false;
	}
}

} // namespace xn
