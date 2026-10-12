#pragma once
// scene/scene.h — Container for materials, geometry, lights, BVH, and environment

#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

#include "geometry/mesh.h"
#include "geometry/blas.h"
#include "geometry/prim_blas.h"
#include "geometry/tlas.h"
#include "scene/environment.h"
#include "volume/medium.h"
#include "material/material.h"

namespace xn {

// ─── Area light (triangle) ────────────────────────────────────────────────────
struct Light {
	uint32_t mesh_id; // index into scene.meshes
	uint32_t tri_idx; // triangle index within that mesh
	Vec3 emission;
	float area;
	int instance_id = -1;
	Vec3 p0{}, p1{}, p2{}, normal{};
	bool material_emission = false;
	bool emissionTwoSided = false;
	float selection_pmf = 0.f;
	double selection_cdf = 0.;
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
// A delta-direction source: selection has a discrete PMF, and its NEE MIS weight is one.
struct DirectionalLight {
	Vec3 direction = {0.f, 1.f, 0.f}; // pointing toward the sun (unit vector, y-up default)
	Vec3 color = {1.0f, 0.95f, 0.8f};
	float intensity = 5.f;

	Vec3 emission() const noexcept {
		return color * intensity;
	}
};

// Intensity is radiant intensity (W/sr), and spot direction points outward.
struct PointLight {
	Vec3 position{}, intensity{1.f}, direction{0.f, 0.f, -1.f};
	float cos_inner = -1.f, cos_outer = -1.f; // cos_outer<0 denotes an omnidirectional point

	float falloff(Vec3 toward_surface) const {
		if (cos_outer < 0.f)
			return 1.f;
		float c = dot(direction, toward_surface);

		if (c <= cos_outer)
			return 0.f;

		if (c >= cos_inner)
			return 1.f;
		float t = (c - cos_outer) / (cos_inner - cos_outer);

		return t * t * (3.f - 2.f * t);
	}
};

// ─── Scene ────────────────────────────────────────────────────────────────────
struct MeshPlacement {
	size_t mesh_id = 0;
	AffineTransform transform{};
};

struct Scene {
	// ── Geometry ──────────────────────────────────────────────────────────────
	// One TriangleMesh per loaded object.  Each mesh is stored in world space
	// (transforms are baked into vertex data at load time).
	// blas_list[i] is the bottom-level BVH for meshes[i].
	// Stable pointers are required: unique_ptr avoids invalidation on push_back.
	std::vector<TriangleMesh> meshes;
	// Explicit placements suppress automatic identity instances. Mesh assets
	// remain shared between placements on every backend.
	bool explicit_instances = false;
	std::vector<MeshPlacement> mesh_placements;
	std::vector<std::unique_ptr<BLAS>> blas_list;

	// All analytic primitives (spheres, boxes, disks, quads, planes).
	// One PrimBLAS is built over this group, then added as an Instance to the TLAS.
	PrimGroup all_prims;
	std::unique_ptr<PrimBLAS> prim_blas;

	// Top-level BVH over all instances.
	TLAS tlas;

	// ── Scene data ────────────────────────────────────────────────────────────
	std::vector<Material> materials;
	std::vector<HomogeneousMedium> media;
	bool has_alpha_masks = false;
	bool has_alpha_blend = false;
	std::vector<Light> lights;
	std::vector<DirectionalLight> dir_lights;
	std::vector<PointLight> point_lights;
	Environment sky; // environment / sky model

	// ── Build ─────────────────────────────────────────────────────────────────
	// Call once after all meshes and primitives have been added.
	// Builds one BLAS per mesh + a PrimBLAS (if any primitives exist),
	// then assembles the TLAS.
	void build_acceleration() {
		has_alpha_masks = has_alpha_blend = false;

		for (const auto &m : materials) {
			has_alpha_blend |= m.opacityMode == Material::OpacityMode::Blend &&
							   (m.opacity < 1.f || bool(m.opacityTexture));
			has_alpha_masks |= m.opacityMode == Material::OpacityMode::Mask &&
							   (bool(m.opacityTexture) || m.opacity < m.alphaCutoff);
		}
		prim_blas.reset();
		blas_list.clear();
		blas_list.reserve(meshes.size());

		std::vector<Instance> instances;
		instances.reserve(meshes.size() + 1); // +1 for optional prim_blas

		int id = 0;

		for (size_t i = 0; i < meshes.size(); ++i) {
			auto blas = std::make_unique<BLAS>();
			blas->build(meshes[i]);

			blas_list.push_back(std::move(blas));
		}
		auto add_mesh_instance = [&](size_t mesh_id, const AffineTransform &transform) {
			if (mesh_id >= meshes.size())
				throw std::runtime_error("Instance references nonexistent mesh");
			Instance inst;
			inst.geom = GeomHandle::from_mesh(blas_list[mesh_id].get());
			inst.xform = transform;
			inst.instance_id = id++;
			inst.rebuild_world_aabb();
			instances.push_back(std::move(inst));
		};

		if (explicit_instances) {
			for (const auto &placement : mesh_placements)
				add_mesh_instance(placement.mesh_id, placement.transform);
		} else {
			for (size_t i = 0; i < meshes.size(); ++i)
				add_mesh_instance(i, AffineTransform{});
		}

		// Add PrimBLAS instance if analytic primitives are present
		if (!all_prims.empty()) {
			prim_blas = std::make_unique<PrimBLAS>();
			prim_blas->build(all_prims);

			Instance inst;
			inst.geom = GeomHandle::from_prim(prim_blas.get());
			inst.xform = AffineTransform{}; // identity
			inst.instance_id = id++;
			inst.rebuild_world_aabb();

			instances.push_back(std::move(inst));
		}

		tlas.build(std::move(instances));
		prepare_lights();

		std::printf("[Scene] Built %zu mesh BLAS + %s prim BLAS + TLAS\n", blas_list.size(),
					prim_blas ? "1" : "0");
	}

	void prepare_lights() {
		auto definitions = lights;
		std::unordered_map<uint64_t, const Light *> overrides;
		overrides.reserve(definitions.size());

		for (const auto &def : definitions)
			overrides.try_emplace((uint64_t(def.mesh_id) << 32) | def.tri_idx, &def);
		lights.clear();

		for (const auto &inst : tlas.instances()) {
			if (inst.geom.type != GeomHandle::MESH_BLAS)
				continue;

			for (size_t m = 0; m < meshes.size(); ++m) {
				if (blas_list[m].get() != inst.geom.mesh)
					continue;
				const auto &mesh = meshes[m];

				for (int t = 0; t < mesh.num_triangles(); ++t) {
					Vec3 emission = materials[mesh.mat_ids[t]].emission;
					bool from_material = true;

					if (auto found = overrides.find((uint64_t(m) << 32) | uint32_t(t));
						found != overrides.end()) {
						emission = found->second->emission;
						from_material = found->second->material_emission;
					}

					if (max_component(emission) <= 0.f)
						continue;
					Light light{(uint32_t)m, (uint32_t)t, emission, 0.f};
					light.instance_id = inst.instance_id;
					light.material_emission = from_material;
					light.emissionTwoSided = materials[mesh.mat_ids[t]].emissionTwoSided;
					mesh.get_triangle(t, light.p0, light.p1, light.p2);
					light.p0 = inst.xform.local_to_world.transform_point(light.p0);
					light.p1 = inst.xform.local_to_world.transform_point(light.p1);
					light.p2 = inst.xform.local_to_world.transform_point(light.p2);
					light.area = 0.5f * cross(light.p1 - light.p0, light.p2 - light.p0).length();
					light.normal = inst.xform.normal_to_world(mesh.geo_normal(t));

					if (light.area > 0.f)
						lights.push_back(light);
				}
			}
		}
		prepare_light_distribution();
	}

	// Radiant-power proxy, mixed with uniform selection to retain support for
	// textured/masked emission poorly represented by the constant multiplier.
	void prepare_light_distribution() {
		std::sort(lights.begin(), lights.end(), [](const Light &a, const Light &b) {
			return a.instance_id < b.instance_id ||
				   (a.instance_id == b.instance_id && a.tri_idx < b.tri_idx);
		});
		auto power = [](const Light &l) {
			return (l.emissionTwoSided ? 2. : 1.) * double(l.area) *
				   (.2126 * double(l.emission.x) + .7152 * double(l.emission.y) +
					.0722 * double(l.emission.z));
		};
		double total = 0.;

		for (const auto &l : lights)
			total += power(l);
		double cdf = 0.;

		for (auto &l : lights) {
			double p = total > 0. ? .95 * power(l) / total + .05 / double(lights.size())
								  : 1. / double(lights.size());
			l.selection_pmf = float(p);
			cdf += p;
			l.selection_cdf = cdf;
		}

		if (!lights.empty())
			lights.back().selection_cdf = 1.;
	}

	const Light *light_for_hit(const HitRecord &hit) const {
		auto it = std::lower_bound(
			lights.begin(), lights.end(), hit, [](const Light &light, const HitRecord &h) {
				return light.instance_id < h.instance_id ||
					   (light.instance_id == h.instance_id && light.tri_idx < uint32_t(h.prim_id));
			});

		return it != lights.end() && it->instance_id == hit.instance_id &&
					   it->tri_idx == uint32_t(hit.prim_id)
				   ? &*it
				   : nullptr;
	}

	Vec3 emission(const HitRecord &hit) const {
		if (const auto *light = light_for_hit(hit)) {
			if (!hit.front_face && !light->emissionTwoSided)
				return Vec3(0.f);

			return light->material_emission
					   ? light->emission *
							 materials[hit.mat_id]
								 .emissionTexture
								 .sample(hit.tex_u, hit.tex_v, materials[hit.mat_id].textureLod)
								 .rgb()
					   : light->emission;
		}
		const auto &mat = materials[hit.mat_id];

		return hit.front_face || mat.emissionTwoSided ? mapped_emission(mat, hit.tex_u, hit.tex_v)
													  : Vec3(0.f);
	}

	Vec3 light_emission(const Light &light, float u, float v) const {
		const auto &mesh = meshes[light.mesh_id];
		const auto &mat = materials[mesh.mat_ids[light.tri_idx]];
		HitRecord hit;
		mesh.surface_uv(light.tri_idx, u, v, hit);
		float coverage = mat.opacityMode == Material::OpacityMode::Blend
							 ? surface_opacity(mat, hit)
							 : (surface_opaque(mat, hit) ? 1.f : 0.f);

		return coverage *
			   (light.material_emission
					? light.emission *
						  mat.emissionTexture.sample(hit.tex_u, hit.tex_v, mat.textureLod).rgb()
					: light.emission);
	}

	Material evaluate(HitRecord &hit) const {
		return evaluate_material(materials[hit.mat_id], hit);
	}

	float light_pdf(const HitRecord &hit, Vec3 from, Vec3 direction) const {
		const auto *light = light_for_hit(hit);

		if (!light)
			return 0.f;
		float cosine = dot(light->normal, -direction);

		if (light->emissionTwoSided)
			cosine = std::abs(cosine);

		return cosine > 0.f
				   ? light->selection_pmf * (hit.pos - from).length_sq() / (cosine * light->area)
				   : 0.f;
	}

	// ── Queries ───────────────────────────────────────────────────────────────
	HitFilter opacity_filter() const {
		return {this, [](const void *context, const HitRecord &hit, uint32_t seed) {
					const auto &scene = *static_cast<const Scene *>(context);

					return surface_opaque(scene.materials[hit.mat_id], hit, seed);
				}};
	}

	bool intersect(const Ray &ray, HitRecord &rec) const {
		if (!has_alpha_masks && !has_alpha_blend)
			return tlas.intersect(ray, rec);
		auto filter = opacity_filter();

		return tlas.intersect(ray, rec, &filter);
	}

	bool intersects(const Ray &ray) const {
		if (!has_alpha_masks && !has_alpha_blend)
			return tlas.intersects(ray);
		auto filter = opacity_filter();

		return tlas.intersects(ray, &filter);
	}

	bool intersects(const Ray &ray, int &hit_prim_id) const {
		HitRecord rec;
		rec.t = ray.tmax;

		if (!intersect(ray, rec))
			return false;
		hit_prim_id = rec.prim_id;

		return true;
	}

	// ── Area light sampling ───────────────────────────────────────────────────
	const Light &sample_light(double u_pick, float &pdf) const {
		if (lights.empty()) {
			pdf = 0.f;
			static const Light null_light{};

			return null_light;
		}
		int lo = 0, hi = int(lights.size()) - 1;

		while (lo < hi) {
			int mid = lo + (hi - lo) / 2;

			if (u_pick < lights[mid].selection_cdf)
				hi = mid;
			else
				lo = mid + 1;
		}
		pdf = lights[lo].selection_pmf;

		return lights[lo];
	}

	// ── Directional light sampling ────────────────────────────────────────────
	// Returns the index of the sampled directional light (uniform selection).
	// pdf is the selection probability (1 / N).
	int sample_dir_light(float u_pick, float &pdf) const {
		if (dir_lights.empty()) {
			pdf = 0.f;

			return -1;
		}
		int idx = std::clamp(static_cast<int>(u_pick * static_cast<float>(dir_lights.size())), 0,
							 static_cast<int>(dir_lights.size()) - 1);
		pdf = 1.f / static_cast<float>(dir_lights.size());

		return idx;
	}
};

} // namespace xn
