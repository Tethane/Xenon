// cuda/gpu_scene.cu — Host-side scene flattening and GPU upload
//
// Reads the CPU Scene and produces a flat GpuSceneData on device memory.
// Called once after scene loading / acceleration structure build.

#include <cstdio>
#include <vector>
#include <stdexcept>
#include <unordered_map>

#include "cuda/gpu_scene.cuh"
#include "cuda/cuda_utils.cuh"
#include "scene/scene.h"
#include "camera/camera.h"

namespace xn {

// ─── Helper: upload a vector to device memory ────────────────────────────────

template <typename T> static T *upload_vec(const std::vector<T> &src) {
	if (src.empty())
		return nullptr;
	T *d_ptr = nullptr;
	CUDA_CHECK(cudaMalloc(&d_ptr, src.size() * sizeof(T)));
	CUDA_CHECK(cudaMemcpy(d_ptr, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));

	return d_ptr;
}

template <typename T> static T *upload_data(const T *src, size_t count) {
	if (count == 0 || !src)
		return nullptr;
	T *d_ptr = nullptr;
	CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(T)));
	CUDA_CHECK(cudaMemcpy(d_ptr, src, count * sizeof(T), cudaMemcpyHostToDevice));

	return d_ptr;
}

// ─── Upload ──────────────────────────────────────────────────────────────────

void GpuSceneHost::upload(const Scene &scene) {
	if (uploaded_)
		free_device();

	// Count totals
	int total_vertices = 0;
	int total_indices = 0;
	int total_mat_ids = 0;
	int total_blas_nodes = 0;
	int total_blas_prims = 0;

	int num_meshes = (int)scene.meshes.size();

	for (int i = 0; i < num_meshes; ++i) {
		total_vertices += scene.meshes[i].num_vertices();
		total_indices += (int)scene.meshes[i].indices.size();
		total_mat_ids += (int)scene.meshes[i].mat_ids.size();
		total_blas_nodes += (int)scene.blas_list[i]->nodes().size();
		total_blas_prims += (int)scene.blas_list[i]->prims().size();
	}

	std::printf("[CUDA] Uploading scene: %d meshes, %d vertices, %d triangles, %d materials\n",
				num_meshes, total_vertices, total_indices / 3, (int)scene.materials.size());

	// ── Concatenate mesh data ──────────────────────────────────────────────────

	std::vector<float> all_vx, all_vy, all_vz;
	std::vector<float> all_nx, all_ny, all_nz;
	std::vector<float> all_tu, all_tv;
	std::vector<int32_t> all_indices, all_mat_ids;
	std::vector<GpuBVHNode> all_blas_nodes;
	std::vector<uint32_t> all_blas_prims;
	std::vector<GpuMeshInfo> mesh_info;

	all_vx.reserve(total_vertices);
	all_vy.reserve(total_vertices);
	all_vz.reserve(total_vertices);
	all_nx.reserve(total_vertices);
	all_ny.reserve(total_vertices);
	all_nz.reserve(total_vertices);
	all_indices.reserve(total_indices);
	all_mat_ids.reserve(total_mat_ids);
	all_blas_nodes.reserve(total_blas_nodes);
	all_blas_prims.reserve(total_blas_prims);
	mesh_info.reserve(num_meshes);

	for (int i = 0; i < num_meshes; ++i) {
		const auto &mesh = scene.meshes[i];
		const auto &blas = *scene.blas_list[i];

		GpuMeshInfo mi{};
		mi.vertex_offset = (int)all_vx.size();
		mi.tri_offset = (int)all_indices.size();
		mi.mat_ids_offset = (int)all_mat_ids.size();
		mi.blas_node_offset = (int)all_blas_nodes.size();
		mi.blas_prim_offset = (int)all_blas_prims.size();
		mi.num_triangles = mesh.num_triangles();
		mi.num_vertices = mesh.num_vertices();
		mi.has_normals = mesh.nx.empty() ? 0 : 1;
		mi.has_uv = mesh.tu.size() == mesh.vx.size() && !mesh.tu.empty();

		if (mi.has_uv) {
			all_tu.insert(all_tu.end(), mesh.tu.begin(), mesh.tu.end());
			all_tv.insert(all_tv.end(), mesh.tv.begin(), mesh.tv.end());
		} else {
			all_tu.resize(all_tu.size() + mesh.num_vertices());
			all_tv.resize(all_tv.size() + mesh.num_vertices());
		}

		// Vertices
		all_vx.insert(all_vx.end(), mesh.vx.begin(), mesh.vx.end());
		all_vy.insert(all_vy.end(), mesh.vy.begin(), mesh.vy.end());
		all_vz.insert(all_vz.end(), mesh.vz.begin(), mesh.vz.end());

		// Normals
		if (mi.has_normals) {
			all_nx.insert(all_nx.end(), mesh.nx.begin(), mesh.nx.end());
			all_ny.insert(all_ny.end(), mesh.ny.begin(), mesh.ny.end());
			all_nz.insert(all_nz.end(), mesh.nz.begin(), mesh.nz.end());
		} else {
			all_nx.resize(all_nx.size() + mesh.num_vertices(), 0.f);
			all_ny.resize(all_ny.size() + mesh.num_vertices(), 0.f);
			all_nz.resize(all_nz.size() + mesh.num_vertices(), 0.f);
		}

		// Indices
		all_indices.insert(all_indices.end(), mesh.indices.begin(), mesh.indices.end());

		// Material IDs
		all_mat_ids.insert(all_mat_ids.end(), mesh.mat_ids.begin(), mesh.mat_ids.end());

		// BLAS nodes — convert from BLASNode to GpuBVHNode
		const auto &cpu_nodes = blas.nodes();

		for (const auto &node : cpu_nodes) {
			GpuBVHNode gn{};
			gn.mn = make_float3(node.bbox.mn.x, node.bbox.mn.y, node.bbox.mn.z);
			gn.mx = make_float3(node.bbox.mx.x, node.bbox.mx.y, node.bbox.mx.z);
			gn.child = node.child;
			gn.count = node.count;
			all_blas_nodes.push_back(gn);
		}

		// BLAS prims
		const auto &cpu_prims = blas.prims();
		all_blas_prims.insert(all_blas_prims.end(), cpu_prims.begin(), cpu_prims.end());

		mesh_info.push_back(mi);
	}

	// ── TLAS ───────────────────────────────────────────────────────────────────

	const auto &tlas_cpu_nodes = scene.tlas.nodes();
	const auto &tlas_cpu_inst = scene.tlas.inst_indices();
	const auto &tlas_cpu_instances = scene.tlas.instances();

	std::vector<GpuPrimitive> gpu_primitives;
	std::vector<GpuInstance> gpu_instances;
	auto f3 = [](Vec3 v) { return make_float3(v.x, v.y, v.z); };

	for (const auto &inst : tlas_cpu_instances) {
		GpuInstance gi{};
		gi.instance_id = inst.instance_id;
		gi.mesh_id = -1;

		if (inst.geom.type == GeomHandle::MESH_BLAS) {
			for (int m = 0; m < num_meshes; ++m)
				if (scene.blas_list[m].get() == inst.geom.mesh)
					gi.mesh_id = m;

			if (gi.mesh_id < 0)
				throw std::runtime_error("Instance references an unknown mesh BLAS");
		} else {
			gi.prim_offset = (int)gpu_primitives.size();
			const auto &group = inst.geom.prim->group();
			gi.prim_count = group.size();

			for (const auto &variant : group.prims) {
				GpuPrimitive gp{};
				gp.type = (int)variant.index();
				std::visit(
					[&](const auto &p) {
						gp.mat_id = p.mat_id;

						if constexpr (requires { p.center; })
							gp.center = f3(p.center);

						if constexpr (requires { p.normal; })
							gp.normal = f3(p.normal);

						if constexpr (requires { p.half; })
							gp.half = f3(p.half);

						if constexpr (requires { p.radius; })
							gp.radius = p.radius;

						if constexpr (requires { p.offset; })
							gp.offset = p.offset;

						if constexpr (requires { p.origin; }) {
							gp.origin = f3(p.origin);
							gp.u_vec = f3(p.u_vec);
							gp.v_vec = f3(p.v_vec);
						}
					},
					variant);
				gpu_primitives.push_back(gp);
			}
		}

		// Copy transforms
		for (int c = 0; c < 4; ++c) {
			gi.local_to_world.cols[c] = make_float3(inst.xform.local_to_world.cols[c].x,
													inst.xform.local_to_world.cols[c].y,
													inst.xform.local_to_world.cols[c].z);
			gi.world_to_local.cols[c] = make_float3(inst.xform.world_to_local.cols[c].x,
													inst.xform.world_to_local.cols[c].y,
													inst.xform.world_to_local.cols[c].z);
		}

		gi.world_aabb_mn =
			make_float3(inst.world_aabb.mn.x, inst.world_aabb.mn.y, inst.world_aabb.mn.z);
		gi.world_aabb_mx =
			make_float3(inst.world_aabb.mx.x, inst.world_aabb.mx.y, inst.world_aabb.mx.z);

		gpu_instances.push_back(gi);
	}

	std::vector<GpuBVHNode> tlas_nodes;
	tlas_nodes.reserve(tlas_cpu_nodes.size());

	for (const auto &node : tlas_cpu_nodes) {
		GpuBVHNode gn{};
		gn.mn = make_float3(node.bbox.mn.x, node.bbox.mn.y, node.bbox.mn.z);
		gn.mx = make_float3(node.bbox.mx.x, node.bbox.mx.y, node.bbox.mx.z);
		gn.child = node.child;
		gn.count = node.count;
		tlas_nodes.push_back(gn);
	}

	std::vector<uint32_t> gpu_tlas_inst_indices = tlas_cpu_inst;

	// ── Materials ──────────────────────────────────────────────────────────────

	std::vector<GpuTexture> gpu_textures;
	std::vector<GpuTextureLevel> gpu_levels;
	std::vector<float4> gpu_pixels;
	std::unordered_map<const Texture *, int> texture_ids;
	auto upload_map = [&](const TextureMap &map) {
		GpuTextureMap result;
		result.channel = map.channel;

		if (!map)
			return result;
		auto [it, inserted] = texture_ids.emplace(map.image.get(), int(gpu_textures.size()));
		result.texture = it->second;

		if (inserted) {
			const auto &t = *map.image;
			GpuTexture gt{int(gpu_levels.size()), int(t.levels.size())};
			int offset = int(gpu_pixels.size());

			for (const auto &l : t.levels)
				gpu_levels.push_back({l.width, l.height, l.offset + offset});

			for (const auto &p : t.pixels)
				gpu_pixels.push_back(make_float4(p.r, p.g, p.b, p.a));
			gpu_textures.push_back(gt);
		}

		return result;
	};
	std::vector<GpuMaterial> gpu_mats;
	gpu_mats.reserve(scene.materials.size());

	for (const auto &m : scene.materials) {
		GpuMaterial gm{};
		gm.baseColor = make_float3(m.baseColor.x, m.baseColor.y, m.baseColor.z);
		gm.roughness = m.roughness;
		gm.diffuseRoughness = m.diffuseRoughness;
		gm.multipleScattering = m.multipleScattering;
		gm.clearcoat = m.clearcoat;
		gm.clearcoatRoughness = m.clearcoatRoughness;
		gm.clearcoatIor = m.clearcoatIor;
		gm.absorption = make_float3(m.absorption.x, m.absorption.y, m.absorption.z);
		gm.metallic = m.metallic;
		gm.ior = m.ior;
		gm.transmission = m.transmission;
		gm.subsurface = m.subsurface;
		gm.subsurfaceColor =
			make_float3(m.subsurfaceColor.x, m.subsurfaceColor.y, m.subsurfaceColor.z);
		gm.alpha = m.alpha;
		gm.alpha_x = m.alpha_x;
		gm.alpha_y = m.alpha_y;
		gm.F0 = make_float3(m.F0.x, m.F0.y, m.F0.z);
		gm.emission = make_float3(m.emission.x, m.emission.y, m.emission.z);
		gm.emissionTwoSided = m.emissionTwoSided;
		gm.anisotropy = m.anisotropy;
		gm.opacity = m.opacity;
		gm.alphaCutoff = m.alphaCutoff;
		gm.opacityMode = int(m.opacityMode);
		gm.normalScale = m.normalScale;
		gm.bumpScale = m.bumpScale;
		gm.textureLod = m.textureLod;
		gm.baseColorTexture = upload_map(m.baseColorTexture);
		gm.roughnessTexture = upload_map(m.roughnessTexture);
		gm.metallicTexture = upload_map(m.metallicTexture);
		gm.emissionTexture = upload_map(m.emissionTexture);
		gm.opacityTexture = upload_map(m.opacityTexture);
		gm.normalTexture = upload_map(m.normalTexture);
		gm.bumpTexture = upload_map(m.bumpTexture);
		gm.flags = 0;

		if (m.isConductor)
			gm.flags |= GpuMaterial::FLAG_CONDUCTOR;

		if (m.isTransmissive)
			gm.flags |= GpuMaterial::FLAG_TRANSMISSIVE;

		if (m.hasSubsurface)
			gm.flags |= GpuMaterial::FLAG_SUBSURFACE;

		if (m.isDelta)
			gm.flags |= GpuMaterial::FLAG_DELTA;

		if (m.isEmissive)
			gm.flags |= GpuMaterial::FLAG_EMISSIVE;
		gpu_mats.push_back(gm);
	}

	// ── Lights ─────────────────────────────────────────────────────────────────

	std::vector<GpuLight> gpu_lights;
	gpu_lights.reserve(scene.lights.size());

	for (const auto &l : scene.lights) {
		GpuLight gl{};
		gl.instance_id = l.instance_id;
		gl.p0 = f3(l.p0);
		gl.p1 = f3(l.p1);
		gl.p2 = f3(l.p2);
		gl.normal = f3(l.normal);
		gl.mesh_id = l.mesh_id;
		gl.tri_idx = l.tri_idx;
		gl.emission = make_float3(l.emission.x, l.emission.y, l.emission.z);
		gl.area = l.area;
		gl.material_emission = l.material_emission;
		gl.emissionTwoSided = l.emissionTwoSided;
		gl.selection_pmf = l.selection_pmf;
		gl.selection_cdf = l.selection_cdf;
		gpu_lights.push_back(gl);
	}

	std::vector<GpuDirLight> gpu_dir_lights;
	gpu_dir_lights.reserve(scene.dir_lights.size());

	for (const auto &dl : scene.dir_lights) {
		GpuDirLight gdl{};
		gdl.direction = make_float3(dl.direction.x, dl.direction.y, dl.direction.z);
		gdl.color = make_float3(dl.color.x, dl.color.y, dl.color.z);
		gdl.intensity = dl.intensity;
		gpu_dir_lights.push_back(gdl);
	}

	std::vector<GpuPointLight> gpu_points;

	for (const auto &p : scene.point_lights)
		gpu_points.push_back(
			{f3(p.position), f3(p.intensity), f3(p.direction), p.cos_inner, p.cos_outer});
	d_point_lights_ = upload_vec(gpu_points);
	scene_data_.point_lights = d_point_lights_;
	scene_data_.num_point_lights = int(gpu_points.size());
	std::vector<GpuMedium> gpu_media;

	for (const auto &m : scene.media)
		gpu_media.push_back({f3(m.minimum), f3(m.maximum), f3(m.sigma_a), f3(m.sigma_s), m.g});
	d_media_ = upload_vec(gpu_media);
	scene_data_.media = d_media_;
	scene_data_.num_media = int(gpu_media.size());

	// ── Upload to device ───────────────────────────────────────────────────────

	d_tu_ = upload_vec(all_tu);
	d_tv_ = upload_vec(all_tv);
	d_textures_ = upload_vec(gpu_textures);
	d_texture_levels_ = upload_vec(gpu_levels);
	d_texture_pixels_ = upload_vec(gpu_pixels);
	d_vx_ = upload_vec(all_vx);
	d_vy_ = upload_vec(all_vy);
	d_vz_ = upload_vec(all_vz);
	d_nx_ = upload_vec(all_nx);
	d_ny_ = upload_vec(all_ny);
	d_nz_ = upload_vec(all_nz);
	d_indices_ = upload_vec(all_indices);
	d_mat_ids_ = upload_vec(all_mat_ids);
	d_blas_nodes_ = upload_vec(all_blas_nodes);
	d_blas_prims_ = upload_vec(all_blas_prims);
	d_tlas_nodes_ = upload_vec(tlas_nodes);
	d_tlas_inst_indices_ = upload_vec(gpu_tlas_inst_indices);
	d_mesh_info_ = upload_vec(mesh_info);
	d_instances_ = upload_vec(gpu_instances);
	d_primitives_ = upload_vec(gpu_primitives);
	d_materials_ = upload_vec(gpu_mats);
	d_lights_ = upload_vec(gpu_lights);
	d_dir_lights_ = upload_vec(gpu_dir_lights);

	// ── Fill scene data struct ─────────────────────────────────────────────────

	scene_data_.texcoords_u = d_tu_;
	scene_data_.texcoords_v = d_tv_;
	scene_data_.textures = d_textures_;
	scene_data_.texture_levels = d_texture_levels_;
	scene_data_.texture_pixels = d_texture_pixels_;
	scene_data_.vertices_x = d_vx_;
	scene_data_.vertices_y = d_vy_;
	scene_data_.vertices_z = d_vz_;
	scene_data_.normals_x = d_nx_;
	scene_data_.normals_y = d_ny_;
	scene_data_.normals_z = d_nz_;
	scene_data_.indices = d_indices_;
	scene_data_.mat_ids = d_mat_ids_;
	scene_data_.blas_nodes = d_blas_nodes_;
	scene_data_.blas_prims = d_blas_prims_;
	scene_data_.tlas_nodes = d_tlas_nodes_;
	scene_data_.tlas_inst_indices = d_tlas_inst_indices_;
	scene_data_.mesh_info = d_mesh_info_;
	scene_data_.instances = d_instances_;
	scene_data_.primitives = d_primitives_;
	scene_data_.materials = d_materials_;
	scene_data_.lights = d_lights_;
	scene_data_.dir_lights = d_dir_lights_;

	// Environment
	scene_data_.env.zenith_color =
		make_float3(scene.sky.zenith_color.x, scene.sky.zenith_color.y, scene.sky.zenith_color.z);
	scene_data_.env.horizon_color = make_float3(
		scene.sky.horizon_color.x, scene.sky.horizon_color.y, scene.sky.horizon_color.z);
	scene_data_.env.ground_color =
		make_float3(scene.sky.ground_color.x, scene.sky.ground_color.y, scene.sky.ground_color.z);
	scene_data_.env.horizon_sharpness = scene.sky.horizon_sharpness;
	scene_data_.env.intensity = scene.sky.intensity;

	if (scene.sky.image) {
		const auto &level = scene.sky.image->levels[0];
		std::vector<float4> pixels;
		pixels.reserve(size_t(level.width) * level.height);

		for (size_t i = 0; i < size_t(level.width) * level.height; ++i) {
			auto p = scene.sky.image->pixels[i];
			pixels.push_back(make_float4(p.r, p.g, p.b, p.a));
		}
		d_environment_pixels_ = upload_vec(pixels);
		d_environment_cdf_ = upload_vec(scene.sky.cdf);
		scene_data_.env.pixels = d_environment_pixels_;
		scene_data_.env.cdf = d_environment_cdf_;
		scene_data_.env.width = level.width;
		scene_data_.env.height = level.height;
		scene_data_.env.rotation = scene.sky.rotation;
	}

	// Counts
	scene_data_.num_meshes = num_meshes;
	scene_data_.num_materials = (int)scene.materials.size();
	scene_data_.has_alpha_blend = scene.has_alpha_blend;
	scene_data_.num_lights = (int)gpu_lights.size();
	scene_data_.num_dir_lights = (int)gpu_dir_lights.size();
	scene_data_.num_instances = (int)gpu_instances.size();
	scene_data_.num_tlas_nodes = (int)tlas_nodes.size();

	uploaded_ = true;

	size_t total_bytes = 0;
	total_bytes += total_vertices * 3 * sizeof(float) * 2; // positions + normals
	total_bytes += total_vertices * sizeof(float) * 2;
	total_bytes += gpu_points.size() * sizeof(GpuPointLight);
	total_bytes += gpu_media.size() * sizeof(GpuMedium);
	total_bytes += gpu_textures.size() * sizeof(GpuTexture) +
				   gpu_levels.size() * sizeof(GpuTextureLevel) + gpu_pixels.size() * sizeof(float4);
	total_bytes += total_indices * sizeof(int32_t);
	total_bytes += total_mat_ids * sizeof(int32_t);
	total_bytes += all_blas_nodes.size() * sizeof(GpuBVHNode);
	total_bytes += all_blas_prims.size() * sizeof(uint32_t);
	total_bytes += tlas_nodes.size() * sizeof(GpuBVHNode);
	total_bytes += tlas_cpu_inst.size() * sizeof(uint32_t);
	total_bytes += mesh_info.size() * sizeof(GpuMeshInfo);
	total_bytes += gpu_instances.size() * sizeof(GpuInstance);
	total_bytes += gpu_primitives.size() * sizeof(GpuPrimitive);
	total_bytes += gpu_mats.size() * sizeof(GpuMaterial);
	total_bytes += gpu_lights.size() * sizeof(GpuLight);
	total_bytes += gpu_dir_lights.size() * sizeof(GpuDirLight);

	std::printf("[CUDA] Scene uploaded: %.2f MB device memory\n", total_bytes / (1024.0 * 1024.0));
	std::printf("[CUDA]   %d BLAS nodes, %d TLAS nodes, %d instances\n", (int)all_blas_nodes.size(),
				(int)tlas_nodes.size(), (int)gpu_instances.size());
}

// ─── Free device memory ──────────────────────────────────────────────────────

void GpuSceneHost::free_device() {
	if (!uploaded_)
		return;

	auto safe_free = [](auto *&p) {
		if (p) {
			cudaFree(p);
			p = nullptr;
		}
	};

	safe_free(d_point_lights_);
	safe_free(d_media_);
	safe_free(d_environment_pixels_);
	safe_free(d_environment_cdf_);
	safe_free(d_tu_);
	safe_free(d_tv_);
	safe_free(d_textures_);
	safe_free(d_texture_levels_);
	safe_free(d_texture_pixels_);
	safe_free(d_vx_);
	safe_free(d_vy_);
	safe_free(d_vz_);
	safe_free(d_nx_);
	safe_free(d_ny_);
	safe_free(d_nz_);
	safe_free(d_indices_);
	safe_free(d_mat_ids_);
	safe_free(d_blas_nodes_);
	safe_free(d_blas_prims_);
	safe_free(d_tlas_nodes_);
	safe_free(d_tlas_inst_indices_);
	safe_free(d_primitives_);
	safe_free(d_mesh_info_);
	safe_free(d_instances_);
	safe_free(d_materials_);
	safe_free(d_lights_);
	safe_free(d_dir_lights_);

	scene_data_ = {};
	uploaded_ = false;
}

GpuSceneHost::~GpuSceneHost() {
	free_device();
}

// ─── Camera upload ───────────────────────────────────────────────────────────

GpuCamera GpuSceneHost::upload_camera(const Camera &cam, int width, int height) const {
	GpuCamera gc{};
	Vec3 eye = cam.get_eye();
	Vec3 ll = cam.get_lower_left_corner();
	Vec3 h = cam.get_horizontal();
	Vec3 v = cam.get_vertical();

	gc.eye = make_float3(eye.x, eye.y, eye.z);
	gc.lower_left_corner = make_float3(ll.x, ll.y, ll.z);
	gc.horizontal = make_float3(h.x, h.y, h.z);
	gc.vertical = make_float3(v.x, v.y, v.z);
	Vec3 right = cam.get_right(), up = cam.get_up();
	gc.right = make_float3(right.x, right.y, right.z);
	gc.up = make_float3(up.x, up.y, up.z);
	gc.lens_radius = cam.get_lens_radius();
	gc.focus_distance = cam.get_focus_distance();
	gc.medium_material = cam.get_medium_material();
	Vec3 absorption = cam.get_absorption();
	gc.absorption = make_f3(absorption.x, absorption.y, absorption.z);

	return gc;
}

} // namespace xn
