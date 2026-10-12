#include "scene/scene.h"
#include "camera/sampler.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#ifdef XENON_HAS_CUDA
#include "cuda/gpu_scene.cuh"
#include "cuda/cuda_utils.cuh"
#ifdef XENON_TEST_OPTIX
#include "optix/optix_backend.h"
#endif
#endif

using namespace xn;
static int failures = 0;

static void check(bool ok, const char *message) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", message);
		++failures;
	}
}

static bool close(Vec3 a, Vec3 b, float eps = 2e-4f) {
	return (a - b).length() < eps;
}

static bool brute(const Scene &scene, const Ray &ray, HitRecord &rec) {
	bool hit = false;

	for (const auto &inst : scene.tlas.instances()) {
		Ray local = inst.xform.to_local(ray);
		HitRecord candidate = rec;
		bool found = false;

		if (inst.geom.type == GeomHandle::MESH_BLAS) {
			for (size_t m = 0; m < scene.meshes.size(); ++m) {
				if (scene.blas_list[m].get() != inst.geom.mesh)
					continue;

				for (int t = 0; t < scene.meshes[m].num_triangles(); ++t)
					found |= scene.meshes[m].intersect_triangle(local, t, candidate);
			}
		} else {
			const auto &group = inst.geom.prim->group();

			for (int p = 0; p < group.size(); ++p)
				found |= prim_intersect(group.prims[p], local, candidate, p);
		}

		if (found) {
			rec = candidate;
			rec.pos = ray.at(rec.t);
			rec.normal = inst.xform.normal_to_world(rec.normal);
			rec.geo_normal = inst.xform.normal_to_world(rec.geo_normal);
			rec.front_face = dot(rec.geo_normal, ray.dir) < 0.f;
			rec.instance_id = inst.instance_id;
			hit = true;
		}
	}

	return hit;
}

#ifdef XENON_HAS_CUDA
__global__ void trace(GpuSceneData scene, const GpuRay *rays, GpuHitRecord *hits, bool *shadows,
					  int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n)
		return;
	GpuHitRecord rec;
	gpu_tlas_intersect(rays[i], scene, rec);
	hits[i] = rec;
	shadows[i] = gpu_tlas_intersects(rays[i], scene);
}
#endif

static void compare(Scene &scene, const std::vector<Ray> &rays) {
#ifdef XENON_HAS_CUDA
	GpuSceneHost gpu;
	gpu.upload(scene);
	std::vector<GpuRay> gr;

	for (auto r : rays)
		gr.push_back({make_f3(r.origin.x, r.origin.y, r.origin.z),
					  make_f3(r.dir.x, r.dir.y, r.dir.z), r.tmin, r.tmax});
	GpuRay *dr;
	GpuHitRecord *dh;
	bool *ds;
	CUDA_CHECK(cudaMalloc(&dr, gr.size() * sizeof(GpuRay)));
	CUDA_CHECK(cudaMalloc(&dh, gr.size() * sizeof(GpuHitRecord)));
	CUDA_CHECK(cudaMalloc(&ds, gr.size() * sizeof(bool)));
	CUDA_CHECK(cudaMemcpy(dr, gr.data(), gr.size() * sizeof(GpuRay), cudaMemcpyHostToDevice));
#ifdef XENON_TEST_OPTIX
	OptixBackend optix;
	optix.build(scene, gpu.get_scene_data());
	optix.query(gpu.get_scene_data(), dr, dh, ds, gr.size());
#else
	trace<<<(gr.size() + 255) / 256, 256>>>(gpu.get_scene_data(), dr, dh, ds, gr.size());
#endif
	CUDA_CHECK(cudaDeviceSynchronize());
	std::vector<GpuHitRecord> gh(gr.size());
	auto shadows = std::make_unique<bool[]>(gr.size());
	CUDA_CHECK(cudaMemcpy(gh.data(), dh, gr.size() * sizeof(GpuHitRecord), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(shadows.get(), ds, gr.size() * sizeof(bool), cudaMemcpyDeviceToHost));
	cudaFree(dr);
	cudaFree(dh);
	cudaFree(ds);
#endif
	for (size_t i = 0; i < rays.size(); ++i) {
		HitRecord ref, cpu;
		bool expected = brute(scene, rays[i], ref);
		bool actual = scene.intersect(rays[i], cpu);
		check(actual == expected, "CPU closest hit/miss");
		check(scene.intersects(rays[i]) == expected, "CPU shadow interval");

		if (expected && actual) {
			check(std::abs(cpu.t - ref.t) < 2e-4f, "CPU hit distance");
			check(cpu.prim_id == ref.prim_id && cpu.instance_id == ref.instance_id,
				  "CPU hit identity");
			check(close(cpu.normal, ref.normal) && close(cpu.geo_normal, ref.geo_normal),
				  "CPU normals");
		}
#ifdef XENON_HAS_CUDA
		const auto &h = gh[i];
		check(h.valid() == expected, "CUDA closest hit/miss");
		check(shadows[i] == expected, "CUDA shadow interval");

		if (expected && h.valid()) {
			check(std::abs(h.t - ref.t) < 2e-3f * std::max(1.f, ref.t), "CUDA hit distance");
			check(h.prim_id == ref.prim_id && h.instance_id == ref.instance_id &&
					  h.mat_id == ref.mat_id,
				  "CUDA hit identity");
			check(
				close({h.normal.x, h.normal.y, h.normal.z}, ref.normal, 2e-3f) &&
					close({h.geo_normal.x, h.geo_normal.y, h.geo_normal.z}, ref.geo_normal, 2e-3f),
				"CUDA normals");
			check(h.front_face == ref.front_face, "CUDA entering/exiting");
		}
#endif
	}
}

int main() {
	Box box{{0, 0, 0}, {1, 1, 1}, 0};
	HitRecord h;
	check(box.intersect({{0, 0, 0}, {0, 1, 0}}, h) && h.t == 1.f && !h.front_face &&
			  h.normal.y == 1.f,
		  "box exit from inside");
	h = {};
	check(!box.intersect({{2, 0, 0}, {0, 1, 0}}, h), "parallel box miss");
	h = {};
	check(!box.intersect({{0, 0, 3}, {0, 0, -1}, 0.f, 1.f}, h), "primitive honors tmax");
	Vec3 shifted = offset_ray_origin({1000000, 0, 0}, {1, 0, 0}, {1, 0, 0});
	check(shifted.x > 1000000, "ray offset moves outward at large coordinates");
	AABB bounds{{-1, -1, -1}, {1, 1, 1}};
	float tn, tf;
	check(bounds.intersect({{1, 0, 3}, {0, 0, -1}}, tn, tf) && tn == 2.f, "parallel boundary slab");

	float t, u, v;
	check(ray_triangle({{2e-5f, 2e-5f, 1}, {0, 0, -1}}, {0, 0, 0}, {1e-4f, 0, 0}, {0, 1e-4f, 0}, t,
					   u, v) &&
			  t == 1.f,
		  "small triangles are scale invariant");
	Sphere sphere{{0, 0, 0}, 1, 0};
	h = {};
	check(sphere.intersect({{0, 0, 0}, {0, 0, 1}}, h) && h.t == 1.f && !h.front_face &&
			  h.normal.z == 1.f,
		  "sphere exit preserves outward normal");
	h = {};
	check(sphere.intersect({{0, 0, 1000000}, {0, 0, -1}}, h) && h.t == 999999.f,
		  "distant sphere roots retain precision");
	Plane plane{{0, -1, 0}, -2, 0};
	h = {};
	check(plane.intersect({{20000, 3, 0}, {0, -1, 0}}, h) && h.t == 1.f && !h.front_face,
		  "negative-normal infinite plane");

	Scene scene;
	scene.materials.resize(1);
	TriangleMesh mesh;

	for (int i = 0; i < 24; ++i) {
		float x = (i % 6) - 3.f, y = (i / 6) - 2.f;
		int v = mesh.num_vertices();
		mesh.add_vertex({x, y, 0});
		mesh.add_vertex({x + 0.8f, y, 0});
		mesh.add_vertex({x, y + 0.8f, 0});
		mesh.add_triangle(v, v + 1, v + 2);
	}
	mesh.compute_smooth_normals();
	scene.meshes.push_back(mesh);
	scene.all_prims.add(Sphere{{-2, 0, 0}, 0.7f, 0});
	scene.all_prims.add(box);
	scene.all_prims.add(Disk{{2, 0, 0}, {0, 0, 1}, 0.8f, 0});
	scene.all_prims.add(Quad{{-1, 2, 0}, {2, 0, 0}, {0.5f, 1, 0}, 0});
	scene.build_acceleration();
	auto instances = scene.tlas.instances();
	instances[0].xform = AffineTransform::from_trs({0, 0, -2}, 1.7f, {20, 35, 17});
	instances[1].xform = AffineTransform::from_trs({1, 0, 0}, 0.8f, {15, -20, 5});
	Instance mirror = instances[0];
	mirror.xform = AffineTransform::from_trs({-4, 0, 1}, -0.7f, {10, 22, 32});
	mirror.instance_id = 2;
	instances.push_back(mirror);
	Instance stretched = instances[0];
	stretched.instance_id = 3;
	stretched.xform.local_to_world.cols[0] *= 2.f;

	for (int c = 0; c < 4; ++c)
		stretched.xform.world_to_local.cols[c].x *= 0.5f;
	stretched.xform.local_to_world.cols[3] += Vec3(4, 1, 2);
	stretched.xform.world_to_local.cols[3] =
		stretched.xform.world_to_local.transform_dir(-stretched.xform.local_to_world.cols[3]);
	instances.push_back(stretched);
	scene.tlas.build(instances);
	PCGState rng = seed_pcg(87);
	std::vector<Ray> rays;

	for (int i = 0; i < 10000; ++i) {
		Vec3 org{12 * rng.next_float() - 6, 10 * rng.next_float() - 5, 5};
		Vec3 dir =
			normalize(Vec3(0.4f * rng.next_float() - 0.2f, 0.4f * rng.next_float() - 0.2f, -1));
		rays.push_back({org, dir, 0.f, (i % 3 == 0) ? 3.f : kInfinity});
	}
	compare(scene, rays);
	scene.all_prims.add(Plane{normalize(Vec3(1, 2, 3)), 2.f, 0});
	scene.build_acceleration();
	rays.push_back({{20000, 20000, 20000}, normalize(Vec3(-1, -2, -3))});
	compare(scene, rays);
	Scene empty;
	empty.materials.resize(1);
	empty.build_acceleration();
	compare(empty, rays);
	Scene analytic;
	analytic.materials.resize(1);
	analytic.all_prims.add(box);
	analytic.build_acceleration();
	compare(analytic, rays);

	auto path = std::filesystem::temp_directory_path() /
				("xenon-geometry-" + std::to_string(getpid()) + ".obj");
	auto load = [&](const char *text, TriangleMesh &m) {
		std::ofstream(path) << text;

		return load_obj(path.string(), m, {});
	};
	TriangleMesh obj;
	check(load("  v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf -3//1 -2 -1 # mixed normals\n", obj),
		  "OBJ relative indices and whitespace");
	check(obj.num_triangles() == 1 && obj.nz[1] > 0.99f, "OBJ fills missing normals");
	check(!load("v 0 0 0\nf 1 2 99\n", obj), "OBJ out-of-range indices rejected");
	check(!load("v 0 0 0\nf 0 0 0\n", obj), "OBJ zero indices rejected");
	check(!load("v 0 0 0\nf 1x 1 1\n", obj), "OBJ malformed indices rejected");
	check(load("v 0 0 0\nv 2 0 0\nv 2 2 0\nv 1 1 0\nv 0 2 0\nf 1 2 3 4 5\n", obj),
		  "OBJ concave polygon triangulation");
	float area = 0.f;

	for (int i = 0; i < obj.num_triangles(); ++i) {
		Vec3 a, b, c;
		obj.get_triangle(i, a, b, c);
		area += 0.5f * cross(b - a, c - a).length();
	}
	check(std::abs(area - 3.f) < 1e-6f, "concave polygon preserves area");
	std::filesystem::remove(path);
	std::printf("Geometry: %d failures\n", failures);

	return failures ? 1 : 0;
}
