#include "scene/scene.h"
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
	std::printf("%s: %s\n", ok ? "PASS" : "FAIL", message);
	failures += !ok;
}

static Material blend(float opacity) {
	Material m;
	m.opacityMode = Material::OpacityMode::Blend;
	m.opacity = opacity;
	material_prepare(m);

	return m;
}

static TriangleMesh layer(float z, int material = 0) {
	TriangleMesh m;
	m.add_vertex({-10, -10, z});
	m.add_vertex({10, -10, z});
	m.add_vertex({0, 10, z});
	m.set_uv(0, 0, 0);
	m.set_uv(1, 1, 0);
	m.set_uv(2, .5, 1);
	m.add_triangle(0, 1, 2, material);

	return m;
}
#ifdef XENON_HAS_CUDA
__global__ void query_opacity(GpuSceneData scene, const GpuRay *rays, GpuHitRecord *hits,
							  bool *shadows, int count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= count)
		return;
	GpuHitRecord rec;
	gpu_tlas_intersect(rays[i], scene, rec);
	hits[i] = rec;
	shadows[i] = gpu_tlas_intersects(rays[i], scene);
}
#endif
static void transmission(Scene &scene, Ray ray, double expected, const char *name) {
	constexpr int count = 65536;
	std::vector<Ray> rays(count, ray);

	for (int i = 0; i < count; ++i)
		rays[i].alpha_seed = opacity_hash(uint32_t(i) + 927341u);
#ifdef XENON_HAS_CUDA
	GpuSceneHost gpu;
	gpu.upload(scene);
	std::vector<GpuRay> gr(count);

	for (int i = 0; i < count; ++i)
		gr[i] = {make_f3(ray.origin.x, ray.origin.y, ray.origin.z),
				 make_f3(ray.dir.x, ray.dir.y, ray.dir.z), ray.tmin, ray.tmax, rays[i].alpha_seed};
	GpuRay *dr = nullptr;
	GpuHitRecord *dh = nullptr;
	bool *ds = nullptr;
	CUDA_CHECK(cudaMalloc(&dr, count * sizeof(GpuRay)));
	CUDA_CHECK(cudaMalloc(&dh, count * sizeof(GpuHitRecord)));
	CUDA_CHECK(cudaMalloc(&ds, count * sizeof(bool)));
	CUDA_CHECK(cudaMemcpy(dr, gr.data(), count * sizeof(GpuRay), cudaMemcpyHostToDevice));
#ifdef XENON_TEST_OPTIX
	OptixBackend optix;
	optix.build(scene, gpu.get_scene_data());
	optix.query(gpu.get_scene_data(), dr, dh, ds, count);
#else
	query_opacity<<<(count + 255) / 256, 256>>>(gpu.get_scene_data(), dr, dh, ds, count);
#endif
	CUDA_CHECK(cudaDeviceSynchronize());
	std::vector<GpuHitRecord> hits(count);
	auto shadows = std::make_unique<bool[]>(count);
	CUDA_CHECK(cudaMemcpy(hits.data(), dh, count * sizeof(GpuHitRecord), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(shadows.get(), ds, count * sizeof(bool), cudaMemcpyDeviceToHost));
	cudaFree(dr);
	cudaFree(dh);
	cudaFree(ds);
#endif
	int misses = 0, disagreements = 0;

	for (int i = 0; i < count; ++i) {
		HitRecord hit;
		bool found = scene.intersect(rays[i], hit);
		misses += !found;
		disagreements += found != scene.intersects(rays[i]);
#ifdef XENON_HAS_CUDA
		const auto &gh = hits[i];
		disagreements += found != gh.valid() || found != shadows[i];

		if (found && gh.valid())
			disagreements += std::abs(hit.t - gh.t) > 1e-5f || hit.prim_id != gh.prim_id ||
							 hit.instance_id != gh.instance_id || hit.front_face != gh.front_face;
#endif
	}
	double measured = double(misses) / count;
	std::printf("%s transmittance %.7f, analytic %.7f\n", name, measured, expected);
	check(std::abs(measured - expected) < .008, name);
	check(disagreements == 0, "seeded closest/shadow queries agree exactly across backends");
}

int main() {
	Scene layers;
	layers.materials = {blend(.25f), blend(.8f)};
	auto texture = std::make_shared<Texture>();
	texture->levels = {{1, 1, 0}};
	texture->pixels = {{.75f, .75f, .75f, 1}};
	layers.materials[1].opacityTexture.image = texture;
	layers.meshes = {layer(1, 0), layer(0, 1)};
	layers.build_acceleration();
	check(layers.has_alpha_blend && !layers.has_alpha_masks,
		  "blend coverage detected without changing mask mode");
	transmission(layers, Ray{{.3f, .1f, 3}, {0, 0, -1}}, .75 * .4,
				 "two textured layers multiply transmission");
	transmission(layers, Ray{{.3f, .1f, 3}, {0, 0, -1}, 0, 3}, .75,
				 "finite shadow excludes its endpoint layer");

	Scene instances;
	instances.materials = {blend(.5f)};
	instances.meshes = {layer(0)};
	instances.explicit_instances = true;
	Mat4x3 affine = Mat4x3::identity();
	affine.cols[0] = {-2, 0, 0};
	affine.cols[1] = {.2f, 3, 0};
	affine.cols[3] = {0, 0, 1};
	instances.mesh_placements = {{0, AffineTransform{}}, {0, AffineTransform::from_matrix(affine)}};
	instances.build_acceleration();
	auto logical = instances.tlas.instances();
	logical[0].instance_id = 31;
	logical[1].instance_id = 99;
	instances.tlas.build(logical);
	transmission(instances, Ray{{.3f, .1f, 3}, {0, 0, -1}}, .25,
				 "shared mirrored affine instances have independent coverage");

	Scene sphere;
	sphere.materials = {blend(.5f)};
	sphere.all_prims.add(Sphere{{0, 0, 0}, 1, 0});
	sphere.build_acceleration();
	transmission(sphere, Ray{{0, 0, 3}, {0, 0, -1}}, .25,
				 "analytic sphere entry and exit independently attenuate");
	transmission(sphere, Ray{{0, 0, 0}, {0, 0, -1}}, .5,
				 "analytic sphere interior ray samples exit coverage");
	Scene planes;
	planes.materials = {blend(.2f), blend(.4f), blend(.6f)};
	planes.all_prims.add(Plane{{0, 0, 1}, 1, 0});
	planes.all_prims.add(Plane{{0, 0, 1}, 0, 1});
	planes.all_prims.add(Plane{{0, 0, 1}, -1, 2});
	planes.build_acceleration();
	transmission(planes, Ray{{200000, 0, 3}, {0, 0, -1}}, .8 * .6 * .4,
				 "unbounded analytic alpha planes retain infinite support");

	Scene coplanar;
	Material cut;
	cut.opacity = 0;
	material_prepare(cut);
	Material opaque;
	material_prepare(opaque);
	coplanar.materials = {cut, opaque};
	auto mesh = layer(0);
	mesh.add_triangle(0, 1, 2, 1);
	coplanar.meshes = {mesh};
	coplanar.build_acceleration();
	transmission(coplanar, Ray{{0, 0, 1}, {0, 0, -1}}, 0,
				 "rejected coplanar mask preserves opaque geometry");
	layers.materials[0].emission = {4, 8, 12};
	layers.build_acceleration();
	auto radiance = layers.light_emission(layers.lights.front(), .2f, .3f);
	check((radiance - Vec3(1, 2, 3)).length() < 1e-6f,
		  "emissive NEE scales raw emission by fractional coverage");

	auto dir =
		std::filesystem::temp_directory_path() / ("xenon-opacity-test-" + std::to_string(getpid()));
	std::filesystem::create_directories(dir);
	{
		std::ofstream f(dir / "blend.mat");
		f << "version 2\nopacityMode blend\nopacity .3\n";
	}
	check(load_material((dir / "blend.mat").string()).opacityMode == Material::OpacityMode::Blend,
		  "material parser accepts explicit blend mode");
	{
		std::ofstream f(dir / "bad.mat");
		f << "version 2\nopacityMode translucent\n";
	}
	bool rejected = false;

	try {
		load_material((dir / "bad.mat").string());
	} catch (const std::exception &) {
		rejected = true;
	}
	check(rejected, "unknown opacity mode rejected");
	std::filesystem::remove_all(dir);

	return failures ? 1 : 0;
}
