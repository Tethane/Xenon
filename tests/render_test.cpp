#include "render/wavefront.h"
#include <cstdio>
#include <cmath>
#ifdef XENON_HAS_CUDA
#include "cuda/cuda_renderer.cuh"
#endif

using namespace xn;
static int failures = 0;

static void check(bool ok, const char *message) {
	std::printf("%s: %s\n", ok ? "PASS" : "FAIL", message);

	if (!ok)
		++failures;
}

static std::vector<float> render(const Scene &scene, const Camera &camera, int spp, int depth,
								 int min_bounces = 3, int w = 8, int h = 8) {
	TripleSwapchain swap(w, h);
	WavefrontRenderer cpu(w, h);
	cpu.set_bounces(min_bounces, depth);

	for (int i = 0; i < spp; ++i)
		cpu.render_frame_tiled(scene, camera, swap);
	const float *buffer = swap.get_read_buffer();
	std::vector<float> result(buffer, buffer + w * h * 3);
	cpu.reset_accumulation();
	cpu.render_frame_tiled(scene, camera, swap);
	WavefrontRenderer fresh(w, h);
	fresh.set_bounces(min_bounces, depth);
	TripleSwapchain one(w, h);
	fresh.render_frame_tiled(scene, camera, one);
	bool reset_ok = true;

	for (int i = 0; i < w * h * 3; ++i)
		reset_ok &= swap.get_read_buffer()[i] == one.get_read_buffer()[i];
	check(reset_ok, "CPU accumulation reset clears previous samples");
#ifdef XENON_HAS_CUDA
	CudaRenderer gpu(w, h, depth, min_bounces
#ifdef XENON_TEST_OPTIX
					 ,
					 true
#endif
	);
	gpu.upload_scene(scene);

	for (int i = 0; i < spp; ++i)
		gpu.render_frame(scene, camera, swap);
	buffer = swap.get_read_buffer();
	double error = 0, scale = 0;

	for (int i = 0; i < w * h * 3; ++i) {
		error += std::abs(buffer[i] - result[i]);
		scale += std::abs(result[i]);
	}
	std::printf("  CPU/CUDA relative linear difference %.7g\n", error / std::max(1.0, scale));
	check(std::isfinite(error) && error / std::max(1.0, scale) < 0.006,
		  "CPU/CUDA linear framebuffer agreement");
	result.assign(buffer, buffer + w * h * 3);
#endif
	for (float value : result)
		if (!std::isfinite(value)) {
			check(false, "finite radiance");
			break;
		}

	return result;
}

static float mean(const std::vector<float> &image) {
	double sum = 0;

	for (size_t i = 0; i < image.size(); i += 3)
		sum += image[i];

	return sum / (image.size() / 3);
}

static Camera camera(float fov = 0.01f) {
	Camera c;
	c.look_at({0, 0, 1}, {0, 0, 0}, {0, 1, 0}, fov, 1);

	return c;
}

static void area_light(Scene &scene, bool front = true) {
	TriangleMesh light;
	light.add_vertex({-1, -1, 2});
	light.add_vertex({1, -1, 2});
	light.add_vertex({1, 1, 2});
	light.add_vertex({-1, 1, 2});

	if (front) {
		light.add_triangle(0, 2, 1, 1);
		light.add_triangle(0, 3, 2, 1);
	} else {
		light.add_triangle(0, 1, 2, 1);
		light.add_triangle(0, 2, 3, 1);
	}
	light.compute_smooth_normals();
	scene.meshes.push_back(light);
}

int main() {
	Scene sky;
	sky.sky.zenith_color = {1, 0, 0};
	sky.sky.horizon_color = {0, 0, 0};
	sky.sky.ground_color = {0, 0, 1};
	sky.sky.horizon_sharpness = 1;
	sky.build_acceleration();
	auto orientation = render(sky, camera(90), 16, 1, 0, 7, 5);
	check(orientation[3 * 3] > orientation[(4 * 7 + 3) * 3] &&
			  orientation[(4 * 7 + 3) * 3 + 2] > orientation[3 * 3 + 2],
		  "top-down framebuffer: red sky above blue ground");

	Scene scene;
	Material matte;
	matte.ior = 1.f;
	matte.baseColor = Vec3(0.5f);
	material_prepare(matte);
	Material emitter;
	emitter.baseColor = Vec3(0.f);
	emitter.emission = Vec3(1.f);
	material_prepare(emitter);
	scene.materials = {matte, emitter};
	scene.sky.intensity = 0;
	scene.all_prims.add(Plane{{0, 0, 1}, 0, 0});
	scene.dir_lights = {DirectionalLight{{0, 0, 1}, Vec3(1.f), 3},
						DirectionalLight{{0, 0, 1}, Vec3(1.f), 5}};
	scene.build_acceleration();
	auto sun = render(scene, camera(), 256, 1, 0);
	check(std::abs(mean(sun) - 4 * kInvPi) < 0.008f,
		  "multiple suns reproduce Lambertian irradiance without HDR clamping");

	for (auto &light : scene.dir_lights)
		light.direction = normalize(Vec3(2, 0, 1));
	area_light(scene);
	scene.build_acceleration();
	check(scene.lights.size() == 2 && std::abs(scene.lights[0].area - 2.f) < 1e-6f,
		  "material emission discovers actual triangle areas");
	double area_reference = 0;

	for (int y = 0; y < 512; ++y)
		for (int x = 0; x < 512; ++x) {
			double px = 2.0 * (x + 0.5) / 512 - 1, py = 2.0 * (y + 0.5) / 512 - 1;
			double d = 4 + px * px + py * py;
			area_reference += 0.5 * kInvPi * 4 / (d * d) * 4 / (512 * 512);
		}
	auto combined = render(scene, camera(), 512, 1, 0);
	check(std::abs(mean(combined) - (4 * kInvPi / std::sqrt(5.f) + area_reference)) < 0.012f,
		  "area plus sunlight: both shadow contributions survive");
	scene.dir_lights.clear();
	auto area = render(scene, camera(), 512, 3);
	std::printf("  area result %.7f reference %.7f\n", mean(area), area_reference);
	check(std::abs(mean(area) - area_reference) < 0.003f,
		  "area-light MIS converges to independent irradiance integral");
	auto nee_only = render(scene, camera(), 512, 1, 0);
	check(std::abs(mean(nee_only) - area_reference) < 0.003f &&
			  std::abs(mean(nee_only) - mean(area)) < 0.003f,
		  "adding BSDF light hits with MIS does not double direct illumination");

	// Fixed random streams and no roulette: radiance is linear in source strength
	// pixel by pixel, including both NEE and BSDF-sampled emitter hits.
	auto unit = render(scene, camera(), 32, 3);

	for (auto &light : scene.lights)
		light.emission *= 17.f;
	auto bright = render(scene, camera(), 32, 3);
	bool linear = true;

	for (size_t i = 0; i < unit.size(); ++i)
		linear &= std::abs(bright[i] - 17.f * unit[i]) < 1e-4f;
	check(linear,
		  "17x emitter radiance produces 17x linear output, with no clamping or extra weight");

	for (auto &light : scene.lights)
		light.emission /= 17.f;

	// Subdivide the same emitting square into eight triangles. Selection PMF
	// and area PDF must compensate for the change in triangle count.
	Scene subdivided;
	subdivided.materials = scene.materials;
	subdivided.sky.intensity = 0;
	subdivided.all_prims.add(Plane{{0, 0, 1}, 0, 0});
	TriangleMesh tiled;

	for (int y = 0; y < 2; ++y)
		for (int x = 0; x < 2; ++x) {
			int base = tiled.num_vertices();
			tiled.add_vertex({float(x - 1), float(y - 1), 2});
			tiled.add_vertex({float(x), float(y - 1), 2});
			tiled.add_vertex({float(x), float(y), 2});
			tiled.add_vertex({float(x - 1), float(y), 2});
			tiled.add_triangle(base, base + 2, base + 1, 1);
			tiled.add_triangle(base, base + 3, base + 2, 1);
		}
	tiled.compute_smooth_normals();
	subdivided.meshes.push_back(tiled);
	subdivided.build_acceleration();
	check(std::abs(mean(render(subdivided, camera(), 512, 3)) - area_reference) < 0.003f,
		  "emitter subdivision preserves irradiance and light selection normalization");

	// Far-field square: finite-area integral is the reference, with inverse-
	// square falloff emerging as its angular extent shrinks.
	for (float distance : {20.f, 40.f}) {
		Scene far_light;
		far_light.materials = scene.materials;
		far_light.sky.intensity = 0;
		far_light.materials[1].emission = Vec3(100.f);
		far_light.all_prims.add(Plane{{0, 0, 1}, 0, 0});
		area_light(far_light);

		for (auto &z : far_light.meshes[0].vz)
			z = distance;
		far_light.build_acceleration();
		double reference = 0;

		for (int y = 0; y < 256; ++y)
			for (int x = 0; x < 256; ++x) {
				double px = 2.0 * (x + 0.5) / 256 - 1, py = 2.0 * (y + 0.5) / 256 - 1;
				double d = distance * distance + px * px + py * py;
				reference += 50 * kInvPi * distance * distance / (d * d) * 4 / (256 * 256);
			}
		check(std::abs(mean(render(far_light, camera(), 64, 1, 0)) - reference) < 0.0002f,
			  "area light distance attenuation agrees with independent finite-area integral");
	}

	scene.lights = {{0, 0, Vec3(2.f), 999.f}, {0, 1, Vec3(2.f), 999.f}};
	scene.prepare_lights();
	check(scene.lights[0].area == 2.f, "legacy light area replaced by geometric area");
	auto override_light = render(scene, camera(), 512, 3);
	check(std::abs(mean(override_light) - 2 * area_reference) < 0.006f,
		  "explicit emission override agrees with NEE and hit emission");
	Camera light_camera;
	light_camera.look_at({0, 0, 1}, {0, 0, 2}, {0, 1, 0}, 10, 1);
	check(std::abs(mean(render(scene, light_camera, 4, 1)) - 2.f) < 1e-6f,
		  "camera sees explicit emitter radiance");

	scene.meshes.clear();
	scene.lights.clear();
	area_light(scene, false);
	scene.build_acceleration();
	auto back = render(scene, camera(), 64, 3);
	check(mean(back) == 0.f, "backfaces of one-sided emitters contribute no light");

	scene.dir_lights = {DirectionalLight{normalize(Vec3(2, 0, 1)), Vec3(1.f), 2}};
	auto invalid_area = render(scene, camera(), 8, 1, 0);
	check(std::abs(mean(invalid_area) - kInvPi / std::sqrt(5.f)) < 1e-5f,
		  "rejected area sample does not suppress sunlight");
	scene.dir_lights.clear();

	scene.meshes.clear();
	scene.lights.clear();
	scene.sky.zenith_color = scene.sky.horizon_color = scene.sky.ground_color = Vec3(1.f);
	scene.sky.intensity = 1;
	scene.build_acceleration();
	auto furnace = render(scene, camera(), 512, 2);
	check(std::abs(mean(furnace) - 0.5f) < 0.006f,
		  "Lambertian environment MIS converges to albedo");
	auto environment_nee = render(scene, camera(), 512, 1, 0);
	check(std::abs(mean(environment_nee) - 0.5f) < 0.012f &&
			  std::abs(mean(environment_nee) - mean(furnace)) < 0.012f,
		  "one-bounce environment NEE matches independent Lambertian furnace integral");
	auto roulette = render(scene, camera(), 512, 2, 0);
	check(std::abs(mean(roulette) - 0.5f) < 0.009f, "roulette preserves expected radiance");

	Scene constant;
	constant.sky.zenith_color = constant.sky.horizon_color = constant.sky.ground_color = {2, 4, 8};
	constant.build_acceleration();
	auto first = render(constant, camera(), 1, 1, 0);
	auto accumulated = render(constant, camera(), 37, 1, 0);
	bool averaged = true;

	for (size_t i = 0; i < first.size(); ++i)
		averaged &= first[i] == float(2 << (i % 3)) && accumulated[i] == first[i];
	check(averaged, "HDR accumulation is averaged exactly once at 1 and 37 samples");

	scene.sky.intensity = 0;
	scene.dir_lights = {DirectionalLight{normalize(Vec3(1, 0, 1)), Vec3(1.f), 2}};
	scene.all_prims.add(Sphere{{0.5f, 0, 0.5f}, 0.2f, 0});
	scene.build_acceleration();
	check(mean(render(scene, camera(), 8, 1, 0)) == 0.f, "analytic geometry blocks sunlight");

	Scene distant;
	distant.materials = {matte};
	distant.sky.intensity = 0;
	distant.all_prims.add(Plane{{1, 0, 0}, 1000000.f, 0});
	distant.dir_lights = {DirectionalLight{{1, 0, 0}, Vec3(1.f), 2}};
	distant.build_acceleration();
	Camera far_camera;
	far_camera.look_at({1000002, 0, 0}, {1000000, 0, 0}, {0, 1, 0}, 0.01f, 1);
	check(std::abs(mean(render(distant, far_camera, 4, 1)) - kInvPi) < 1e-5f,
		  "large-coordinate shadows do not self-occlude");

	Scene glass;
	Material dielectric;
	dielectric.baseColor = Vec3(1.f);
	dielectric.transmission = 1;
	dielectric.roughness = 0;
	material_prepare(dielectric);
	glass.materials = {dielectric};
	glass.all_prims.add(Sphere{{0, 0, -1}, 0.5f, 0});
	glass.sky.zenith_color = glass.sky.horizon_color = glass.sky.ground_color = Vec3(1.f);
	glass.build_acceleration();
	auto transmission = render(glass, camera(12), 256, 12, 12);
	check(std::abs(mean(transmission) - 1.f) < 0.004f,
		  "closed glass restores eta scaling in a white environment");
	dielectric.roughness = 0.4f;
	material_prepare(dielectric);
	glass.materials[0] = dielectric;
	auto rough = render(glass, camera(12), 256, 12, 12);
	check(mean(rough) > 0.7f && mean(rough) <= 1.01f,
		  "rough closed glass stays finite and energy bounded");

	std::printf("Rendering: %d failures\n", failures);

	return failures ? 1 : 0;
}
