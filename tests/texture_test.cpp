#include "material/texture.h"
#include "material/material.h"
#include "scene/scene_file.h"
#include "render/wavefront.h"
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <unistd.h>
#ifdef XENON_HAS_CUDA
#include "cuda/cuda_renderer.cuh"
#include "cuda/gpu_scene.cuh"
#ifdef XENON_TEST_OPTIX
#include "optix/optix_backend.h"
#endif
#endif
using namespace xn;
static int failures = 0;

static void check(bool ok, const char *label) {
	std::printf("%s: %s\n", ok ? "PASS" : "FAIL", label);
	failures += !ok;
}

static bool near(float a, float b, float eps = 1e-5f) {
	return std::abs(a - b) < eps;
}

static void write(const std::filesystem::path &p, const std::string &s) {
	std::ofstream f(p);
	f << s;
}

static std::shared_ptr<Texture> constant(TexturePixel p) {
	auto t = std::make_shared<Texture>();
	t->levels = {{1, 1, 0}};
	t->pixels = {p};

	return t;
}
#ifdef XENON_HAS_CUDA
struct QueryResult {
	GpuHitRecord hit;
	GpuMaterial mat;
	bool found, shadow;
};

__global__ void query(GpuSceneData scene, GpuRay ray, QueryResult *out) {
	out->hit = GpuHitRecord{};
	out->found = gpu_tlas_intersect(ray, scene, out->hit);
	out->shadow = gpu_tlas_intersects(ray, scene);

	if (out->found)
		out->mat = gpu_evaluate_material(scene, out->hit);
}
#ifdef XENON_TEST_OPTIX
__global__ void evaluate_query(GpuSceneData scene, const GpuHitRecord *hit, const bool *shadow,
							   QueryResult *out) {
	out->hit = *hit;
	out->found = hit->valid();
	out->shadow = *shadow;

	if (out->found)
		out->mat = gpu_evaluate_material(scene, out->hit);
}
#endif
static void compare_gpu_query(Scene &scene, Ray ray) {
	GpuSceneHost gpu;
	gpu.upload(scene);
	QueryResult *device = nullptr, result{};
	cudaMalloc(&device, sizeof(result));
	GpuRay gr{make_f3(ray.origin.x, ray.origin.y, ray.origin.z),
			  make_f3(ray.dir.x, ray.dir.y, ray.dir.z), ray.tmin, ray.tmax};
#ifdef XENON_TEST_OPTIX
	OptixBackend optix;
	optix.build(scene, gpu.get_scene_data());
	GpuRay *dr = nullptr;
	GpuHitRecord *dh = nullptr;
	bool *ds = nullptr;
	cudaMalloc(&dr, sizeof(gr));
	cudaMalloc(&dh, sizeof(GpuHitRecord));
	cudaMalloc(&ds, sizeof(bool));
	cudaMemcpy(dr, &gr, sizeof(gr), cudaMemcpyHostToDevice);
	optix.query(gpu.get_scene_data(), dr, dh, ds, 1);
	evaluate_query<<<1, 1>>>(gpu.get_scene_data(), dh, ds, device);
	cudaFree(dr);
	cudaFree(dh);
	cudaFree(ds);
#else
	query<<<1, 1>>>(gpu.get_scene_data(), gr, device);
#endif
	cudaError_t status = cudaDeviceSynchronize();
	cudaMemcpy(&result, device, sizeof(result), cudaMemcpyDeviceToHost);
	cudaFree(device);
	check(status == cudaSuccess, "GPU texture query kernel runs");
	HitRecord hit;
	bool found = scene.intersect(ray, hit);
	check(found == result.found && scene.intersects(ray) == result.shadow,
		  "CPU/CUDA alpha visibility agreement");

	if (!found)
		return;
	Material m = scene.evaluate(hit);
	check(near(hit.t, result.hit.t) && near(hit.tex_u, result.hit.tex_u) &&
			  near(hit.tex_v, result.hit.tex_v),
		  "CPU/CUDA textured hit coordinates");
	check(near(hit.normal.x, result.hit.normal.x) && near(hit.normal.y, result.hit.normal.y) &&
			  near(hit.normal.z, result.hit.normal.z),
		  "CPU/CUDA normal and bump evaluation");
	check(near(m.baseColor.x, result.mat.baseColor.x) &&
			  near(m.baseColor.y, result.mat.baseColor.y) &&
			  near(m.roughness, result.mat.roughness) && near(m.metallic, result.mat.metallic),
		  "CPU/CUDA color and scalar maps");
}
#endif
int main() {
	auto dir =
		std::filesystem::temp_directory_path() / ("xenon-texture-test-" + std::to_string(getpid()));
	std::filesystem::create_directories(dir);
	// Top row red, green; bottom blue, half-grey. TGA 32-bit top-origin BGRA.
	unsigned char header[18] = {};
	header[2] = 2;
	header[12] = 2;
	header[14] = 2;
	header[16] = 32;
	header[17] = 0x28;
	unsigned char pixels[] = {0, 0, 255, 255, 0, 255, 0, 255, 255, 0, 0, 0, 128, 128, 128, 128};
	{
		std::ofstream f(dir / "map.tga", std::ios::binary);
		f.write((char *)header, 18);
		f.write((char *)pixels, 16);
	}
	auto srgb = load_texture((dir / "map.tga").string(), true);
	auto linear = load_texture((dir / "map.tga").string(), false);
	check(near(srgb->sample(.25f, .75f).r, 1) && near(srgb->sample(.25f, .25f).b, 1),
		  "UV bottom origin and image top origin agree");
	check(near(srgb->sample(.75f, .25f).r, .2158605f) &&
			  near(srgb->sample(.75f, .25f).a, 128 / 255.f),
		  "sRGB decodes RGB but leaves alpha linear");
	check(near(linear->sample(.75f, .25f).r, 128 / 255.f),
		  "linear maps do not undergo gamma decoding");
	check(near(srgb->sample(-.75f, 1.75f).r, 1), "repeat wraps negative and positive UVs");
	check(near(srgb->sample(.5f, .75f).r, .5f) && near(srgb->sample(.5f, .75f).g, .5f),
		  "bilinear interpolation operates in linear light");
	float avg = (1.f + .2158605f) / 4.f;
	check(srgb->levels.size() == 2 && near(srgb->sample(.1f, .8f, 1).r, avg),
		  "mipmap average preserves linear energy");
	check(near(srgb->sample(.25f, .75f, .5f).r, (1 + avg) / 2), "fractional LOD is trilinear");
	Texture odd;
	odd.levels = {{3, 1, 0}};
	odd.pixels = {{1, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}};
	odd.generate_mips();
	check(near(odd.pixels.back().r, 1.f / 3.f), "odd-size mipmap preserves mean");

	write(dir / "seam.obj", "v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 "
							"1\nvt 2 0\nvn 0 0 1\nf 1/1/1 2/2/1 3/3/1\nf 1/5/1 3/3/1 4/4/1\n");
	TriangleMesh seam;
	check(load_obj((dir / "seam.obj").string(), seam, {}), "OBJ UV fixture loads");
	check(seam.num_vertices() == 5 && seam.tu[seam.indices[0]] == 0 &&
			  seam.tu[seam.indices[3]] == 2,
		  "OBJ vertex deduplication retains UV seams");

	write(dir / "mapped.mat", "version 2\nname mapped\ntype principled\nbaseColor 1 1 1\nroughness "
							  "1\nmetallic 1\nbaseColorTexture \"map.tga\" srgb\nroughnessTexture "
							  "\"map.tga\" linear g\nmetallicTexture \"map.tga\" linear b\n");
	Material mapped = load_material((dir / "mapped.mat").string());
	HitRecord h;
	h.tex_u = .75f;
	h.tex_v = .25f;
	h.normal = h.geo_normal = {0, 0, 1};
	h.dpdu = {1, 0, 0};
	h.dpdv = {0, 1, 0};
	Material evaluated = evaluate_material(mapped, h);
	check(near(evaluated.baseColor.x, .2158605f) && near(evaluated.roughness, 128 / 255.f) &&
			  near(evaluated.metallic, 128 / 255.f),
		  "material paths, color space and channel selectors");
	check(near(evaluated.alpha, evaluated.roughness * evaluated.roughness),
		  "texture parameters refresh derived BSDF values");
	Material normal;
	normal.normalTexture.image = constant({.5f, .8f, .9f, 1});
	material_prepare(normal);
	h.normal = {0, 0, 1};
	h.dpdv = {0, 1, 0};
	evaluate_material(normal, h);
	float positive = h.normal.y;
	h.normal = {0, 0, 1};
	h.dpdv = {0, -1, 0};
	evaluate_material(normal, h);
	check(positive > .5f && h.normal.y < -.5f && near(h.normal.z, .8f),
		  "OpenGL +Y normal maps preserve mirrored-UV handedness");
	auto ramp = std::make_shared<Texture>();
	ramp->levels = {{4, 1, 0}};
	ramp->pixels = {{0, 0, 0, 1}, {.25f, .25f, .25f, 1}, {.5f, .5f, .5f, 1}, {.75f, .75f, .75f, 1}};
	Material bump;
	bump.bumpTexture.image = ramp;
	bump.bumpScale = 1.f;
	material_prepare(bump);
	h.normal = {0, 0, 1};
	h.dpdu = {2, 0, 0};
	h.dpdv = {.5f, 1, 0};
	h.tex_u = .5f;
	h.tex_v = .3f;
	evaluate_material(bump, h);
	Vec3 expected_bump = normalize(Vec3(-.5f, .25f, 1.f));
	check((h.normal - expected_bump).length() < 1e-5f,
		  "bump gradient matches analytic slope on skewed UV basis");

	Scene scene;
	scene.materials = {mapped};
	scene.sky.intensity = 0;
	TriangleMesh mesh;
	mesh.add_vertex({-1, -1, 0});
	mesh.add_vertex({1, -1, 0});
	mesh.add_vertex({1, 1, 0});
	mesh.add_vertex({-1, 1, 0});
	mesh.set_uv(0, 0, 0);
	mesh.set_uv(1, 1, 0);
	mesh.set_uv(2, 1, 1);
	mesh.set_uv(3, 0, 1);
	mesh.add_triangle(0, 1, 2);
	mesh.add_triangle(0, 2, 3);
	mesh.compute_smooth_normals();
	scene.meshes = {mesh};
	scene.dir_lights.push_back({{0, 0, 1}, {1, 1, 1}, 2});
	scene.build_acceleration();
	Ray ray{{.4f, .2f, 1}, {0, 0, -1}};
	HitRecord hit;
	check(scene.intersect(ray, hit), "textured mesh intersects");
	check(near(hit.tex_u, .7f) && near(hit.tex_v, .6f) && near(hit.dpdu.x, 2) &&
			  near(hit.dpdv.y, 2),
		  "hit UVs and differential geometry interpolate");
#ifdef XENON_HAS_CUDA
	compare_gpu_query(scene, ray);
	scene.materials[0].bumpTexture.image = ramp;
	scene.materials[0].bumpScale = 1.f;
	compare_gpu_query(scene, Ray{{0, .2f, 1}, {0, 0, -1}});
	scene.materials[0].bumpTexture.image.reset();
#endif
	scene.materials[0].normalTexture = normal.normalTexture;
	scene.materials[0].opacityTexture = {linear, 3};
	scene.materials[0].alphaCutoff = .5f;
	scene.build_acceleration();
	check(!scene.intersect(Ray{{-.5f, -.5f, 1}, {0, 0, -1}}, hit),
		  "transparent alpha-mask texel does not block camera ray");
	check(!scene.intersects(Ray{{-.5f, -.5f, 1}, {0, 0, -1}}),
		  "transparent alpha-mask texel does not block shadow ray");
#ifdef XENON_HAS_CUDA
	compare_gpu_query(scene, Ray{{-.5f, -.5f, 1}, {0, 0, -1}});
	compare_gpu_query(scene, Ray{{-.5f, .5f, 1}, {0, 0, -1}});
#endif
	// Nonuniform scale changes tangent vectors using A and normals using inverse transpose.
	auto instances = scene.tlas.instances();
	instances[0].xform.local_to_world.cols[0] = {2, 0, 0};
	instances[0].xform.local_to_world.cols[1] = {0, 3, 0};
	instances[0].xform.world_to_local.cols[0] = {.5f, 0, 0};
	instances[0].xform.world_to_local.cols[1] = {0, 1.f / 3, 0};
	scene.tlas.build(instances);
	HitRecord scaled;
	check(scene.intersect(Ray{{-1, 1.5f, 1}, {0, 0, -1}}, scaled) && near(scaled.dpdu.x, 4) &&
			  near(scaled.dpdv.y, 6),
		  "nonuniform instance transforms propagate surface derivatives");
#ifdef XENON_HAS_CUDA
	compare_gpu_query(scene, Ray{{-1, 1.5f, 1}, {0, 0, -1}});
#endif
	scene.build_acceleration();
	Camera cam;
	cam.look_at({0, 0, 2}, {0, 0, 0}, {0, 1, 0}, 45, 1);
	WavefrontRenderer cpu(12, 12);
	cpu.set_bounces(0, 2);
	TripleSwapchain swap(12, 12);

	for (int i = 0; i < 32; ++i)
		cpu.render_frame_tiled(scene, cam, swap);
	std::vector<float> image(swap.get_read_buffer(), swap.get_read_buffer() + 12 * 12 * 3);
	double sum = 0;

	for (float v : image) {
		sum += v;

		if (!std::isfinite(v))
			++failures;
	}
	check(sum > 1, "textured normal-mapped alpha-mask fixture renders finite nonblack radiance");
#ifdef XENON_HAS_CUDA
	CudaRenderer renderer(12, 12, 2, 0
#ifdef XENON_TEST_OPTIX
						  ,
						  true
#endif
	);
	renderer.upload_scene(scene);

	for (int i = 0; i < 32; ++i)
		renderer.render_frame(scene, cam, swap);
	double error = 0;

	for (size_t i = 0; i < image.size(); ++i)
		error += std::abs(image[i] - swap.get_read_buffer()[i]);
	std::printf("Textured CPU/CUDA relative L1 = %.9g\n", error / sum);
	check(error / sum < .006, "textured CPU/CUDA linear renders agree");
#endif
	write(dir / "valid.xenon", "version 2\nconfig 8 8 1\ncamera 0 0 2 0 0 0 45\ncamera_up 0 1 "
							   "0\nlens .1 2\nmatfile \"mapped.mat\"\nmesh \"seam.obj\"\n");
	Scene loaded;
	SceneConfig cfg;
	check(load_scene((dir / "valid.xenon").string(), loaded, cam, cfg) && cfg.version == 2 &&
			  cam.get_lens_radius() == .05f,
		  "version2 scene relative paths and physical lens load");
	auto rejected = [&](std::string name, std::string text) {
		write(dir / name, text);

		try {
			load_material((dir / name).string());

			return false;
		} catch (const std::exception &) {
			return true;
		}
	};
	check(rejected("unknown.mat", "baseColor 1 1 1\nunsupported 3\n"),
		  "unknown material parameter rejected");
	check(rejected("space.mat", "normalTexture \"map.tga\" srgb\n"),
		  "normal-map sRGB misuse rejected");
	check(rejected("sss.mat", "version 2\nsubsurface .5\n"),
		  "version2 physical subsurface unsupported status explicit");
	bool bad_spec = false;

	try {
		material_from_legacy("bad", Vec3(.5f), 0, .5f, .2f, 1.5f, 0, -1);
	} catch (const std::exception &) {
		bad_spec = true;
	}
	check(bad_spec, "ignored legacy specular is rejected explicitly");
	std::filesystem::remove_all(dir);

	return failures ? 1 : 0;
}
