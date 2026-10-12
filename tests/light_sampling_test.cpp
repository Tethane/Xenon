#include "scene/scene.h"
#include <cstdio>
#ifdef XENON_HAS_CUDA
#include "cuda/gpu_scene.cuh"
#include "cuda/cuda_utils.cuh"
#endif
using namespace xn;
static int failures = 0;

static void check(bool ok, const char *message) {
	std::printf("%s: %s\n", ok ? "PASS" : "FAIL", message);
	failures += !ok;
}
#ifdef XENON_HAS_CUDA
struct Selection {
	int index, hit_index;
	float pmf;
};

__global__ void selections(GpuSceneData scene, Selection *out, int count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= count)
		return;
	float pmf;
	int index = gpu_sample_light(scene, (double(i) + .5) / count, pmf);
	const auto &light = scene.lights[index];
	GpuHitRecord hit;
	hit.instance_id = light.instance_id;
	hit.prim_id = light.tri_idx;
	out[i] = {index, gpu_light_for_hit(scene, hit), pmf};
}
#endif
int main() {
	Scene scene;

	for (int i = 0; i < 3; ++i) {
		Material m;
		m.emission = Vec3(i == 0 ? 1.f : i == 1 ? 4.f : 20.f);
		m.emissionTwoSided = i == 1;
		material_prepare(m);
		scene.materials.push_back(m);
		float scale = float(i + 1);
		TriangleMesh mesh;
		mesh.add_vertex({-scale, -scale, 0});
		mesh.add_vertex({scale, -scale, 0});
		mesh.add_vertex({0, scale, 0});
		mesh.add_triangle(0, 1, 2, i);
		scene.meshes.push_back(mesh);
	}
	scene.build_acceleration();
	constexpr double power[] = {2, 64, 360};
	bool probabilities = true, lookup = true;

	for (int i = 0; i < 3; ++i) {
		const auto &l = scene.lights[i];
		double expected = .95 * power[i] / 426. + .05 / 3.;
		probabilities &= std::abs(l.selection_pmf - expected) < 5e-8;
		HitRecord hit;
		hit.instance_id = l.instance_id;
		hit.prim_id = l.tri_idx;
		hit.pos = {0, 0, 2};
		lookup &= scene.light_for_hit(hit) == &l;
		probabilities &=
			std::abs(scene.light_pdf(hit, {0, 0, 0}, {0, 0, -1}) - expected * 4 / l.area) < 1e-7;
		probabilities &= std::abs(scene.light_pdf(hit, {0, 0, 0}, {0, 0, 1}) -
								  (i == 1 ? expected * 4 / l.area : 0.)) < 1e-7;
	}
	check(probabilities, "radiant power and world area determine PMF and solid-angle PDF");
	check(lookup, "emitter lookup preserves logical instance and primitive identity");
	constexpr int count = 65536;
	int frequency[3]{};
	double estimator = 0;

	for (int i = 0; i < count; ++i) {
		float pdf;
		const auto &l = scene.sample_light((double(i) + .5) / count, pdf);
		int index = int(&l - scene.lights.data());
		++frequency[index];
		estimator += power[index] / pdf;
	}
	bool frequencies = true;

	for (int i = 0; i < 3; ++i)
		frequencies &=
			std::abs(double(frequency[i]) / count - scene.lights[i].selection_pmf) < 2. / count;
	check(frequencies, "inverse CDF sample frequencies match independently computed PMFs");
	check(std::abs(estimator / count - 426.) < .03,
		  "power estimator converges to independent analytic sum");
#ifdef XENON_HAS_CUDA
	GpuSceneHost gpu;
	gpu.upload(scene);
	Selection *device = nullptr;
	CUDA_CHECK(cudaMalloc(&device, count * sizeof(Selection)));
	selections<<<(count + 255) / 256, 256>>>(gpu.get_scene_data(), device, count);
	CUDA_CHECK(cudaDeviceSynchronize());
	std::vector<Selection> out(count);
	CUDA_CHECK(cudaMemcpy(out.data(), device, count * sizeof(Selection), cudaMemcpyDeviceToHost));
	cudaFree(device);
	bool parity = true;

	for (int i = 0; i < count; ++i) {
		float p;
		const auto &l = scene.sample_light((double(i) + .5) / count, p);
		int index = int(&l - scene.lights.data());
		parity &= out[i].index == index && out[i].hit_index == index && out[i].pmf == p;
	}
	check(parity,
		  "CUDA uploaded PMF, emitter lookup and all selection decisions exactly match CPU");
#endif
	// Very unequal many-emitter distributions must retain rare candidates even
	// near CDF=1, where float CDF storage can collapse their intervals to zero.
	Scene many;
	constexpr int emitters = 131072;
	many.lights.resize(emitters);

	for (int i = 0; i < emitters; ++i) {
		auto &l = many.lights[i];
		l.instance_id = i;
		l.tri_idx = 0;
		l.area = 1;
		l.emission = Vec3(i == 0 ? 1e20f : 1.f);
	}
	many.prepare_light_distribution();
	bool support = true;
	double previous = 0;

	for (int i = 0; i < emitters; ++i) {
		const auto &l = many.lights[i];
		float pdf;
		const auto &sampled = many.sample_light((previous + l.selection_cdf) * .5, pdf);
		support &= l.selection_cdf > previous && l.selection_pmf >= float(.05 / emitters) &&
				   &sampled == &l;
		previous = l.selection_cdf;
	}
	check(support && previous == 1.,
		  "131072 emitters with extreme contrast retain every positive CDF interval");

	return failures ? 1 : 0;
}
