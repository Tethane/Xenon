#include "material/bsdf.h"
#include "scene/scene.h"
#include "cuda/gpu_scene.cuh"
#include "cuda/gpu_bsdf.cuh"
#include "cuda/cuda_utils.cuh"
#include <cstdio>

using namespace xn;

struct Result {
	GpuBSDFSample sample;
	float3 f;
	float pdf;
	bool valid;
};

__global__ void path_randoms(uint32_t *values) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= 1024)
		return;
	auto rng =
		gpu_seed_path_pcg(i * 2685821657u, i * 7919u, (i & 1) ? (42ULL | (1ULL << 63)) : 42ULL);

	for (int d = 0; d < 8; ++d)
		values[i * 8 + d] = rng.next_uint();
}

__global__ void evaluate(const GpuMaterial *materials, const float3 *directions, Result *results,
						 int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n)
		return;
	const auto &mat = materials[i / 256];
	float3 wo = directions[2 * i], wi = directions[2 * i + 1];
	GpuPCG rng = gpu_seed_pcg(i + 1357);
	Result r{};
	r.valid = gpu_bsdf_sample(wo, mat, rng, r.sample);
	r.f = gpu_bsdf_eval_for_nee(wo, wi, mat);
	r.pdf = gpu_bsdf_pdf_for_nee(wo, wi, mat);
	results[i] = r;
}

static bool near(float a, float b) {
	return std::isfinite(a) && std::isfinite(b) &&
		   std::abs(a - b) < 0.004f * std::max(1.f, std::max(std::abs(a), std::abs(b)));
}

static bool near(Vec3 a, float3 b) {
	return near(a.x, b.x) && near(a.y, b.y) && near(a.z, b.z);
}

int main() {
	uint32_t *device_randoms;
	CUDA_CHECK(cudaMalloc(&device_randoms, 1024 * 8 * sizeof(uint32_t)));
	path_randoms<<<4, 256>>>(device_randoms);
	std::vector<uint32_t> randoms(1024 * 8);
	CUDA_CHECK(cudaMemcpy(randoms.data(), device_randoms, randoms.size() * sizeof(uint32_t),
						  cudaMemcpyDeviceToHost));
	cudaFree(device_randoms);

	for (unsigned i = 0; i < 1024; ++i) {
		auto rng =
			seed_path_pcg(i * 2685821657u, i * 7919u, (i & 1) ? (42ULL | (1ULL << 63)) : 42ULL);

		for (int d = 0; d < 8; ++d)
			if (randoms[i * 8 + d] != rng.next_uint()) {
				std::fprintf(stderr, "CPU/CUDA path RNG mismatch\n");

				return 1;
			}
	}
	Scene scene;

	for (float metal : {0.f, 0.35f, 1.f})
		for (float trans : {0.f, 0.4f, 1.f})
			for (float rough : {0.f, 0.02f, 0.15f, 0.5f, 1.f})
				for (float aniso : {-0.6f, 0.f, 0.7f}) {
					Material m;
					m.baseColor = {0.7f, 0.5f, 0.9f};
					m.metallic = metal;
					m.transmission = trans;
					m.roughness = rough;
					m.anisotropy = aniso;
					m.subsurface = 0.3f;
					material_prepare(m);
					scene.materials.push_back(m);
				}

	for (float metal : {0.f, .35f, 1.f})
		for (float rough : {.1f, .5f, 1.f})
			for (float diffuse : {0.f, .7f}) {
				Material m;
				m.multipleScattering = true;
				m.baseColor = {.7f, .5f, .9f};
				m.metallic = metal;
				m.roughness = rough;
				m.diffuseRoughness = diffuse;
				material_prepare(m);
				scene.materials.push_back(m);
			}

	for (float metal : {0.f, 1.f})
		for (float rough : {0.f, .5f, 1.f})
			for (float coat_rough : {0.f, .4f, 1.f}) {
				Material m;
				m.multipleScattering = true;
				m.baseColor = {.7f, .5f, .9f};
				m.metallic = metal;
				m.roughness = rough;
				m.clearcoat = .8f;
				m.clearcoatRoughness = coat_rough;
				material_prepare(m);
				scene.materials.push_back(m);
			}
	GpuSceneHost gpu;
	gpu.upload(scene);
	int n = scene.materials.size() * 256;
	std::vector<float3> directions(2 * n);
	std::vector<Vec3> cpu_dirs(2 * n);
	PCGState rng = seed_pcg(74);

	for (int i = 0; i < 2 * n; ++i) {
		float z = 2 * rng.next_float() - 1, phi = 2 * kPi * rng.next_float();
		float r = std::sqrt(1 - z * z);
		Vec3 d{r * std::cos(phi), r * std::sin(phi), z};
		cpu_dirs[i] = d;
		directions[i] = make_f3(d.x, d.y, d.z);
	}
	float3 *dd;
	Result *dr;
	CUDA_CHECK(cudaMalloc(&dd, directions.size() * sizeof(float3)));
	CUDA_CHECK(cudaMalloc(&dr, n * sizeof(Result)));
	CUDA_CHECK(cudaMemcpy(dd, directions.data(), directions.size() * sizeof(float3),
						  cudaMemcpyHostToDevice));
	evaluate<<<(n + 255) / 256, 256>>>(gpu.get_scene_data().materials, dd, dr, n);
	CUDA_CHECK(cudaDeviceSynchronize());
	std::vector<Result> results(n);
	CUDA_CHECK(cudaMemcpy(results.data(), dr, n * sizeof(Result), cudaMemcpyDeviceToHost));
	cudaFree(dd);
	cudaFree(dr);
	int failures = 0;

	for (int i = 0; i < n; ++i) {
		const auto &mat = scene.materials[i / 256];
		const auto &gpu_result = results[i];
		auto rng = seed_pcg(i + 1357);
		BSDFSample sample;
		Vec3 wo = cpu_dirs[2 * i], wi = cpu_dirs[2 * i + 1];
		bool valid = bsdf_sample(wo, mat, rng, sample);
		bool ok = valid == gpu_result.valid && near(bsdf_eval_for_nee(wo, wi, mat), gpu_result.f) &&
				  near(bsdf_pdf_for_nee(wo, wi, mat), gpu_result.pdf);

		if (valid && gpu_result.valid)
			ok &= near(sample.wi, gpu_result.sample.wi) && near(sample.f, gpu_result.sample.f) &&
				  near(sample.pdf, gpu_result.sample.pdf) &&
				  near(sample.eta, gpu_result.sample.eta) &&
				  sample.is_delta == gpu_result.sample.delta;

		if (!ok) {
			if (failures < 8)
				std::fprintf(stderr,
							 "CPU/CUDA BSDF mismatch material=%d sample=%d pdf=%g/%g f=%g/%g\n",
							 i / 256, i, sample.pdf, gpu_result.sample.pdf, sample.f.x,
							 gpu_result.sample.f.x);
			++failures;
		}
	}
	std::printf("CUDA BSDF: %d cases, %d failures\n", n, failures);

	return failures ? 1 : 0;
}
