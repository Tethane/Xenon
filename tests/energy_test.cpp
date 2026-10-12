#include "material/bsdf.h"
#include <cstdio>
using namespace xn;
static int failures = 0;

static void check(bool ok, const char *message) {
	std::printf("%s: %s\n", ok ? "PASS" : "FAIL", message);
	failures += !ok;
}

static Vec3 direction(float mu, float phi) {
	float s = std::sqrt(std::max(0.f, 1 - mu * mu));

	return {s * std::cos(phi), s * std::sin(phi), mu};
}

static double integrate(const Material &m, Vec3 wo, int n = 512) {
	double result = 0;

	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j) {
			Vec3 wi = direction((i + .5f) / n, 2 * kPi * (j + .5f) / n);
			result += bsdf_eval_for_nee(wo, wi, m).x * wi.z * (2 * kPi / (n * n));
		}

	return result;
}

int main() {
	check(std::abs(energy::ggx(1, 1) - (1 - std::log(2.f))) < .0005,
		  "GGX LUT agrees with analytic 1-ln(2) at maximum roughness");

	for (float rough : {.35f, .6f, 1.f})
		for (float mu : {.05f, .25f, 1.f}) {
			Material m;
			m.multipleScattering = true;
			m.metallic = 1;
			m.baseColor = Vec3(1.f);
			m.roughness = rough;
			material_prepare(m);
			double e = integrate(m, direction(mu, 0));
			std::printf("  compensated metal rough=%.2f mu=%.2f energy=%.7f\n", rough, mu, e);
			check(std::abs(e - 1) < .006,
				  "compensated GGX white metal preserves energy across roughness and view angles");
			Vec3 wo = direction(mu, .3), wi = direction(.4f, 1.7f);
			check(std::abs(bsdf_eval_for_nee(wo, wi, m).x - bsdf_eval_for_nee(wi, wo, m).x) < 1e-5,
				  "metal compensation obeys reciprocity");
			m.metallic = 0;
			material_prepare(m);
			e = integrate(m, direction(mu, 0));
			check(std::abs(bsdf_eval_for_nee(wo, wi, m).x - bsdf_eval_for_nee(wi, wo, m).x) < 1e-5,
				  "compensated dielectric base obeys reciprocity");
			std::printf("  compensated dielectric rough=%.2f mu=%.2f energy=%.7f\n", rough, mu, e);
			check(
				std::abs(e - 1) < .006,
				"directional-energy diffuse/specular accounting preserves white dielectric energy");
		}

	for (float ior : {1.f, 1.05f, 1.333f, 2.5f, 3.f}) {
		Material m;
		m.multipleScattering = true;
		m.baseColor = Vec3(1.f);
		m.roughness = .7f;
		m.ior = ior;
		material_prepare(m);
		check(std::abs(integrate(m, direction(.2f, 0)) - 1) < .008,
			  "dielectric energy LUT interpolates independent IOR values");
	}

	for (float rough : {.2f, .6f, 1.f})
		for (float mu : {.01f, .25f, 1.f}) {
			Material m;
			m.baseColor = Vec3(1.f);
			m.ior = 1;
			m.diffuseRoughness = rough;
			material_prepare(m);
			double e = integrate(m, direction(mu, 0), 1024);
			std::printf("  EON rough=%.2f mu=%.2f energy=%.7f\n", rough, mu, e);
			check(std::abs(e - 1) < .002,
				  "EON white furnace preserves energy including grazing angles");
			m.baseColor = Vec3(.5f);
			material_prepare(m);
			e = integrate(m, direction(mu, 0));
			check(e >= 0 && e <= .501, "EON colored rough diffuse respects absorption");
			Vec3 wo = direction(mu, .3), wi = direction(.4f, 1.7f);
			check(std::abs(rough_diffuse_eval(wo, wi, m.baseColor, rough).x -
						   rough_diffuse_eval(wi, wo, m.baseColor, rough).x) < 1e-5,
				  "EON reciprocity");
		}

	for (float metal : {0.f, .4f, 1.f}) {
		Material m;
		m.multipleScattering = true;
		m.baseColor = {.8f, .5f, .2f};
		m.metallic = metal;
		m.roughness = .8f;
		m.diffuseRoughness = .6f;
		material_prepare(m);
		Vec3 wo = direction(.3f, 0);
		PCGState rng = seed_pcg(814);
		double sampled_energy = 0, mass[2] = {}, expected[2] = {};
		bool consistent = true;
		constexpr int count = 262144;

		for (int i = 0; i < count; ++i) {
			BSDFSample sample;

			if (!bsdf_sample(wo, m, rng, sample))
				continue;
			sampled_energy += sample.f.x * std::abs(sample.wi.z) / sample.pdf / count;
			mass[sample.wi.x > 0] += 1.0 / count;
			consistent &= std::abs(sample.pdf - bsdf_pdf_for_nee(wo, sample.wi, m)) < 1e-6f &&
						  std::abs(sample.f.x - bsdf_eval_for_nee(wo, sample.wi, m).x) < 1e-6f;
		}

		for (int i = 0; i < 512; ++i)
			for (int j = 0; j < 512; ++j) {
				Vec3 wi = direction((i + .5f) / 512, 2 * kPi * (j + .5f) / 512);
				expected[wi.x > 0] += bsdf_pdf_for_nee(wo, wi, m) * (2 * kPi / (512 * 512));
			}
		check(consistent, "compensation sampler returns complete f and mixture PDF");
		check(std::abs(mass[0] - expected[0]) < .005 && std::abs(mass[1] - expected[1]) < .005,
			  "sampled compensation directions match independently integrated PDF bins");
		check(std::abs(sampled_energy - integrate(m, wo)) < .005,
			  "sample estimator agrees with BRDF quadrature");
		check(sampled_energy <= 1.001 && sampled_energy >= 0,
			  "colored compensated material never creates furnace energy");
	}

	for (float rough : {0.f, .3f, 1.f})
		for (float coat_rough : {0.f, .4f, 1.f}) {
			Material m;
			m.baseColor = Vec3(1.f);
			m.metallic = 1.f;
			m.multipleScattering = true;
			m.roughness = rough;
			m.clearcoat = .7f;
			m.clearcoatRoughness = coat_rough;
			material_prepare(m);
			Vec3 wo = direction(.3f, .4f), wi = direction(.6f, 2.f);
			check(std::abs(bsdf_eval_for_nee(wo, wi, m).x - bsdf_eval_for_nee(wi, wo, m).x) < 1e-5,
				  "clearcoat over compensated metal obeys reciprocity");
			PCGState rng = seed_pcg(144);
			double energy = 0, mass[2] = {};
			bool consistent = true;

			for (int j = 0; j < 262144; ++j) {
				BSDFSample sample;

				if (!bsdf_sample(wo, m, rng, sample))
					continue;
				energy += sample.f.x * std::abs(sample.wi.z) / sample.pdf / 262144;

				if (!sample.is_delta)
					mass[sample.wi.x > 0] += 1.0 / 262144;

				if (!sample.is_delta)
					consistent &= std::abs(sample.pdf - bsdf_pdf_for_nee(wo, sample.wi, m)) < 1e-6;
			}
			std::printf("  coated metal base=%.2f coat=%.2f energy=%.7f\n", rough, coat_rough,
						energy);
			check(energy > .5 && energy <= 1.006,
				  "approximate reciprocal clearcoat is bounded in white furnace");
			check(consistent, "clearcoat samples include the complete mixed PDF");

			if (rough > 0.f && coat_rough > 0.f) {
				double expected[2] = {};

				for (int i = 0; i < 512; ++i)
					for (int j = 0; j < 512; ++j) {
						Vec3 incoming = direction((i + .5f) / 512, 2 * kPi * (j + .5f) / 512);
						expected[incoming.x > 0] +=
							bsdf_pdf_for_nee(wo, incoming, m) * (2 * kPi / (512 * 512));
					}
				check(std::abs(expected[0] - mass[0]) < .006 &&
						  std::abs(expected[1] - mass[1]) < .006,
					  "coated sampled directions match independent PDF integrals");
				check(std::abs(energy - integrate(m, wo)) < .008,
					  "coated Monte Carlo energy agrees with independent quadrature");
			}

			if (rough == 0.f && coat_rough == 0.f) {
				float f = fresnel_dielectric(wo.z, m.clearcoatIor);
				check(std::abs(energy - (1 - m.clearcoat * f * (1 - f))) < .0001,
					  "coincident delta coat/base matches analytic symmetric attenuation");
			}
		}

	for (float diffuse : {0.f, .6f, 1.f}) {
		Material m;
		m.baseColor = Vec3(1.f);
		m.multipleScattering = true;
		m.diffuseRoughness = diffuse;
		m.roughness = .7;
		m.clearcoat = .5;
		m.clearcoatRoughness = .4;
		material_prepare(m);
		Vec3 wo = direction(.2f, .3f), wi = direction(.7f, 1.3f);
		check(std::abs(bsdf_eval_for_nee(wo, wi, m).x - bsdf_eval_for_nee(wi, wo, m).x) < 1e-5,
			  "coated rough dielectric model remains reciprocal");
		check(integrate(m, wo) <= 1.006,
			  "coat and rough diffuse combination remains energy bounded");
	}

	for (int setting : {0, 1, 2}) {
		Material m;
		m.multipleScattering = true;

		if (setting == 0)
			m.anisotropy = .1;

		if (setting == 1)
			m.transmission = .5;

		if (setting == 2)
			m.ior = 4;
		bool rejected = false;

		try {
			material_prepare(m);
		} catch (const std::exception &) {
			rejected = true;
		}
		check(rejected, "unsupported compensation domain is rejected explicitly");
	}

	return failures ? 1 : 0;
}
