#include "material/bsdf.h"
#include <cstdio>
#include <cmath>

using namespace xn;
static int failures = 0;

static void check(const char *name, bool ok) {
	std::printf("%s: %s\n", ok ? "PASS" : "FAIL", name);
	if (!ok)
		++failures;
}

static bool near(float a, float b, float eps = 1e-4f) {
	return std::abs(a - b) <= eps * std::max(1.f, std::max(std::abs(a), std::abs(b)));
}

static bool finite(Vec3 v) {
	return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

static Vec3 direction(float z, float phi) {
	float r = std::sqrt(std::max(0.f, 1.f - z * z));
	return {r * std::cos(phi), r * std::sin(phi), z};
}

static void test_distribution() {
	for (auto axes : {Vec3(0.3f, 0.3f, 0), Vec3(0.2f, 0.8f, 0), Vec3(1, 1, 0)}) {
		Vec3 wo = normalize(Vec3(1, 0, 0.05f));
		double ndf = 0, visible = 0;
		const int nz = 512, np = 512;
		for (int z = 0; z < nz; ++z)
			for (int p = 0; p < np; ++p) {
				Vec3 wh = direction((z + 0.5f) / nz, 2 * kPi * (p + 0.5f) / np);
				ndf += ggx_d_aniso(wh, axes.x, axes.y) * wh.z;
				visible += vndf_pdf(wo, wh, axes.x, axes.y);
			}
		ndf *= 2 * kPi / (nz * np);
		visible *= 2 * kPi / (nz * np);
		check("GGX projected NDF integrates to one", std::abs(ndf - 1) < 0.003);
		check("VNDF integrates to one at grazing incidence", std::abs(visible - 1) < 0.003);
	}
	check("Smith masking tends to zero at grazing",
		  ggx_g1({1, 0, 0}, 0.5f) == 0.f && ggx_g1({1, 0, 1e-6f}, 0.5f) < 1e-5f);
	check("power MIS remains finite for large PDFs", near(mis_weight_power2(1e30f, 2e30f), 0.2f));
}

static void test_material(float metallic, float transmission, float roughness, float anisotropy,
						  float subsurface, Vec3 wo) {
	Material mat;
	mat.baseColor = {0.8f, 0.6f, 0.4f};
	mat.metallic = metallic;
	mat.transmission = transmission;
	mat.roughness = roughness;
	mat.anisotropy = anisotropy;
	mat.subsurface = subsurface;
	material_prepare(mat);
	PCGState rng = seed_pcg(732);
	const int n = 160000;
	int valid = 0;
	double sampled_pdf_mass[4] = {}, sampled_energy = 0;
	bool consistent = true, finite_samples = true;
	for (int i = 0; i < n; ++i) {
		BSDFSample s;
		if (!bsdf_sample(wo, mat, rng, s))
			continue;
		++valid;
		finite_samples &=
			finite(s.wi) && finite(s.f) && std::isfinite(s.pdf) && near(s.wi.length(), 1.f);
		Vec3 f = bsdf_eval_for_nee(wo, s.wi, mat);
		float p = bsdf_pdf_for_nee(wo, s.wi, mat);
		consistent &= near(p, s.pdf) && near(f.x, s.f.x) && near(f.y, s.f.y) && near(f.z, s.f.z);
		sampled_pdf_mass[(s.wi.z > 0 ? 2 : 0) + (s.wi.x > 0 ? 1 : 0)] += 1.0 / n;
		sampled_energy += s.f.x * std::abs(s.wi.z) / s.pdf / n;
	}
	double integrated_mass[4] = {}, integrated_energy = 0;
	const int nz = 512, np = 512;
	for (int z = 0; z < nz; ++z)
		for (int p = 0; p < np; ++p) {
			Vec3 wi = direction(2.f * (z + 0.5f) / nz - 1.f, 2 * kPi * (p + 0.5f) / np);
			double dw = 4 * kPi / (nz * np);
			integrated_mass[(wi.z > 0 ? 2 : 0) + (wi.x > 0 ? 1 : 0)] +=
				bsdf_pdf_for_nee(wo, wi, mat) * dw;
			integrated_energy += bsdf_eval_for_nee(wo, wi, mat).x * std::abs(wi.z) * dw;
		}
	bool mass = true;
	for (int i = 0; i < 4; ++i)
		mass &= std::abs(sampled_pdf_mass[i] - integrated_mass[i]) < 0.012;
	std::printf(
		"  m=%.1f t=%.1f rough=%.1f aniso=%.1f wo.z=%.3f energy=%.5f integrated=%.5f valid=%d\n",
		metallic, transmission, roughness, anisotropy, wo.z, sampled_energy, integrated_energy,
		valid);
	check("sample f/PDF matches complete mixture evaluation", consistent && valid > n / 10);
	check("sampled directions match independently integrated PDF quadrants", mass);
	check("sampling matches quadrature energy",
		  std::abs(sampled_energy - integrated_energy) < 0.015);
	check("samples finite and normalized", finite_samples);
	if (transmission == 0.f)
		check("reflection conserves energy", sampled_energy <= 1.005);
}

static void test_delta() {
	Material glass;
	glass.baseColor = Vec3(1.f);
	glass.transmission = 1.f;
	glass.roughness = 0.f;
	material_prepare(glass);
	PCGState rng = seed_pcg(1234);
	for (Vec3 wo : {Vec3(0, 0, 1), normalize(Vec3(0.5f, 0, 1)), normalize(Vec3(0.2f, 0, -1)),
					normalize(Vec3(1, 0, -0.1f))}) {
		int reflected = 0;
		bool ok = true;
		float F = fresnel_dielectric(wo.z, glass.ior);
		for (int i = 0; i < 20000; ++i) {
			BSDFSample s;
			if (!bsdf_sample(wo, glass, rng, s)) {
				ok = false;
				continue;
			}
			bool refl = same_hemisphere(wo, s.wi);
			if (refl)
				++reflected;
			Vec3 weight = s.f * std::abs(s.wi.z) / s.pdf;
			float eta = wo.z > 0.f ? 1.f / glass.ior : glass.ior;
			ok &= s.is_delta && finite(weight) && near(weight.x, refl ? 1.f : eta * eta);
			if (!refl)
				ok &= near(std::sqrt(s.wi.x * s.wi.x + s.wi.y * s.wi.y),
						   eta * std::sqrt(wo.x * wo.x + wo.y * wo.y));
		}
		check("glass Fresnel event probabilities", std::abs(reflected / 20000.f - F) < 0.012f);
		check("glass radiance weights, Snell refraction and TIR", ok);
	}
	glass.ior = 1.f;
	glass.roughness = 0.8f;
	material_prepare(glass);
	BSDFSample sample;
	check("index-matched glass transmits unchanged", bsdf_sample({0, 0, 1}, glass, rng, sample) &&
														 sample.wi.z == -1.f &&
														 near(sample.f.x, 1.f));
	glass.metallic = 0.25f;
	material_prepare(glass);
	double matched_energy = 0;
	for (int i = 0; i < 100000; ++i) {
		BSDFSample s;
		if (bsdf_sample({0, 0, 1}, glass, rng, s))
			matched_energy += s.f.x * std::abs(s.wi.z) / s.pdf / 100000;
	}
	check("mixed metal and index-matched glass retains transmission",
		  matched_energy > 0.8 && matched_energy <= 1.01);
	Material metal;
	metal.baseColor = Vec3(1.f);
	metal.metallic = 1.f;
	metal.roughness = 0.f;
	material_prepare(metal);
	Vec3 wo = normalize(Vec3(1, 0, 0.01f));
	check("grazing perfect mirror weight is one",
		  bsdf_sample(wo, metal, rng, sample) &&
			  near(sample.f.x * std::abs(sample.wi.z) / sample.pdf, 1.f));
	Material matte;
	matte.roughness = 0.f;
	material_prepare(matte);
	check("smooth opaque dielectric retains diffuse NEE",
		  !matte.isDelta && bsdf_eval_for_nee({0, 0, 1}, {0, 0, 1}, matte).x > 0.f);
}

static void test_furnace() {
	Material m;
	m.baseColor = Vec3(1.f);
	m.roughness = 1.f;
	m.metallic = 1.f;
	material_prepare(m);
	PCGState rng = seed_pcg(999);
	double energy = 0;
	int valid = 0;
	for (int i = 0; i < 500000; ++i) {
		BSDFSample s;
		if (!bsdf_sample({0, 0, 1}, m, rng, s))
			continue;
		energy += s.f.x * std::abs(s.wi.z) / s.pdf / 500000;
		++valid;
	}
	check("rough GGX white furnace agrees with 1-ln(2)",
		  std::abs(energy - (1 - std::log(2.0))) < 0.003);
	check("GGX rejected samples retain probability mass",
		  std::abs(valid / 500000.f - 0.5f) < 0.004f);
	m.metallic = 0.f;
	m.ior = 1.f;
	m.baseColor = Vec3(0.7f);
	material_prepare(m);
	BSDFSample s;
	check("Lambertian weight equals albedo",
		  bsdf_sample({0, 0, 1}, m, rng, s) && near(s.f.x * s.wi.z / s.pdf, 0.7f));
}

int main() {
	test_distribution();
	for (Vec3 wo : {normalize(Vec3(0.5f, 0.1f, 1)), normalize(Vec3(1, 0, 0.1f)),
					normalize(Vec3(0.1f, 0, -1))}) {
		test_material(1, 0, 0.7f, 0.5f, 0, wo);
		test_material(0, 0, 0.7f, 0, 0, wo);
		test_material(0, 1, 0.7f, 0, 0, wo);
		test_material(0.3f, 0.4f, 0.7f, 0.2f, 0.5f, wo);
	}
	test_delta();
	test_furnace();
	std::printf("BSDF: %d failures\n", failures);
	return failures ? 1 : 0;
}
