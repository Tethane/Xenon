#pragma once
// cuda/gpu_bsdf.cuh — Solid-angle BSDFs matching material/bsdf.h.

#include "cuda/gpu_types.cuh"
#include "material/energy_lut.h"

namespace xn {

struct GpuBSDFSample {
	float3 wi = {}, f = {};
	float pdf = 0.f, eta = 1.f;
	bool delta = false;
	int lobe = 0;
};

namespace gpu_bsdf {

enum {
	LOBE_DIFFUSE,
	LOBE_MICROFACET_REFL,
	LOBE_MICROFACET_TRANS,
	LOBE_DELTA_REFL,
	LOBE_DELTA_TRANS,
	LOBE_DIFFUSE_SUBSURFACE
};

__device__ inline float clamp(float x, float lo, float hi) {
	return fminf(fmaxf(x, lo), hi);
}

__device__ inline void swap(float &a, float &b) {
	float t = a;
	a = b;
	b = t;
}

__device__ inline float abs_cos_theta(float3 w) {
	return fabsf(w.z);
}

__device__ inline float cos_theta(float3 w) {
	return w.z;
}

__device__ inline float cos2_theta(float3 w) {
	return w.z * w.z;
}

__device__ inline float sin2_theta(float3 w) {
	return fmaxf(0.f, 1.f - cos2_theta(w));
}

__device__ inline float sin_theta(float3 w) {
	return sqrtf(sin2_theta(w));
}

__device__ inline float tan_theta(float3 w) {
	float c = cos_theta(w);

	return sin_theta(w) / c;
}

__device__ inline float tan2_theta(float3 w) {
	float c2 = cos2_theta(w);

	return sin2_theta(w) / c2;
}

__device__ inline float cos_phi(float3 w) {
	float s = sin_theta(w);

	return (s == 0.f) ? 1.f : clamp(w.x / s, -1.f, 1.f);
}

__device__ inline float sin_phi(float3 w) {
	float s = sin_theta(w);

	return (s == 0.f) ? 0.f : clamp(w.y / s, -1.f, 1.f);
}

__device__ inline float cos2_phi(float3 w) {
	float c = cos_phi(w);

	return c * c;
}

__device__ inline float sin2_phi(float3 w) {
	float s = sin_phi(w);

	return s * s;
}

__device__ inline bool same_hemisphere(float3 a, float3 b) {
	return a.z * b.z > 0.f;
}

__device__ inline float mis_weight_power2(float pdf_a, float pdf_b) {
	float scale = fmaxf(pdf_a, pdf_b);

	if (scale == 0.f)
		return 0.f;
	float a = pdf_a / scale, b = pdf_b / scale;

	return a * a / (a * a + b * b);
}

// ═════════════════════════════════════════════════════════════════════════════
// GGX / Trowbridge-Reitz NDF (isotropic and anisotropic)
// ═════════════════════════════════════════════════════════════════════════════

// Isotropic GGX NDF
__device__ inline float ggx_d(float3 wh, float alpha) {
	float d = wh.x * wh.x + wh.y * wh.y + alpha * alpha * wh.z * wh.z;

	return alpha * alpha / (kGpuPi * d * d);
}

// Anisotropic GGX NDF (Heitz 2014)
__device__ inline float ggx_d_aniso(float3 wh, float ax, float ay) {
	float x = wh.x / ax, y = wh.y / ay;
	float d = x * x + y * y + wh.z * wh.z;

	return 1.f / (kGpuPi * ax * ay * d * d);
}

// Smith G1 for GGX (isotropic)
__device__ inline float ggx_lambda(float3 w, float alpha) {
	if (w.z == 0.f)
		return kGpuInfinity;
	float t2 = (w.x * w.x + w.y * w.y) / (w.z * w.z);

	return 0.5f * (sqrtf(1.f + alpha * alpha * t2) - 1.f);
}

__device__ inline float ggx_g1(float3 w, float alpha) {
	return 1.f / (1.f + ggx_lambda(w, alpha));
}

// Smith G1 anisotropic
__device__ inline float ggx_lambda_aniso(float3 w, float ax, float ay) {
	if (w.z == 0.f)
		return kGpuInfinity;
	float x = ax * w.x, y = ay * w.y;

	return 0.5f * (sqrtf(1.f + (x * x + y * y) / (w.z * w.z)) - 1.f);
}

__device__ inline float ggx_g1_aniso(float3 w, float ax, float ay) {
	return 1.f / (1.f + ggx_lambda_aniso(w, ax, ay));
}

// Height-correlated Smith G2
__device__ inline float ggx_g2(float3 wo, float3 wi, float alpha) {
	return 1.f / (1.f + ggx_lambda(wo, alpha) + ggx_lambda(wi, alpha));
}

__device__ inline float ggx_g2_aniso(float3 wo, float3 wi, float ax, float ay) {
	return 1.f / (1.f + ggx_lambda_aniso(wo, ax, ay) + ggx_lambda_aniso(wi, ax, ay));
}

// ═════════════════════════════════════════════════════════════════════════════
// Fresnel
// ═════════════════════════════════════════════════════════════════════════════

__device__ inline float fresnel_dielectric(float cos_theta_i, float ior) {
	if (ior == 1.f)
		return 0.f;
	cos_theta_i = clamp(cos_theta_i, -1.f, 1.f);
	float eta_i = 1.f, eta_t = ior;

	if (cos_theta_i < 0.f) {
		swap(eta_i, eta_t);
		cos_theta_i = -cos_theta_i;
	}
	float sin2_i = 1.f - cos_theta_i * cos_theta_i;
	float sin2_t = (eta_i / eta_t) * (eta_i / eta_t) * sin2_i;

	if (sin2_t >= 1.f)
		return 1.f; // TIR
	float cos_t = sqrtf(fmaxf(0.f, 1.f - sin2_t));
	float r_parl =
		((eta_t * cos_theta_i) - (eta_i * cos_t)) / ((eta_t * cos_theta_i) + (eta_i * cos_t));
	float r_perp =
		((eta_i * cos_theta_i) - (eta_t * cos_t)) / ((eta_i * cos_theta_i) + (eta_t * cos_t));

	return (r_parl * r_parl + r_perp * r_perp) * 0.5f;
}

// Schlick Fresnel for conductors (RGB F0)
__device__ inline float3 fresnel_schlick(float cos_theta, float3 f0) {
	float m = clamp(1.f - cos_theta, 0.f, 1.f);
	float m2 = m * m;

	return f0 + (make_f3(1.f) - f0) * (m2 * m2 * m);
}

// Schlick Fresnel scalar
__device__ inline float fresnel_schlick_scalar(float cos_theta, float f0) {
	float m = clamp(1.f - cos_theta, 0.f, 1.f);
	float m2 = m * m;

	return f0 + (1.f - f0) * (m2 * m2 * m);
}

// ═════════════════════════════════════════════════════════════════════════════
// Sampling utilities
// ═════════════════════════════════════════════════════════════════════════════

__device__ inline float3 sample_cosine_hemisphere(float u1, float u2) {
	float r = sqrtf(u1);
	float phi = 2.0f * kGpuPi * u2;

	return {r * cosf(phi), r * sinf(phi), sqrtf(fmaxf(0.0f, 1.0f - u1))};
}

// ─────────────────────────────────────────────────────────────────────────────
// VNDF sampling (Heitz 2018, "Sampling the GGX Distribution of Visible Normals")
// Samples the visible normal distribution for anisotropic GGX.
// wo must be in local space with N=(0,0,1).
// Returns the half-vector wh in local space.
// ─────────────────────────────────────────────────────────────────────────────
__device__ inline float3 sample_vndf_ggx(float3 wo, float ax, float ay, float u1, float u2) {
	if (wo.z < 0.f)
		wo = -wo;
	// 1. Stretch wo
	float3 Vh = normalize3(make_f3(ax * wo.x, ay * wo.y, wo.z));

	// 2. Orthonormal basis around Vh
	float len2 = Vh.x * Vh.x + Vh.y * Vh.y;
	float3 T1 =
		(len2 > 1e-7f) ? make_f3(-Vh.y, Vh.x, 0.f) * (1.f / sqrtf(len2)) : make_f3(1.f, 0.f, 0.f);
	float3 T2 = cross3(Vh, T1);

	// 3. Sample point with polar coordinates (r, phi)
	float r = sqrtf(u1);
	float phi = 2.0f * kGpuPi * u2;
	float t1 = r * cosf(phi);
	float t2 = r * sinf(phi);
	float s = 0.5f * (1.0f + Vh.z);
	t2 = (1.0f - s) * sqrtf(fmaxf(0.f, 1.0f - t1 * t1)) + s * t2;

	// 4. Compute normal in stretched space
	float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

	// 5. Unstretch
	float3 wh = normalize3(make_f3(ax * Nh.x, ay * Nh.y, fmaxf(0.f, Nh.z)));

	return wh;
}

// VNDF PDF for sampling (visible normal distribution)
// This is the pdf of sampling wh via VNDF, NOT the pdf of the final wi direction.
__device__ inline float vndf_pdf(float3 wo, float3 wh, float ax, float ay) {
	float cos_o = abs_cos_theta(wo);

	if (cos_o == 0.f || dot3(wo, wh) * wo.z <= 0.f)
		return 0.f;

	return ggx_d_aniso(wh, ax, ay) * ggx_g1_aniso(wo, ax, ay) * fabsf(dot3(wo, wh)) / cos_o;
}

// ═════════════════════════════════════════════════════════════════════════════
// QUEUE 1: DIFFUSE
//   Lambert diffuse: f = albedo / π, cosine hemisphere sampling
// ═════════════════════════════════════════════════════════════════════════════

// EON rough diffuse (Portsmouth, Kutz and Hill, JCGT 2025). Its parameter is
// the micro-surface single-scattering reflectance; the compensation is reciprocal.
__device__ inline float3 rough_diffuse_eval(float3 wo, float3 wi, float3 color, float rough) {
	if (!same_hemisphere(wo, wi))
		return make_f3(0.f);

	if (rough == 0.f)
		return color * kGpuInvPi;
	float co = fabsf(wo.z), ci = fabsf(wi.z), s = wo.x * wi.x + wo.y * wi.y;
	float shape = 1.f + rough * (s > 0.f ? s / fmaxf(co, ci) : s);
	float a = 1.f / (1.f + (.5f - 2.f / (3.f * kGpuPi)) * rough);
	float average = energy::fon_average(rough);
	float3 numerator = (color * color) * average,
		   denominator = make_f3(1.f) - color * (1.f - average);
	float3 multi = make_f3(numerator.x / denominator.x, numerator.y / denominator.y,
						   numerator.z / denominator.z);
	float compensation = (1.f - energy::fon(rough, co)) * (1.f - energy::fon(rough, ci)) /
						 fmaxf(1e-7f, 1.f - average);

	return (color * (a * shape) + multi * compensation) * kGpuInvPi;
}

namespace diffuse_bsdf {

__device__ inline float3 eval(float3 wo, float3 wi, const GpuMaterial &mat) {
	if (!same_hemisphere(wo, wi))
		return make_f3(0.f);

	return rough_diffuse_eval(wo, wi, mat.baseColor, mat.diffuseRoughness) * (1.f - mat.metallic);
}

__device__ inline float pdf(float3 wo, float3 wi) {
	if (!same_hemisphere(wo, wi))
		return 0.f;

	return fabsf(wi.z) * kGpuInvPi;
}

__device__ inline bool sample(float3 wo, const GpuMaterial &mat, GpuPCG &rng, GpuBSDFSample &res) {
	float u1 = rng.next_float(), u2 = rng.next_float();
	res.wi = sample_cosine_hemisphere(u1, u2);

	if (wo.z < 0.f)
		res.wi.z = -res.wi.z;
	res.pdf = fabsf(res.wi.z) * kGpuInvPi;

	if (res.pdf < 1e-8f)
		return false;
	res.f = eval(wo, res.wi, mat);
	res.delta = false;
	res.lobe = LOBE_DIFFUSE;

	return true;
}

} // namespace diffuse_bsdf

// ═════════════════════════════════════════════════════════════════════════════
// QUEUE 2: MICROFACET REFLECTION
//   GGX/Trowbridge-Reitz with Smith G2, VNDF sampling, anisotropic support
//   Fresnel: conductor → Schlick RGB (F0), dielectric → exact Fresnel
//   Single-scattering microfacet model
// ═════════════════════════════════════════════════════════════════════════════

namespace microfacet_refl_bsdf {

__device__ inline float3 fresnel_term(float cos_h, const GpuMaterial &mat) {
	float3 metal = fresnel_schlick(fabsf(cos_h), mat.baseColor);

	return lerp3(
		make_f3(fresnel_dielectric(mat.multipleScattering ? fabsf(cos_h) : cos_h, mat.ior)), metal,
		mat.metallic);
}

__device__ inline float3 eval(float3 wo, float3 wi, const GpuMaterial &mat) {
	if (!same_hemisphere(wo, wi))
		return make_f3(0.f);
	float3 wh = normalize3(wo + wi);

	if (wh.z < 0.f)
		wh = -wh;
	float D = ggx_d_aniso(wh, mat.alpha_x, mat.alpha_y);
	float G = ggx_g2_aniso(wo, wi, mat.alpha_x, mat.alpha_y);

	return fresnel_term(dot3(wo, wh), mat) * D * G / (4.f * fabsf(wo.z * wi.z));
}

__device__ inline float pdf(float3 wo, float3 wi, const GpuMaterial &mat) {
	if (!same_hemisphere(wo, wi))
		return 0.f;
	float3 wh = normalize3(wo + wi);

	if (wh.z < 0.f)
		wh = -wh;

	return vndf_pdf(wo, wh, mat.alpha_x, mat.alpha_y) / (4.f * fabsf(dot3(wo, wh)));
}

__device__ inline bool sample(float3 wo, const GpuMaterial &mat, GpuPCG &rng, GpuBSDFSample &res) {
	if (wo.z == 0.f)
		return false;
	float u1 = rng.next_float(), u2 = rng.next_float();
	float3 wh = sample_vndf_ggx(wo, mat.alpha_x, mat.alpha_y, u1, u2);
	res.wi = reflect3(-wo, wh);

	if (!same_hemisphere(wo, res.wi))
		return false;
	res.pdf = pdf(wo, res.wi, mat);
	res.f = eval(wo, res.wi, mat);
	res.delta = false;
	res.lobe = LOBE_MICROFACET_REFL;

	return res.pdf > 0.f;
}

} // namespace microfacet_refl_bsdf

// ═════════════════════════════════════════════════════════════════════════════
// QUEUE 3: MICROFACET TRANSMISSION
//   Walter 2007 GGX BTDF, VNDF sampling, correct half-vector for transmission
//   Unified glass sampling chooses reflection or transmission at the sampled microfacet.
// ═════════════════════════════════════════════════════════════════════════════

namespace microfacet_trans_bsdf {

__device__ inline float3 eval(float3 wo, float3 wi, const GpuMaterial &mat) {
	if (wo.z == 0.f || wi.z == 0.f || same_hemisphere(wo, wi) || mat.ior == 1.f)
		return make_f3(0.f);
	float eta = wo.z > 0.f ? mat.ior : 1.f / mat.ior;
	float3 wh = normalize3(wo + eta * wi);

	if (wh.z < 0.f)
		wh = -wh;
	float co = dot3(wo, wh), ci = dot3(wi, wh);

	if (co * wo.z <= 0.f || ci * wi.z <= 0.f)
		return make_f3(0.f);
	float denom = co + eta * ci;

	if (denom == 0.f)
		return make_f3(0.f);
	float D = ggx_d_aniso(wh, mat.alpha_x, mat.alpha_y);
	float G = ggx_g2_aniso(wo, wi, mat.alpha_x, mat.alpha_y);
	float F = fresnel_dielectric(co, mat.ior);
	// Radiance transport cancels eta^2 in the transmission Jacobian.
	return mat.baseColor * ((1.f - F) * D * G * fabsf(co * ci / (wo.z * wi.z * denom * denom)));
}

__device__ inline float pdf(float3 wo, float3 wi, const GpuMaterial &mat) {
	if (wo.z == 0.f || wi.z == 0.f || same_hemisphere(wo, wi) || mat.ior == 1.f)
		return 0.f;
	float eta = wo.z > 0.f ? mat.ior : 1.f / mat.ior;
	float3 wh = normalize3(wo + eta * wi);

	if (wh.z < 0.f)
		wh = -wh;
	float co = dot3(wo, wh), ci = dot3(wi, wh);

	if (co * wo.z <= 0.f || ci * wi.z <= 0.f)
		return 0.f;
	float denom = co + eta * ci;

	if (denom == 0.f)
		return 0.f;

	return vndf_pdf(wo, wh, mat.alpha_x, mat.alpha_y) * fabsf(eta * eta * ci / (denom * denom));
}

__device__ inline bool sample(float3 wo, const GpuMaterial &mat, GpuPCG &rng, GpuBSDFSample &res) {
	if (wo.z == 0.f || mat.ior == 1.f)
		return false;
	float u1 = rng.next_float(), u2 = rng.next_float();
	float3 wh = sample_vndf_ggx(wo, mat.alpha_x, mat.alpha_y, u1, u2);
	res.eta = wo.z > 0.f ? mat.ior : 1.f / mat.ior;

	if (!refract3(-wo, wo.z > 0.f ? wh : -wh, 1.f / res.eta, res.wi))
		return false;
	res.wi = normalize3(res.wi);

	if (same_hemisphere(wo, res.wi))
		return false;
	res.pdf = pdf(wo, res.wi, mat);
	res.f = eval(wo, res.wi, mat);
	res.delta = false;
	res.lobe = LOBE_MICROFACET_TRANS;

	return res.pdf > 0.f;
}

} // namespace microfacet_trans_bsdf

// ═════════════════════════════════════════════════════════════════════════════
// QUEUE 4: DELTA REFLECTION
//   Perfect mirror. pdf = 1 (discrete event), eval = 0 for continuous measure.
//   sample() returns f = F / |cos_theta_i|.
// ═════════════════════════════════════════════════════════════════════════════

namespace delta_refl_bsdf {

// eval() always returns 0 for delta distributions (continuous measure)
__device__ inline float3 eval(float3 /*wo*/, float3 /*wi*/, const GpuMaterial & /*mat*/) {
	return make_f3(0.f);
}

__device__ inline float pdf(float3 /*wo*/, float3 /*wi*/, const GpuMaterial & /*mat*/) {
	return 0.f; // delta distribution — pdf is zero for any specific direction
}

__device__ inline bool sample(float3 wo, const GpuMaterial &mat, GpuPCG & /*rng*/,
							  GpuBSDFSample &res) {
	if (fabsf(wo.z) < 1e-8f)
		return false;

	float3 n = make_f3(0.f, 0.f, (wo.z > 0.f) ? 1.f : -1.f);
	res.wi = reflect3(-wo, n);

	// Fresnel term
	float3 F_val = microfacet_refl_bsdf::fresnel_term(wo.z, mat);

	// For delta: throughput = F / cos (cos cancels in rendering equation)
	float cos_i = fabsf(res.wi.z);

	if (cos_i < 1e-8f)
		return false;

	res.f = F_val / cos_i;
	res.pdf = 1.f;
	res.delta = true;
	res.lobe = LOBE_DELTA_REFL;

	return true;
}

} // namespace delta_refl_bsdf

// ═════════════════════════════════════════════════════════════════════════════
// QUEUE 5: DELTA TRANSMISSION
//   Perfect refraction (Snell's law). TIR → return invalid.
//   Unified sampling selects reflection with probability one under TIR.
// ═════════════════════════════════════════════════════════════════════════════

namespace delta_trans_bsdf {

__device__ inline float3 eval(float3 /*wo*/, float3 /*wi*/, const GpuMaterial & /*mat*/) {
	return make_f3(0.f);
}

__device__ inline float pdf(float3 /*wo*/, float3 /*wi*/, const GpuMaterial & /*mat*/) {
	return 0.f;
}

__device__ inline bool sample(float3 wo, const GpuMaterial &mat, GpuPCG & /*rng*/,
							  GpuBSDFSample &res) {
	if (fabsf(wo.z) < 1e-8f)
		return false;

	float eta = (wo.z > 0.f) ? (1.f / mat.ior) : mat.ior;
	float3 n = make_f3(0.f, 0.f, (wo.z > 0.f) ? 1.f : -1.f);

	// Snell's law refraction
	if (!refract3(-wo, n, eta, res.wi)) {
		return false; // TIR
	}
	res.wi = normalize3(res.wi);

	float F = fresnel_dielectric(wo.z, mat.ior);
	res.eta = 1.f / eta;
	float cos_i = fabsf(res.wi.z);

	if (cos_i < 1e-8f)
		return false;

	// Throughput: (1-F) * eta^2 * baseColor / cos
	// The eta^2 accounts for the solid angle compression across interface
	res.f = mat.baseColor * ((1.f - F) * eta * eta / cos_i);
	res.pdf = 1.f;
	res.delta = true;
	res.lobe = LOBE_DELTA_TRANS;

	return true;
}

} // namespace delta_trans_bsdf

// ═════════════════════════════════════════════════════════════════════════════
// QUEUE 6: DIFFUSE SUBSURFACE (Christensen-Burley BSSRDF approximation)
//   Approximates subsurface scattering as a modified diffuse lobe.
//   Uses the Burley diffusion profile with mean free path.
//   TODO: Full random-walk subsurface for higher quality.
// ═════════════════════════════════════════════════════════════════════════════

namespace subsurface_bsdf {

// Christensen-Burley diffusion profile Rd(r)
// Rd(r) = (A * exp(-r/d) + exp(-r/(3d))) / (8 * pi * d * r)
// where d = mean free path, A ≈ 1.25 - 0.25*(F_dr)
// For the queue, we approximate as a weighted diffuse with subsurface color blend.
// This is a "diffuse-like" approximation — physically plausible, not exact.

__device__ inline float3 eval(float3 wo, float3 wi, const GpuMaterial &mat) {
	if (!same_hemisphere(wo, wi))
		return make_f3(0.f);
	// Blend between surface diffuse and subsurface color based on subsurface weight
	float3 color = lerp3(mat.baseColor, mat.subsurfaceColor, mat.subsurface);

	return color * (1.f - mat.metallic) * kGpuInvPi;
}

__device__ inline float pdf(float3 wo, float3 wi) {
	if (!same_hemisphere(wo, wi))
		return 0.f;

	return fabsf(wi.z) * kGpuInvPi;
}

__device__ inline bool sample(float3 wo, const GpuMaterial &mat, GpuPCG &rng, GpuBSDFSample &res) {
	float u1 = rng.next_float(), u2 = rng.next_float();
	res.wi = sample_cosine_hemisphere(u1, u2);

	if (wo.z < 0.f)
		res.wi.z = -res.wi.z;
	res.pdf = fabsf(res.wi.z) * kGpuInvPi;

	if (res.pdf < 1e-8f)
		return false;

	float3 color = lerp3(mat.baseColor, mat.subsurfaceColor, mat.subsurface);
	res.f = color * (1.f - mat.metallic) * kGpuInvPi;
	res.delta = false;
	res.lobe = LOBE_DIFFUSE_SUBSURFACE;

	return true;
}

} // namespace subsurface_bsdf

// ═════════════════════════════════════════════════════════════════════════════
// Unified BSDF evaluation for NEE
//
// Evaluate the complete continuous mixture for NEE and MIS.
// For delta lobes, eval = 0 and pdf = 0 (by convention).
// ═════════════════════════════════════════════════════════════════════════════

__device__ inline float bsdf_diffuse_weight(const GpuMaterial &mat) {
	return (1.f - mat.metallic) * (1.f - mat.transmission);
}

__device__ inline float bsdf_specular_probability(float3 wo, const GpuMaterial &mat) {
	float F = fresnel_dielectric(mat.multipleScattering ? fabsf(wo.z) : wo.z, mat.ior);

	return mat.metallic + (1.f - mat.metallic) * (mat.transmission + (1.f - mat.transmission) * F);
}

__device__ inline bool bsdf_smooth(const GpuMaterial &mat) {
	return fmaxf(mat.alpha_x, mat.alpha_y) < .001f || (mat.ior == 1.f && mat.metallic == 0.f);
}

__device__ inline float3 bsdf_multiple_fresnel(const GpuMaterial &mat) {
	float average = energy::ggx_average(mat.roughness);
	float fd = energy::multiple_fresnel(energy::dielectric_average(mat.ior), average);
	float3 fm;

	for (int i = 0; i < 3; ++i)
		(&fm.x)[i] = energy::multiple_fresnel(
			(&mat.baseColor.x)[i] + (1.f - (&mat.baseColor.x)[i]) / 21.f, average);

	return lerp3(make_f3(fd), fm, mat.metallic);
}

__device__ inline float bsdf_dielectric_energy(float3 wo, const GpuMaterial &mat) {
	if (bsdf_smooth(mat))
		return fresnel_dielectric(fabsf(wo.z), mat.ior);
	float single = energy::dielectric(mat.roughness, fabsf(wo.z), mat.ior);
	float multiple = energy::multiple_fresnel(energy::dielectric_average(mat.ior),
											  energy::ggx_average(mat.roughness));

	return clamp(single + (1.f - energy::ggx(mat.roughness, fabsf(wo.z))) * multiple, 0.f, 1.f);
}

__device__ inline float bsdf_dielectric_average(const GpuMaterial &mat) {
	float unit = energy::ggx_average(mat.roughness);
	float ms = energy::multiple_fresnel(energy::dielectric_average(mat.ior), unit);

	return energy::dielectric_ggx_average(mat.roughness, mat.ior) + (1.f - unit) * ms;
}

__device__ inline float bsdf_multiple_probability(float3 wo, const GpuMaterial &mat) {
	return mat.multipleScattering && !bsdf_smooth(mat)
			   ? .5f * (1.f - energy::ggx(mat.roughness, fabsf(wo.z)))
			   : 0.f;
}

__device__ inline float3 bsdf_uncoated_eval(float3 wo, float3 wi, const GpuMaterial &mat) {
	if (wo.z == 0.f || wi.z == 0.f)
		return make_f3(0.f);
	float3 result = make_f3(0.f);

	if (same_hemisphere(wo, wi)) {
		if (!bsdf_smooth(mat))
			result = microfacet_refl_bsdf::eval(wo, wi, mat);
		float Fo = fresnel_dielectric(wo.z, mat.ior);
		float Fi = fresnel_dielectric(wi.z, mat.ior);
		float3 color = lerp3(mat.baseColor, mat.subsurfaceColor, mat.subsurface);

		if (mat.multipleScattering) {
			if (!bsdf_smooth(mat)) {
				float shape = (1.f - energy::ggx(mat.roughness, fabsf(wo.z))) *
							  (1.f - energy::ggx(mat.roughness, fabsf(wi.z))) /
							  fmaxf(1e-7f, 1.f - energy::ggx_average(mat.roughness));
				result += bsdf_multiple_fresnel(mat) * (shape * kGpuInvPi);
			}
			float po = 1.f - bsdf_dielectric_energy(wo, mat),
				  pi = 1.f - bsdf_dielectric_energy(wi, mat);
			float average = fmaxf(1e-7f, 1.f - bsdf_dielectric_average(mat));
			// Reciprocal complementary Lambert lobe exactly fills the white
			// substrate's energy budget. Blend a bounded, symmetrically
			// attenuated EON term for independent rough diffuse response.
			float3 diffuse =
				color * ((1.f - mat.diffuseRoughness) * kGpuInvPi / average) +
				rough_diffuse_eval(wo, wi, color, mat.diffuseRoughness) * mat.diffuseRoughness;
			result += diffuse * ((1.f - mat.metallic) * po * pi);
		} else
			result += rough_diffuse_eval(wo, wi, color, mat.diffuseRoughness) *
					  (bsdf_diffuse_weight(mat) * (1.f - Fo) * (1.f - Fi));
	} else if (!bsdf_smooth(mat)) {
		result =
			microfacet_trans_bsdf::eval(wo, wi, mat) * ((1.f - mat.metallic) * mat.transmission);
	}

	return result;
}

__device__ inline float bsdf_base_pdf(float3 wo, float3 wi, const GpuMaterial &mat) {
	if (wo.z == 0.f || wi.z == 0.f)
		return 0.f;
	float spec = bsdf_specular_probability(wo, mat);
	float result = same_hemisphere(wo, wi) ? (1.f - spec) * fabsf(wi.z) * kGpuInvPi : 0.f;

	if (bsdf_smooth(mat))
		return result;
	float eta = wo.z > 0.f ? mat.ior : 1.f / mat.ior;
	float3 wh = normalize3(wo + (same_hemisphere(wo, wi) ? 1.f : eta) * wi);

	if (wh.z < 0.f)
		wh = -wh;
	float F = fresnel_dielectric(dot3(wo, wh), mat.ior);
	float pr = mat.metallic + (1.f - mat.metallic) * F;
	float pt = (1.f - mat.metallic) * mat.transmission * (1.f - F);

	if (pr + pt == 0.f)
		return result;

	if (same_hemisphere(wo, wi))
		result += spec * pr / (pr + pt) * microfacet_refl_bsdf::pdf(wo, wi, mat);
	else
		result += spec * pt / (pr + pt) * microfacet_trans_bsdf::pdf(wo, wi, mat);

	return result;
}

__device__ inline float bsdf_uncoated_pdf(float3 wo, float3 wi, const GpuMaterial &mat) {
	float multi = bsdf_multiple_probability(wo, mat);
	float cosine = same_hemisphere(wo, wi) ? fabsf(wi.z) * kGpuInvPi : 0.f;

	return (1.f - multi) * bsdf_base_pdf(wo, wi, mat) + multi * cosine;
}

__device__ inline bool bsdf_base_sample(float3 wo, const GpuMaterial &mat, GpuPCG &rng,
										GpuBSDFSample &res) {
	res = GpuBSDFSample{};

	if (wo.z == 0.f)
		return false;
	float spec = bsdf_specular_probability(wo, mat);

	if (rng.next_float() >= spec) {
		if (!diffuse_bsdf::sample(wo, mat, rng, res))
			return false;
	} else {
		bool smooth = bsdf_smooth(mat);
		bool delta = smooth;
		float3 wh = make_f3(0, 0, 1);

		if (!smooth) {
			float u1 = rng.next_float(), u2 = rng.next_float();
			wh = sample_vndf_ggx(wo, mat.alpha_x, mat.alpha_y, u1, u2);
		}
		float F = fresnel_dielectric(mat.multipleScattering ? fabsf(dot3(wo, wh)) : dot3(wo, wh),
									 mat.ior);
		float pr = mat.metallic + (1.f - mat.metallic) * F;
		float pt = (1.f - mat.metallic) * mat.transmission * (1.f - F);

		if (pr + pt == 0.f)
			return false;
		float p_refl = pr / (pr + pt);

		if (rng.next_float() < p_refl) {
			res.wi = reflect3(-wo, wh);

			if (!same_hemisphere(wo, res.wi))
				return false;
			res.lobe = smooth ? LOBE_DELTA_REFL : LOBE_MICROFACET_REFL;

			if (smooth) {
				res.f = microfacet_refl_bsdf::fresnel_term(wo.z, mat) / fabsf(res.wi.z);
				res.pdf = spec * p_refl;
			}
		} else {
			res.eta = wo.z > 0.f ? mat.ior : 1.f / mat.ior;

			if (!refract3(-wo, wo.z > 0.f ? wh : -wh, 1.f / res.eta, res.wi))
				return false;
			res.wi = normalize3(res.wi);

			if (same_hemisphere(wo, res.wi) || res.wi.z == 0.f)
				return false;
			res.lobe = (smooth || mat.ior == 1.f) ? LOBE_DELTA_TRANS : LOBE_MICROFACET_TRANS;
			delta = smooth || mat.ior == 1.f;

			if (delta) {
				res.f = mat.baseColor * (pt / (res.eta * res.eta * fabsf(res.wi.z)));
				res.pdf = spec * (1.f - p_refl);
			}
		}
		res.delta = delta;

		if (delta)
			return res.pdf > 0.f;
	}
	res.f = bsdf_uncoated_eval(wo, res.wi, mat);
	res.pdf = bsdf_uncoated_pdf(wo, res.wi, mat);

	return res.pdf > 0.f;
}

__device__ inline bool bsdf_uncoated_sample(float3 wo, const GpuMaterial &mat, GpuPCG &rng,
											GpuBSDFSample &res) {
	float p = bsdf_multiple_probability(wo, mat);

	if (p > 0.f && rng.next_float() < p) {
		res = GpuBSDFSample{};
		float u = rng.next_float(), v = rng.next_float();
		res.wi = sample_cosine_hemisphere(u, v);

		if (wo.z < 0.f)
			res.wi.z = -res.wi.z;
		res.f = bsdf_uncoated_eval(wo, res.wi, mat);
		res.pdf = bsdf_uncoated_pdf(wo, res.wi, mat);

		return res.pdf > 0.f;
	}

	return bsdf_base_sample(wo, mat, rng, res);
}

// Approximate isotropic clear coat using directional-albedo layer scaling.
// The top interface is GGX + reciprocal microfacet compensation. Repeated
// inter-layer bounces and refraction of the substrate lobe are not represented.
__device__ inline GpuMaterial bsdf_coat_material(const GpuMaterial &mat) {
	GpuMaterial coat{};
	coat.baseColor = make_f3(0.f);
	coat.roughness = mat.clearcoatRoughness;
	coat.ior = mat.clearcoatIor;
	coat.metallic = 0.f;
	coat.transmission = 0.f;
	coat.subsurface = 0.f;
	coat.multipleScattering = true;
	coat.alpha = fmaxf(coat.roughness * coat.roughness, 1e-6f);
	coat.alpha_x = coat.alpha_y = coat.alpha;

	return coat;
}

__device__ inline float bsdf_coat_probability(float3 wo, const GpuMaterial &mat) {
	if (mat.clearcoat == 0.f)
		return 0.f;
	GpuMaterial coat = bsdf_coat_material(mat);

	return mat.clearcoat * bsdf_dielectric_energy(wo, coat);
}

__device__ inline float3 bsdf_coat_eval(float3 wo, float3 wi, const GpuMaterial &coat) {
	if (!same_hemisphere(wo, wi) || bsdf_smooth(coat))
		return make_f3(0.f);
	float shape = (1.f - energy::ggx(coat.roughness, fabsf(wo.z))) *
				  (1.f - energy::ggx(coat.roughness, fabsf(wi.z))) /
				  fmaxf(1e-7f, 1.f - energy::ggx_average(coat.roughness));

	return microfacet_refl_bsdf::eval(wo, wi, coat) +
		   bsdf_multiple_fresnel(coat) * (shape * kGpuInvPi);
}

__device__ inline float bsdf_coat_pdf(float3 wo, float3 wi, const GpuMaterial &coat) {
	if (!same_hemisphere(wo, wi) || bsdf_smooth(coat))
		return 0.f;
	float p = bsdf_multiple_probability(wo, coat);

	return (1.f - p) * microfacet_refl_bsdf::pdf(wo, wi, coat) + p * fabsf(wi.z) * kGpuInvPi;
}

__device__ inline float3 bsdf_eval_for_nee(float3 wo, float3 wi, const GpuMaterial &mat) {
	float3 base = bsdf_uncoated_eval(wo, wi, mat);

	if (mat.clearcoat == 0.f || wo.z == 0.f || wi.z == 0.f)
		return base;
	GpuMaterial coat = bsdf_coat_material(mat);
	float eo = bsdf_dielectric_energy(wo, coat), ei = bsdf_dielectric_energy(wi, coat);
	float attenuation = (1.f - mat.clearcoat) + mat.clearcoat * (1.f - eo) * (1.f - ei);

	return base * attenuation + bsdf_coat_eval(wo, wi, coat) * mat.clearcoat;
}

__device__ inline float bsdf_pdf_for_nee(float3 wo, float3 wi, const GpuMaterial &mat) {
	float base = bsdf_uncoated_pdf(wo, wi, mat);

	if (mat.clearcoat == 0.f || wo.z == 0.f || wi.z == 0.f)
		return base;
	float p = bsdf_coat_probability(wo, mat);
	GpuMaterial coat = bsdf_coat_material(mat);

	return (1.f - p) * base + p * bsdf_coat_pdf(wo, wi, coat);
}

__device__ inline bool bsdf_sample(float3 wo, const GpuMaterial &mat, GpuPCG &rng,
								   GpuBSDFSample &res) {
	float p = bsdf_coat_probability(wo, mat);

	if (p == 0.f)
		return bsdf_uncoated_sample(wo, mat, rng, res);
	GpuMaterial coat = bsdf_coat_material(mat);

	if (rng.next_float() < p) {
		res = GpuBSDFSample{};

		if (bsdf_smooth(coat)) {
			res.wi = make_f3(-wo.x, -wo.y, wo.z);
			res.delta = true;
			res.lobe = LOBE_DELTA_REFL;
		} else {
			float q = bsdf_multiple_probability(wo, coat);
			bool cosine = q > 0.f && rng.next_float() < q;
			float u = rng.next_float(), v = rng.next_float();

			if (cosine) {
				res.wi = sample_cosine_hemisphere(u, v);

				if (wo.z < 0.f)
					res.wi.z = -res.wi.z;
			} else
				res.wi = reflect3(-wo, sample_vndf_ggx(wo, coat.alpha_x, coat.alpha_y, u, v));

			if (!same_hemisphere(wo, res.wi))
				return false;
			res.lobe = LOBE_MICROFACET_REFL;
		}
	} else if (!bsdf_uncoated_sample(wo, mat, rng, res))
		return false;

	if (res.delta) {
		// Coincident ideal reflections form one discrete event; include every
		// contributing delta lobe, independent of which component selected it.
		res.f = make_f3(0.f);
		res.pdf = 0.f;

		if (bsdf_smooth(mat)) {
			float eo = bsdf_dielectric_energy(wo, coat), ei = bsdf_dielectric_energy(res.wi, coat);
			float attenuation = (1.f - mat.clearcoat) + mat.clearcoat * (1.f - eo) * (1.f - ei);
			res.f +=
				microfacet_refl_bsdf::fresnel_term(wo.z, mat) * (attenuation / fabsf(res.wi.z));
			res.pdf += (1.f - p) * bsdf_specular_probability(wo, mat);
		}

		if (bsdf_smooth(coat)) {
			res.f += make_f3(fresnel_dielectric(fabsf(wo.z), coat.ior)) *
					 (mat.clearcoat / fabsf(res.wi.z));
			res.pdf += p;
		}
	} else {
		res.f = bsdf_eval_for_nee(wo, res.wi, mat);
		res.pdf = bsdf_pdf_for_nee(wo, res.wi, mat);
	}

	return res.pdf > 0.f;
}

} // namespace gpu_bsdf

__device__ inline bool gpu_bsdf_sample(float3 wo, const GpuMaterial &m, GpuPCG &rng,
									   GpuBSDFSample &res) {
	return gpu_bsdf::bsdf_sample(wo, m, rng, res);
}

__device__ inline float3 gpu_bsdf_eval_for_nee(float3 wo, float3 wi, const GpuMaterial &m) {
	return gpu_bsdf::bsdf_eval_for_nee(wo, wi, m);
}

__device__ inline float gpu_bsdf_pdf_for_nee(float3 wo, float3 wi, const GpuMaterial &m) {
	return gpu_bsdf::bsdf_pdf_for_nee(wo, wi, m);
}

__device__ inline float gpu_power_heuristic(float a, float b) {
	return gpu_bsdf::mis_weight_power2(a, b);
}

} // namespace xn
