#pragma once
// material/bsdf.h — Per-queue BSDF implementations for wavefront path tracing
//
// Six dedicated BSDFs, one per queue. Each provides eval(), sample(), pdf().
// All work in local (tangent) space where N = (0,0,1).
// Unified sampling returns the full mixture f and PDF in solid angle.
// Delta samples use discrete probabilities; every path update is f * abs(wi.z) / pdf.

#include <algorithm>
#include <cmath>

#include "material/material.h"
#include "material/energy_lut.h"
#include "camera/sampler.h"
#include "math/ray.h"
#include "math/vec3.h"
#include "render/wavefront_state.h"

namespace xn {

struct BSDFSample {
	Vec3 wi;
	Vec3 f;
	float pdf = 0.f;
	float eta = 1.f; // eta_t / eta_i for radiance-mode roulette
	bool is_delta = false;
	LobeType lobe = LOBE_DIFFUSE;
};

// ═════════════════════════════════════════════════════════════════════════════
// Math Helpers (shared across all queues)
// ═════════════════════════════════════════════════════════════════════════════

inline float abs_cos_theta(Vec3 w) {
	return std::abs(w.z);
}

inline float cos_theta(Vec3 w) {
	return w.z;
}

inline float cos2_theta(Vec3 w) {
	return w.z * w.z;
}

inline float sin2_theta(Vec3 w) {
	return std::max(0.f, 1.f - cos2_theta(w));
}

inline float sin_theta(Vec3 w) {
	return std::sqrt(sin2_theta(w));
}

inline float tan_theta(Vec3 w) {
	float c = cos_theta(w);
	return sin_theta(w) / c;
}

inline float tan2_theta(Vec3 w) {
	float c2 = cos2_theta(w);
	return sin2_theta(w) / c2;
}

inline float cos_phi(Vec3 w) {
	float s = sin_theta(w);
	return (s == 0.f) ? 1.f : std::clamp(w.x / s, -1.f, 1.f);
}

inline float sin_phi(Vec3 w) {
	float s = sin_theta(w);
	return (s == 0.f) ? 0.f : std::clamp(w.y / s, -1.f, 1.f);
}

inline float cos2_phi(Vec3 w) {
	float c = cos_phi(w);
	return c * c;
}

inline float sin2_phi(Vec3 w) {
	float s = sin_phi(w);
	return s * s;
}

inline bool same_hemisphere(Vec3 a, Vec3 b) {
	return a.z * b.z > 0.f;
}

inline float mis_weight_power2(float pdf_a, float pdf_b) {
	float scale = std::max(pdf_a, pdf_b);
	if (scale == 0.f)
		return 0.f;
	float a = pdf_a / scale, b = pdf_b / scale;
	return a * a / (a * a + b * b);
}

// ═════════════════════════════════════════════════════════════════════════════
// GGX / Trowbridge-Reitz NDF (isotropic and anisotropic)
// ═════════════════════════════════════════════════════════════════════════════

// Isotropic GGX NDF
inline float ggx_d(Vec3 wh, float alpha) {
	float d = wh.x * wh.x + wh.y * wh.y + alpha * alpha * wh.z * wh.z;
	return alpha * alpha / (kPi * d * d);
}

// Anisotropic GGX NDF (Heitz 2014)
inline float ggx_d_aniso(Vec3 wh, float ax, float ay) {
	float x = wh.x / ax, y = wh.y / ay;
	float d = x * x + y * y + wh.z * wh.z;
	return 1.f / (kPi * ax * ay * d * d);
}

// Smith G1 for GGX (isotropic)
inline float ggx_lambda(Vec3 w, float alpha) {
	if (w.z == 0.f)
		return kInfinity;
	float t2 = (w.x * w.x + w.y * w.y) / (w.z * w.z);
	return 0.5f * (std::sqrt(1.f + alpha * alpha * t2) - 1.f);
}

inline float ggx_g1(Vec3 w, float alpha) {
	return 1.f / (1.f + ggx_lambda(w, alpha));
}

// Smith G1 anisotropic
inline float ggx_lambda_aniso(Vec3 w, float ax, float ay) {
	if (w.z == 0.f)
		return kInfinity;
	float x = ax * w.x, y = ay * w.y;
	return 0.5f * (std::sqrt(1.f + (x * x + y * y) / (w.z * w.z)) - 1.f);
}

inline float ggx_g1_aniso(Vec3 w, float ax, float ay) {
	return 1.f / (1.f + ggx_lambda_aniso(w, ax, ay));
}

// Height-correlated Smith G2
inline float ggx_g2(Vec3 wo, Vec3 wi, float alpha) {
	return 1.f / (1.f + ggx_lambda(wo, alpha) + ggx_lambda(wi, alpha));
}

inline float ggx_g2_aniso(Vec3 wo, Vec3 wi, float ax, float ay) {
	return 1.f / (1.f + ggx_lambda_aniso(wo, ax, ay) + ggx_lambda_aniso(wi, ax, ay));
}

// ═════════════════════════════════════════════════════════════════════════════
// Fresnel
// ═════════════════════════════════════════════════════════════════════════════

inline float fresnel_dielectric(float cos_theta_i, float ior) {
	if (ior == 1.f)
		return 0.f;
	cos_theta_i = std::clamp(cos_theta_i, -1.f, 1.f);
	float eta_i = 1.f, eta_t = ior;
	if (cos_theta_i < 0.f) {
		std::swap(eta_i, eta_t);
		cos_theta_i = -cos_theta_i;
	}
	float sin2_i = 1.f - cos_theta_i * cos_theta_i;
	float sin2_t = (eta_i / eta_t) * (eta_i / eta_t) * sin2_i;
	if (sin2_t >= 1.f)
		return 1.f; // TIR
	float cos_t = std::sqrt(std::max(0.f, 1.f - sin2_t));
	float r_parl =
		((eta_t * cos_theta_i) - (eta_i * cos_t)) / ((eta_t * cos_theta_i) + (eta_i * cos_t));
	float r_perp =
		((eta_i * cos_theta_i) - (eta_t * cos_t)) / ((eta_i * cos_theta_i) + (eta_t * cos_t));
	return (r_parl * r_parl + r_perp * r_perp) * 0.5f;
}

// Schlick Fresnel for conductors (RGB F0)
inline Vec3 fresnel_schlick(float cos_theta, Vec3 f0) {
	float m = std::clamp(1.f - cos_theta, 0.f, 1.f);
	float m2 = m * m;
	return f0 + (Vec3(1.f) - f0) * (m2 * m2 * m);
}

// Schlick Fresnel scalar
inline float fresnel_schlick_scalar(float cos_theta, float f0) {
	float m = std::clamp(1.f - cos_theta, 0.f, 1.f);
	float m2 = m * m;
	return f0 + (1.f - f0) * (m2 * m2 * m);
}

// ═════════════════════════════════════════════════════════════════════════════
// Sampling utilities
// ═════════════════════════════════════════════════════════════════════════════

inline Vec3 sample_cosine_hemisphere(float u1, float u2) {
	float r = std::sqrt(u1);
	float phi = 2.0f * kPi * u2;
	return {r * std::cos(phi), r * std::sin(phi), std::sqrt(std::max(0.0f, 1.0f - u1))};
}

// ─────────────────────────────────────────────────────────────────────────────
// VNDF sampling (Heitz 2018, "Sampling the GGX Distribution of Visible Normals")
// Samples the visible normal distribution for anisotropic GGX.
// wo must be in local space with N=(0,0,1).
// Returns the half-vector wh in local space.
// ─────────────────────────────────────────────────────────────────────────────
inline Vec3 sample_vndf_ggx(Vec3 wo, float ax, float ay, float u1, float u2) {
	if (wo.z < 0.f)
		wo = -wo;
	// 1. Stretch wo
	Vec3 Vh = normalize(Vec3(ax * wo.x, ay * wo.y, wo.z));

	// 2. Orthonormal basis around Vh
	float len2 = Vh.x * Vh.x + Vh.y * Vh.y;
	Vec3 T1 =
		(len2 > 1e-7f) ? Vec3(-Vh.y, Vh.x, 0.f) * (1.f / std::sqrt(len2)) : Vec3(1.f, 0.f, 0.f);
	Vec3 T2 = cross(Vh, T1);

	// 3. Sample point with polar coordinates (r, phi)
	float r = std::sqrt(u1);
	float phi = 2.0f * kPi * u2;
	float t1 = r * std::cos(phi);
	float t2 = r * std::sin(phi);
	float s = 0.5f * (1.0f + Vh.z);
	t2 = (1.0f - s) * std::sqrt(std::max(0.f, 1.0f - t1 * t1)) + s * t2;

	// 4. Compute normal in stretched space
	Vec3 Nh = t1 * T1 + t2 * T2 + std::sqrt(std::max(0.f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

	// 5. Unstretch
	Vec3 wh = normalize(Vec3(ax * Nh.x, ay * Nh.y, std::max(0.f, Nh.z)));
	return wh;
}

// VNDF PDF for sampling (visible normal distribution)
// This is the pdf of sampling wh via VNDF, NOT the pdf of the final wi direction.
inline float vndf_pdf(Vec3 wo, Vec3 wh, float ax, float ay) {
	float cos_o = abs_cos_theta(wo);
	if (cos_o == 0.f || dot(wo, wh) * wo.z <= 0.f)
		return 0.f;
	return ggx_d_aniso(wh, ax, ay) * ggx_g1_aniso(wo, ax, ay) * std::abs(dot(wo, wh)) / cos_o;
}

// ═════════════════════════════════════════════════════════════════════════════
// QUEUE 1: DIFFUSE
//   Lambert diffuse: f = albedo / π, cosine hemisphere sampling
// ═════════════════════════════════════════════════════════════════════════════

// EON rough diffuse (Portsmouth, Kutz and Hill, JCGT 2025). Its parameter is
// the micro-surface single-scattering reflectance; the compensation is reciprocal.
inline Vec3 rough_diffuse_eval(Vec3 wo, Vec3 wi, Vec3 color, float rough) {
	if (!same_hemisphere(wo, wi))
		return Vec3(0.f);
	if (rough == 0.f)
		return color * kInvPi;
	float co = std::abs(wo.z), ci = std::abs(wi.z), s = wo.x * wi.x + wo.y * wi.y;
	float shape = 1.f + rough * (s > 0.f ? s / std::max(co, ci) : s);
	float a = 1.f / (1.f + (.5f - 2.f / (3.f * kPi)) * rough);
	float average = energy::fon_average(rough);
	Vec3 multi = (color * color) * average / (Vec3(1.f) - color * (1.f - average));
	float compensation = (1.f - energy::fon(rough, co)) * (1.f - energy::fon(rough, ci)) /
						 std::max(1e-7f, 1.f - average);
	return (color * (a * shape) + multi * compensation) * kInvPi;
}

namespace diffuse_bsdf {

inline Vec3 eval(Vec3 wo, Vec3 wi, const Material &mat) {
	if (!same_hemisphere(wo, wi))
		return Vec3(0.f);
	return rough_diffuse_eval(wo, wi, mat.baseColor, mat.diffuseRoughness) * (1.f - mat.metallic);
}

inline float pdf(Vec3 wo, Vec3 wi) {
	if (!same_hemisphere(wo, wi))
		return 0.f;
	return std::abs(wi.z) * kInvPi;
}

inline bool sample(Vec3 wo, const Material &mat, PCGState &rng, BSDFSample &res) {
	float u1 = rng.next_float(), u2 = rng.next_float();
	res.wi = sample_cosine_hemisphere(u1, u2);
	if (wo.z < 0.f)
		res.wi.z = -res.wi.z;
	res.pdf = std::abs(res.wi.z) * kInvPi;
	if (res.pdf < 1e-8f)
		return false;
	res.f = eval(wo, res.wi, mat);
	res.is_delta = false;
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

inline Vec3 fresnel_term(float cos_h, const Material &mat) {
	Vec3 metal = fresnel_schlick(std::abs(cos_h), mat.baseColor);
	return lerp(Vec3(fresnel_dielectric(mat.multipleScattering ? std::abs(cos_h) : cos_h, mat.ior)),
				metal, mat.metallic);
}

inline Vec3 eval(Vec3 wo, Vec3 wi, const Material &mat) {
	if (!same_hemisphere(wo, wi))
		return Vec3(0.f);
	Vec3 wh = normalize(wo + wi);
	if (wh.z < 0.f)
		wh = -wh;
	float D = ggx_d_aniso(wh, mat.alpha_x, mat.alpha_y);
	float G = ggx_g2_aniso(wo, wi, mat.alpha_x, mat.alpha_y);
	return fresnel_term(dot(wo, wh), mat) * D * G / (4.f * std::abs(wo.z * wi.z));
}

inline float pdf(Vec3 wo, Vec3 wi, const Material &mat) {
	if (!same_hemisphere(wo, wi))
		return 0.f;
	Vec3 wh = normalize(wo + wi);
	if (wh.z < 0.f)
		wh = -wh;
	return vndf_pdf(wo, wh, mat.alpha_x, mat.alpha_y) / (4.f * std::abs(dot(wo, wh)));
}

inline bool sample(Vec3 wo, const Material &mat, PCGState &rng, BSDFSample &res) {
	if (wo.z == 0.f)
		return false;
	float u1 = rng.next_float(), u2 = rng.next_float();
	Vec3 wh = sample_vndf_ggx(wo, mat.alpha_x, mat.alpha_y, u1, u2);
	res.wi = reflect(-wo, wh);
	if (!same_hemisphere(wo, res.wi))
		return false;
	res.pdf = pdf(wo, res.wi, mat);
	res.f = eval(wo, res.wi, mat);
	res.is_delta = false;
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

inline Vec3 eval(Vec3 wo, Vec3 wi, const Material &mat) {
	if (wo.z == 0.f || wi.z == 0.f || same_hemisphere(wo, wi) || mat.ior == 1.f)
		return Vec3(0.f);
	float eta = wo.z > 0.f ? mat.ior : 1.f / mat.ior;
	Vec3 wh = normalize(wo + eta * wi);
	if (wh.z < 0.f)
		wh = -wh;
	float co = dot(wo, wh), ci = dot(wi, wh);
	if (co * wo.z <= 0.f || ci * wi.z <= 0.f)
		return Vec3(0.f);
	float denom = co + eta * ci;
	if (denom == 0.f)
		return Vec3(0.f);
	float D = ggx_d_aniso(wh, mat.alpha_x, mat.alpha_y);
	float G = ggx_g2_aniso(wo, wi, mat.alpha_x, mat.alpha_y);
	float F = fresnel_dielectric(co, mat.ior);
	// Radiance transport cancels eta^2 in the transmission Jacobian.
	return mat.baseColor * ((1.f - F) * D * G * std::abs(co * ci / (wo.z * wi.z * denom * denom)));
}

inline float pdf(Vec3 wo, Vec3 wi, const Material &mat) {
	if (wo.z == 0.f || wi.z == 0.f || same_hemisphere(wo, wi) || mat.ior == 1.f)
		return 0.f;
	float eta = wo.z > 0.f ? mat.ior : 1.f / mat.ior;
	Vec3 wh = normalize(wo + eta * wi);
	if (wh.z < 0.f)
		wh = -wh;
	float co = dot(wo, wh), ci = dot(wi, wh);
	if (co * wo.z <= 0.f || ci * wi.z <= 0.f)
		return 0.f;
	float denom = co + eta * ci;
	if (denom == 0.f)
		return 0.f;
	return vndf_pdf(wo, wh, mat.alpha_x, mat.alpha_y) * std::abs(eta * eta * ci / (denom * denom));
}

inline bool sample(Vec3 wo, const Material &mat, PCGState &rng, BSDFSample &res) {
	if (wo.z == 0.f || mat.ior == 1.f)
		return false;
	float u1 = rng.next_float(), u2 = rng.next_float();
	Vec3 wh = sample_vndf_ggx(wo, mat.alpha_x, mat.alpha_y, u1, u2);
	res.eta = wo.z > 0.f ? mat.ior : 1.f / mat.ior;
	if (!refract(-wo, wo.z > 0.f ? wh : -wh, 1.f / res.eta, res.wi))
		return false;
	res.wi = normalize(res.wi);
	if (same_hemisphere(wo, res.wi))
		return false;
	res.pdf = pdf(wo, res.wi, mat);
	res.f = eval(wo, res.wi, mat);
	res.is_delta = false;
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
inline Vec3 eval(Vec3 /*wo*/, Vec3 /*wi*/, const Material & /*mat*/) {
	return Vec3(0.f);
}

inline float pdf(Vec3 /*wo*/, Vec3 /*wi*/, const Material & /*mat*/) {
	return 0.f; // delta distribution — pdf is zero for any specific direction
}

inline bool sample(Vec3 wo, const Material &mat, PCGState & /*rng*/, BSDFSample &res) {
	if (std::abs(wo.z) < 1e-8f)
		return false;

	Vec3 n = Vec3(0.f, 0.f, (wo.z > 0.f) ? 1.f : -1.f);
	res.wi = reflect(-wo, n);

	// Fresnel term
	Vec3 F_val = microfacet_refl_bsdf::fresnel_term(wo.z, mat);

	// For delta: throughput = F / cos (cos cancels in rendering equation)
	float cos_i = std::abs(res.wi.z);
	if (cos_i < 1e-8f)
		return false;

	res.f = F_val / cos_i;
	res.pdf = 1.f;
	res.is_delta = true;
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

inline Vec3 eval(Vec3 /*wo*/, Vec3 /*wi*/, const Material & /*mat*/) {
	return Vec3(0.f);
}

inline float pdf(Vec3 /*wo*/, Vec3 /*wi*/, const Material & /*mat*/) {
	return 0.f;
}

inline bool sample(Vec3 wo, const Material &mat, PCGState & /*rng*/, BSDFSample &res) {
	if (std::abs(wo.z) < 1e-8f)
		return false;

	float eta = (wo.z > 0.f) ? (1.f / mat.ior) : mat.ior;
	Vec3 n = Vec3(0.f, 0.f, (wo.z > 0.f) ? 1.f : -1.f);

	// Snell's law refraction
	if (!refract(-wo, n, eta, res.wi)) {
		return false; // TIR
	}
	res.wi = normalize(res.wi);

	float F = fresnel_dielectric(wo.z, mat.ior);
	res.eta = 1.f / eta;
	float cos_i = std::abs(res.wi.z);
	if (cos_i < 1e-8f)
		return false;

	// Throughput: (1-F) * eta^2 * baseColor / cos
	// The eta^2 accounts for the solid angle compression across interface
	res.f = mat.baseColor * ((1.f - F) * eta * eta / cos_i);
	res.pdf = 1.f;
	res.is_delta = true;
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

inline Vec3 eval(Vec3 wo, Vec3 wi, const Material &mat) {
	if (!same_hemisphere(wo, wi))
		return Vec3(0.f);
	// Blend between surface diffuse and subsurface color based on subsurface weight
	Vec3 color = lerp(mat.baseColor, mat.subsurfaceColor, mat.subsurface);
	return color * (1.f - mat.metallic) * kInvPi;
}

inline float pdf(Vec3 wo, Vec3 wi) {
	if (!same_hemisphere(wo, wi))
		return 0.f;
	return std::abs(wi.z) * kInvPi;
}

inline bool sample(Vec3 wo, const Material &mat, PCGState &rng, BSDFSample &res) {
	float u1 = rng.next_float(), u2 = rng.next_float();
	res.wi = sample_cosine_hemisphere(u1, u2);
	if (wo.z < 0.f)
		res.wi.z = -res.wi.z;
	res.pdf = std::abs(res.wi.z) * kInvPi;
	if (res.pdf < 1e-8f)
		return false;

	Vec3 color = lerp(mat.baseColor, mat.subsurfaceColor, mat.subsurface);
	res.f = color * (1.f - mat.metallic) * kInvPi;
	res.is_delta = false;
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

inline float bsdf_diffuse_weight(const Material &mat) {
	return (1.f - mat.metallic) * (1.f - mat.transmission);
}

inline float bsdf_specular_probability(Vec3 wo, const Material &mat) {
	float F = fresnel_dielectric(mat.multipleScattering ? std::abs(wo.z) : wo.z, mat.ior);
	return mat.metallic + (1.f - mat.metallic) * (mat.transmission + (1.f - mat.transmission) * F);
}

inline bool bsdf_smooth(const Material &mat) {
	return std::max(mat.alpha_x, mat.alpha_y) < kDeltaAlphaThreshold ||
		   (mat.ior == 1.f && mat.metallic == 0.f);
}

inline Vec3 bsdf_multiple_fresnel(const Material &mat) {
	float average = energy::ggx_average(mat.roughness);
	float fd = energy::multiple_fresnel(energy::dielectric_average(mat.ior), average);
	Vec3 fm;
	for (int i = 0; i < 3; ++i)
		fm[i] =
			energy::multiple_fresnel(mat.baseColor[i] + (1.f - mat.baseColor[i]) / 21.f, average);
	return lerp(Vec3(fd), fm, mat.metallic);
}

inline float bsdf_dielectric_energy(Vec3 wo, const Material &mat) {
	if (bsdf_smooth(mat))
		return fresnel_dielectric(std::abs(wo.z), mat.ior);
	float single = energy::dielectric(mat.roughness, std::abs(wo.z), mat.ior);
	float multiple = energy::multiple_fresnel(energy::dielectric_average(mat.ior),
											  energy::ggx_average(mat.roughness));
	return std::clamp(single + (1.f - energy::ggx(mat.roughness, std::abs(wo.z))) * multiple, 0.f,
					  1.f);
}

inline float bsdf_dielectric_average(const Material &mat) {
	float unit = energy::ggx_average(mat.roughness);
	float ms = energy::multiple_fresnel(energy::dielectric_average(mat.ior), unit);
	return energy::dielectric_ggx_average(mat.roughness, mat.ior) + (1.f - unit) * ms;
}

inline float bsdf_multiple_probability(Vec3 wo, const Material &mat) {
	return mat.multipleScattering && !bsdf_smooth(mat)
			   ? .5f * (1.f - energy::ggx(mat.roughness, std::abs(wo.z)))
			   : 0.f;
}

inline Vec3 bsdf_uncoated_eval(Vec3 wo, Vec3 wi, const Material &mat) {
	if (wo.z == 0.f || wi.z == 0.f)
		return Vec3(0.f);
	Vec3 result(0.f);
	if (same_hemisphere(wo, wi)) {
		if (!bsdf_smooth(mat))
			result = microfacet_refl_bsdf::eval(wo, wi, mat);
		float Fo = fresnel_dielectric(wo.z, mat.ior);
		float Fi = fresnel_dielectric(wi.z, mat.ior);
		Vec3 color = lerp(mat.baseColor, mat.subsurfaceColor, mat.subsurface);
		if (mat.multipleScattering) {
			if (!bsdf_smooth(mat)) {
				float shape = (1.f - energy::ggx(mat.roughness, std::abs(wo.z))) *
							  (1.f - energy::ggx(mat.roughness, std::abs(wi.z))) /
							  std::max(1e-7f, 1.f - energy::ggx_average(mat.roughness));
				result += bsdf_multiple_fresnel(mat) * (shape * kInvPi);
			}
			float po = 1.f - bsdf_dielectric_energy(wo, mat),
				  pi = 1.f - bsdf_dielectric_energy(wi, mat);
			float average = std::max(1e-7f, 1.f - bsdf_dielectric_average(mat));
			// Reciprocal complementary Lambert lobe exactly fills the white
			// substrate's energy budget. Blend a bounded, symmetrically
			// attenuated EON term for independent rough diffuse response.
			Vec3 diffuse =
				color * ((1.f - mat.diffuseRoughness) * kInvPi / average) +
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

inline float bsdf_base_pdf(Vec3 wo, Vec3 wi, const Material &mat) {
	if (wo.z == 0.f || wi.z == 0.f)
		return 0.f;
	float spec = bsdf_specular_probability(wo, mat);
	float result = same_hemisphere(wo, wi) ? (1.f - spec) * std::abs(wi.z) * kInvPi : 0.f;
	if (bsdf_smooth(mat))
		return result;
	float eta = wo.z > 0.f ? mat.ior : 1.f / mat.ior;
	Vec3 wh = normalize(wo + (same_hemisphere(wo, wi) ? 1.f : eta) * wi);
	if (wh.z < 0.f)
		wh = -wh;
	float F = fresnel_dielectric(dot(wo, wh), mat.ior);
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

inline float bsdf_uncoated_pdf(Vec3 wo, Vec3 wi, const Material &mat) {
	float multi = bsdf_multiple_probability(wo, mat);
	float cosine = same_hemisphere(wo, wi) ? std::abs(wi.z) * kInvPi : 0.f;
	return (1.f - multi) * bsdf_base_pdf(wo, wi, mat) + multi * cosine;
}

inline bool bsdf_base_sample(Vec3 wo, const Material &mat, PCGState &rng, BSDFSample &res) {
	res = BSDFSample{};
	if (wo.z == 0.f)
		return false;
	float spec = bsdf_specular_probability(wo, mat);
	if (rng.next_float() >= spec) {
		if (!diffuse_bsdf::sample(wo, mat, rng, res))
			return false;
	} else {
		bool smooth = bsdf_smooth(mat);
		bool delta = smooth;
		Vec3 wh(0, 0, 1);
		if (!smooth) {
			float u1 = rng.next_float(), u2 = rng.next_float();
			wh = sample_vndf_ggx(wo, mat.alpha_x, mat.alpha_y, u1, u2);
		}
		float F = fresnel_dielectric(mat.multipleScattering ? std::abs(dot(wo, wh)) : dot(wo, wh),
									 mat.ior);
		float pr = mat.metallic + (1.f - mat.metallic) * F;
		float pt = (1.f - mat.metallic) * mat.transmission * (1.f - F);
		if (pr + pt == 0.f)
			return false;
		float p_refl = pr / (pr + pt);
		if (rng.next_float() < p_refl) {
			res.wi = reflect(-wo, wh);
			if (!same_hemisphere(wo, res.wi))
				return false;
			res.lobe = smooth ? LOBE_DELTA_REFL : LOBE_MICROFACET_REFL;
			if (smooth) {
				res.f = microfacet_refl_bsdf::fresnel_term(wo.z, mat) / std::abs(res.wi.z);
				res.pdf = spec * p_refl;
			}
		} else {
			res.eta = wo.z > 0.f ? mat.ior : 1.f / mat.ior;
			if (!refract(-wo, wo.z > 0.f ? wh : -wh, 1.f / res.eta, res.wi))
				return false;
			res.wi = normalize(res.wi);
			if (same_hemisphere(wo, res.wi) || res.wi.z == 0.f)
				return false;
			res.lobe = (smooth || mat.ior == 1.f) ? LOBE_DELTA_TRANS : LOBE_MICROFACET_TRANS;
			delta = smooth || mat.ior == 1.f;
			if (delta) {
				res.f = mat.baseColor * (pt / (res.eta * res.eta * std::abs(res.wi.z)));
				res.pdf = spec * (1.f - p_refl);
			}
		}
		res.is_delta = delta;
		if (delta)
			return res.pdf > 0.f;
	}
	res.f = bsdf_uncoated_eval(wo, res.wi, mat);
	res.pdf = bsdf_uncoated_pdf(wo, res.wi, mat);
	return res.pdf > 0.f;
}

inline bool bsdf_uncoated_sample(Vec3 wo, const Material &mat, PCGState &rng, BSDFSample &res) {
	float p = bsdf_multiple_probability(wo, mat);
	if (p > 0.f && rng.next_float() < p) {
		res = BSDFSample{};
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
inline Material bsdf_coat_material(const Material &mat) {
	Material coat;
	coat.baseColor = Vec3(0.f);
	coat.roughness = mat.clearcoatRoughness;
	coat.ior = mat.clearcoatIor;
	coat.metallic = 0.f;
	coat.transmission = 0.f;
	coat.subsurface = 0.f;
	coat.multipleScattering = true;
	coat.alpha = std::max(coat.roughness * coat.roughness, 1e-6f);
	coat.alpha_x = coat.alpha_y = coat.alpha;
	return coat;
}

inline float bsdf_coat_probability(Vec3 wo, const Material &mat) {
	if (mat.clearcoat == 0.f)
		return 0.f;
	Material coat = bsdf_coat_material(mat);
	return mat.clearcoat * bsdf_dielectric_energy(wo, coat);
}

inline Vec3 bsdf_coat_eval(Vec3 wo, Vec3 wi, const Material &coat) {
	if (!same_hemisphere(wo, wi) || bsdf_smooth(coat))
		return Vec3(0.f);
	float shape = (1.f - energy::ggx(coat.roughness, std::abs(wo.z))) *
				  (1.f - energy::ggx(coat.roughness, std::abs(wi.z))) /
				  std::max(1e-7f, 1.f - energy::ggx_average(coat.roughness));
	return microfacet_refl_bsdf::eval(wo, wi, coat) +
		   bsdf_multiple_fresnel(coat) * (shape * kInvPi);
}

inline float bsdf_coat_pdf(Vec3 wo, Vec3 wi, const Material &coat) {
	if (!same_hemisphere(wo, wi) || bsdf_smooth(coat))
		return 0.f;
	float p = bsdf_multiple_probability(wo, coat);
	return (1.f - p) * microfacet_refl_bsdf::pdf(wo, wi, coat) + p * std::abs(wi.z) * kInvPi;
}

inline Vec3 bsdf_eval_for_nee(Vec3 wo, Vec3 wi, const Material &mat) {
	Vec3 base = bsdf_uncoated_eval(wo, wi, mat);
	if (mat.clearcoat == 0.f || wo.z == 0.f || wi.z == 0.f)
		return base;
	Material coat = bsdf_coat_material(mat);
	float eo = bsdf_dielectric_energy(wo, coat), ei = bsdf_dielectric_energy(wi, coat);
	float attenuation = (1.f - mat.clearcoat) + mat.clearcoat * (1.f - eo) * (1.f - ei);
	return base * attenuation + bsdf_coat_eval(wo, wi, coat) * mat.clearcoat;
}

inline float bsdf_pdf_for_nee(Vec3 wo, Vec3 wi, const Material &mat) {
	float base = bsdf_uncoated_pdf(wo, wi, mat);
	if (mat.clearcoat == 0.f || wo.z == 0.f || wi.z == 0.f)
		return base;
	float p = bsdf_coat_probability(wo, mat);
	Material coat = bsdf_coat_material(mat);
	return (1.f - p) * base + p * bsdf_coat_pdf(wo, wi, coat);
}

inline bool bsdf_sample(Vec3 wo, const Material &mat, PCGState &rng, BSDFSample &res) {
	float p = bsdf_coat_probability(wo, mat);
	if (p == 0.f)
		return bsdf_uncoated_sample(wo, mat, rng, res);
	Material coat = bsdf_coat_material(mat);
	if (rng.next_float() < p) {
		res = BSDFSample{};
		if (bsdf_smooth(coat)) {
			res.wi = Vec3(-wo.x, -wo.y, wo.z);
			res.is_delta = true;
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
				res.wi = reflect(-wo, sample_vndf_ggx(wo, coat.alpha_x, coat.alpha_y, u, v));
			if (!same_hemisphere(wo, res.wi))
				return false;
			res.lobe = LOBE_MICROFACET_REFL;
		}
	} else if (!bsdf_uncoated_sample(wo, mat, rng, res))
		return false;
	if (res.is_delta) {
		// Coincident ideal reflections form one discrete event; include every
		// contributing delta lobe, independent of which component selected it.
		res.f = Vec3(0.f);
		res.pdf = 0.f;
		if (bsdf_smooth(mat)) {
			float eo = bsdf_dielectric_energy(wo, coat), ei = bsdf_dielectric_energy(res.wi, coat);
			float attenuation = (1.f - mat.clearcoat) + mat.clearcoat * (1.f - eo) * (1.f - ei);
			res.f +=
				microfacet_refl_bsdf::fresnel_term(wo.z, mat) * (attenuation / std::abs(res.wi.z));
			res.pdf += (1.f - p) * bsdf_specular_probability(wo, mat);
		}
		if (bsdf_smooth(coat)) {
			res.f += Vec3(fresnel_dielectric(std::abs(wo.z), coat.ior)) *
					 (mat.clearcoat / std::abs(res.wi.z));
			res.pdf += p;
		}
	} else {
		res.f = bsdf_eval_for_nee(wo, res.wi, mat);
		res.pdf = bsdf_pdf_for_nee(wo, res.wi, mat);
	}
	return res.pdf > 0.f;
}

// ═════════════════════════════════════════════════════════════════════════════
// Legacy compatibility: bsdf_sample using the old PrincipledBSDF struct
// (Only used by integrator.cpp which still references PrincipledBSDF)
// ═════════════════════════════════════════════════════════════════════════════

struct PrincipledBSDF {
	Vec3 albedo = Vec3(0.5f);
	float metallic = 0.0f;
	float roughness = 0.5f;
	float ior = 1.5f;
	float specular = 0.5f;
	float transmission = 0.0f;
	float transmission_roughness = -1.0f;
};

inline Material material_from_principled(const PrincipledBSDF &p) {
	if (p.specular != .5f)
		throw std::runtime_error("Legacy specular other than 0.5 is unsupported; use IOR");
	Material m;
	m.baseColor = p.albedo;
	m.metallic = p.metallic;
	m.roughness = p.transmission_roughness >= 0.f && p.transmission > .01f
					  ? p.transmission_roughness
					  : p.roughness;
	m.ior = p.ior;
	m.transmission = p.transmission;
	material_prepare(m);
	return m;
}

} // namespace xn
