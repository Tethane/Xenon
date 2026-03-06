#pragma once
// material/bsdf.h — Per-queue BSDF implementations for wavefront path tracing
//
// Six dedicated BSDFs, one per queue. Each provides eval(), sample(), pdf().
// All work in local (tangent) space where N = (0,0,1).
// Classification selects exactly one queue per hit — no branching in kernels.

#include "material/material.h"
#include "camera/sampler.h"
#include "math/ray.h"
#include "math/vec3.h"
#include "render/wavefront_state.h"
#include <algorithm>
#include <cmath>

namespace xn {

struct BSDFSample {
  Vec3  wi;
  Vec3  f;
  float pdf      = 0.f;
  bool  is_delta = false;
  LobeType lobe  = LOBE_DIFFUSE;
};

// ═════════════════════════════════════════════════════════════════════════════
// Math Helpers (shared across all queues)
// ═════════════════════════════════════════════════════════════════════════════

inline float abs_cos_theta(Vec3 w) { return std::abs(w.z); }
inline float cos_theta(Vec3 w)     { return w.z; }
inline float cos2_theta(Vec3 w)    { return w.z * w.z; }
inline float sin2_theta(Vec3 w)    { return std::max(0.f, 1.f - cos2_theta(w)); }
inline float sin_theta(Vec3 w)     { return std::sqrt(sin2_theta(w)); }
inline float tan_theta(Vec3 w)     { float c = cos_theta(w); return (std::abs(c) < 1e-8f) ? 0.f : sin_theta(w) / c; }
inline float tan2_theta(Vec3 w)    { float c2 = cos2_theta(w); return (c2 < 1e-8f) ? 0.f : sin2_theta(w) / c2; }

inline float cos_phi(Vec3 w)  { float s = sin_theta(w); return (s == 0.f) ? 1.f : std::clamp(w.x / s, -1.f, 1.f); }
inline float sin_phi(Vec3 w)  { float s = sin_theta(w); return (s == 0.f) ? 0.f : std::clamp(w.y / s, -1.f, 1.f); }
inline float cos2_phi(Vec3 w) { float c = cos_phi(w); return c * c; }
inline float sin2_phi(Vec3 w) { float s = sin_phi(w); return s * s; }

inline bool same_hemisphere(Vec3 a, Vec3 b) { return a.z * b.z > 0.f; }

inline float mis_weight_power2(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    float sum = a2 + b2;
    return (sum > 0.f) ? a2 / sum : 0.f;
}

// ═════════════════════════════════════════════════════════════════════════════
// GGX / Trowbridge-Reitz NDF (isotropic and anisotropic)
// ═════════════════════════════════════════════════════════════════════════════

// Isotropic GGX NDF
inline float ggx_d(Vec3 wh, float alpha) {
    float c2 = cos2_theta(wh);
    if (c2 <= 0.f) return 0.f;
    float a2 = alpha * alpha;
    float d  = c2 * (a2 - 1.f) + 1.f;
    return a2 / (kPi * d * d);
}

// Anisotropic GGX NDF (Heitz 2014)
inline float ggx_d_aniso(Vec3 wh, float ax, float ay) {
    float c2 = cos2_theta(wh);
    if (c2 <= 0.f) return 0.f;
    float ex = cos_phi(wh) / ax;
    float ey = sin_phi(wh) / ay;
    float tan2 = (ex * ex + ey * ey) * tan2_theta(wh);
    float d = c2 * c2 * (1.f + tan2);
    return 1.f / (kPi * ax * ay * d * d);
}

// Smith G1 for GGX (isotropic)
inline float ggx_lambda(Vec3 w, float alpha) {
    float t2 = tan2_theta(w);
    if (t2 == 0.f) return 0.f;
    float a2t2 = alpha * alpha * t2;
    return (-1.f + std::sqrt(1.f + a2t2)) * 0.5f;
}

inline float ggx_g1(Vec3 w, float alpha) {
    return 1.f / (1.f + ggx_lambda(w, alpha));
}

// Smith G1 anisotropic
inline float ggx_lambda_aniso(Vec3 w, float ax, float ay) {
    float t2 = tan2_theta(w);
    if (t2 == 0.f) return 0.f;
    float cp = cos_phi(w), sp = sin_phi(w);
    float alpha2 = (cp * ax) * (cp * ax) + (sp * ay) * (sp * ay);
    return (-1.f + std::sqrt(1.f + alpha2 * t2)) * 0.5f;
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
    cos_theta_i = std::clamp(cos_theta_i, -1.f, 1.f);
    float eta_i = 1.f, eta_t = ior;
    if (cos_theta_i < 0.f) {
        std::swap(eta_i, eta_t);
        cos_theta_i = -cos_theta_i;
    }
    float sin2_i = 1.f - cos_theta_i * cos_theta_i;
    float sin2_t = (eta_i / eta_t) * (eta_i / eta_t) * sin2_i;
    if (sin2_t >= 1.f) return 1.f; // TIR
    float cos_t = std::sqrt(std::max(0.f, 1.f - sin2_t));
    float r_parl = ((eta_t * cos_theta_i) - (eta_i * cos_t)) /
                   ((eta_t * cos_theta_i) + (eta_i * cos_t));
    float r_perp = ((eta_i * cos_theta_i) - (eta_t * cos_t)) /
                   ((eta_i * cos_theta_i) + (eta_t * cos_t));
    return (r_parl * r_parl + r_perp * r_perp) * 0.5f;
}

// Schlick Fresnel for conductors (RGB F0)
inline Vec3 fresnel_schlick(float cos_theta, Vec3 f0) {
    float m  = std::clamp(1.f - cos_theta, 0.f, 1.f);
    float m2 = m * m;
    return f0 + (Vec3(1.f) - f0) * (m2 * m2 * m);
}

// Schlick Fresnel scalar
inline float fresnel_schlick_scalar(float cos_theta, float f0) {
    float m  = std::clamp(1.f - cos_theta, 0.f, 1.f);
    float m2 = m * m;
    return f0 + (1.f - f0) * (m2 * m2 * m);
}

// ═════════════════════════════════════════════════════════════════════════════
// Sampling utilities
// ═════════════════════════════════════════════════════════════════════════════

inline Vec3 sample_cosine_hemisphere(float u1, float u2) {
    float r   = std::sqrt(u1);
    float phi = 2.0f * kPi * u2;
    return {r * std::cos(phi), r * std::sin(phi),
            std::sqrt(std::max(0.0f, 1.0f - u1))};
}

// ─────────────────────────────────────────────────────────────────────────────
// VNDF sampling (Heitz 2018, "Sampling the GGX Distribution of Visible Normals")
// Samples the visible normal distribution for anisotropic GGX.
// wo must be in local space with N=(0,0,1).
// Returns the half-vector wh in local space.
// ─────────────────────────────────────────────────────────────────────────────
inline Vec3 sample_vndf_ggx(Vec3 wo, float ax, float ay, float u1, float u2) {
    // 1. Stretch wo
    Vec3 Vh = normalize(Vec3(ax * wo.x, ay * wo.y, wo.z));

    // 2. Orthonormal basis around Vh
    float len2 = Vh.x * Vh.x + Vh.y * Vh.y;
    Vec3 T1 = (len2 > 1e-7f)
        ? Vec3(-Vh.y, Vh.x, 0.f) * (1.f / std::sqrt(len2))
        : Vec3(1.f, 0.f, 0.f);
    Vec3 T2 = cross(Vh, T1);

    // 3. Sample point with polar coordinates (r, phi)
    float r   = std::sqrt(u1);
    float phi = 2.0f * kPi * u2;
    float t1  = r * std::cos(phi);
    float t2  = r * std::sin(phi);
    float s   = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * std::sqrt(std::max(0.f, 1.0f - t1 * t1)) + s * t2;

    // 4. Compute normal in stretched space
    Vec3 Nh = t1 * T1 + t2 * T2 +
              std::sqrt(std::max(0.f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    // 5. Unstretch
    Vec3 wh = normalize(Vec3(ax * Nh.x, ay * Nh.y, std::max(0.f, Nh.z)));
    return wh;
}

// VNDF PDF for sampling (visible normal distribution)
// This is the pdf of sampling wh via VNDF, NOT the pdf of the final wi direction.
inline float vndf_pdf(Vec3 wo, Vec3 wh, float ax, float ay) {
    float D     = ggx_d_aniso(wh, ax, ay);
    float G1_wo = ggx_g1_aniso(wo, ax, ay);
    float cos_o = abs_cos_theta(wo);
    if (cos_o < 1e-8f) return 0.f;
    return D * G1_wo * std::max(0.f, dot(wo, wh)) / cos_o;
}

// ═════════════════════════════════════════════════════════════════════════════
// Multi-scatter energy compensation (Kulla & Conty 2017, practical approx)
//
// For single-scatter GGX, energy is lost at high roughness. This provides
// a multiplicative correction factor Fms.
// We use the analytic fits from "Revisiting Physically Based Shading" (Karis).
// ═════════════════════════════════════════════════════════════════════════════

// Directional albedo: E(mu, alpha) — fraction of energy reflected
// Approximation from Kulla-Conty / Lazanyi-Szirmay-Kalos fit
inline float ggx_directional_albedo(float cos_theta, float alpha) {
    // Polynomial fit E(cos_theta, alpha)
    // This is a standard practical approximation
    float c = cos_theta;
    float a = alpha;
    // Approximation: E ≈ 1 - (1-c)^5 * (1 - F0_scalar)
    // But for multi-scatter, we use the complement: 1 - E
    // Simple fit from Turquin 2019
    float E = 1.0f - std::pow(1.0f - c, 5.0f * std::exp(-2.69f * a)) /
              (1.0f + 22.7f * std::pow(a, 1.5f));
    return std::clamp(E, 0.f, 1.f);
}

// Average directional albedo E_avg(alpha)
inline float ggx_avg_albedo(float alpha) {
    // E_avg = integral of E(cos_theta, alpha) * 2*cos*sin over hemisphere
    // Fit: E_avg ≈ 1 / (1 + 0.2 * alpha) approximately
    return 1.0f / (1.0f + 0.3037f * alpha + 0.1218f * alpha * alpha);
}

// Compute multi-scatter Fms compensation factor
inline Vec3 multiscatter_compensation(Vec3 F0, float cos_o, float cos_i, float alpha) {
    float E_o = ggx_directional_albedo(std::abs(cos_o), alpha);
    float E_i = ggx_directional_albedo(std::abs(cos_i), alpha);
    float E_avg = ggx_avg_albedo(alpha);

    // F_avg = integral of F_schlick * cos over hemisphere = (20*F0 + 1) / 21
    Vec3 F_avg = (Vec3(1.f) + F0 * 20.f) * (1.f / 21.f);

    // Multi-scatter compensation
    float denom = 1.f - F_avg.x * (1.f - E_avg); // use scalar approx for denominator
    denom = std::max(denom, 1e-4f);
    Vec3 Fms = F_avg * F_avg * E_avg / Vec3(denom);

    // Additive energy: (1 - E_o) * (1 - E_i) * Fms
    float complement = (1.f - E_o) * (1.f - E_i);
    return Vec3(1.f) + Fms * complement / Vec3(std::max(E_o * E_i, 1e-4f));
}

// ═════════════════════════════════════════════════════════════════════════════
// QUEUE 1: DIFFUSE
//   Lambert diffuse: f = albedo / π, cosine hemisphere sampling
// ═════════════════════════════════════════════════════════════════════════════

namespace diffuse_bsdf {

inline Vec3 eval(Vec3 /*wo*/, Vec3 wi, const Material& mat) {
    if (wi.z <= 0.f) return Vec3(0.f);
    return mat.baseColor * (1.f - mat.metallic) * kInvPi;
}

inline float pdf(Vec3 /*wo*/, Vec3 wi) {
    if (wi.z <= 0.f) return 0.f;
    return wi.z * kInvPi;
}

inline bool sample(Vec3 wo, const Material& mat, PCGState& rng, BSDFSample& res) {
    res.wi = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
    if (wo.z < 0.f) res.wi.z = -res.wi.z;
    res.pdf = std::abs(res.wi.z) * kInvPi;
    if (res.pdf < 1e-8f) return false;
    res.f = mat.baseColor * (1.f - mat.metallic) * kInvPi;
    res.is_delta = false;
    res.lobe = LOBE_DIFFUSE;
    return true;
}

} // namespace diffuse_bsdf

// ═════════════════════════════════════════════════════════════════════════════
// QUEUE 2: MICROFACET REFLECTION
//   GGX/Trowbridge-Reitz with Smith G2, VNDF sampling, anisotropic support
//   Fresnel: conductor → Schlick RGB (F0), dielectric → exact Fresnel
//   Includes multi-scatter energy compensation
// ═════════════════════════════════════════════════════════════════════════════

namespace microfacet_refl_bsdf {

// Compute Fresnel term based on material properties
inline Vec3 fresnel_term(float cos_theta_h, const Material& mat) {
    if (mat.isConductor) {
        return fresnel_schlick(std::abs(cos_theta_h), mat.F0);
    } else {
        float F = fresnel_dielectric(cos_theta_h, mat.ior);
        return Vec3(F);
    }
}

inline Vec3 eval(Vec3 wo, Vec3 wi, const Material& mat) {
    if (wo.z <= 0.f || wi.z <= 0.f) return Vec3(0.f);

    Vec3 wh = normalize(wo + wi);
    if (wh.z <= 0.f) return Vec3(0.f);

    float ax = mat.alpha_x;
    float ay = mat.alpha_y;

    float D  = ggx_d_aniso(wh, ax, ay);
    float G  = ggx_g2_aniso(wo, wi, ax, ay);
    Vec3  F  = fresnel_term(dot(wo, wh), mat);

    float denom = 4.f * wo.z * wi.z;
    if (denom < 1e-8f) return Vec3(0.f);

    Vec3 result = F * D * G / denom;

    // Multi-scatter compensation
    Vec3 Fms = multiscatter_compensation(mat.F0, wo.z, wi.z, mat.alpha);
    result *= Fms;

    return result;
}

inline float pdf(Vec3 wo, Vec3 wi, const Material& mat) {
    if (wo.z <= 0.f || wi.z <= 0.f) return 0.f;

    Vec3 wh = normalize(wo + wi);
    if (wh.z <= 0.f) return 0.f;

    float Dv = vndf_pdf(wo, wh, mat.alpha_x, mat.alpha_y);
    float jacobian = 1.f / (4.f * std::abs(dot(wo, wh)));
    return Dv * jacobian;
}

inline bool sample(Vec3 wo, const Material& mat, PCGState& rng, BSDFSample& res) {
    if (wo.z <= 0.f) return false;

    Vec3 wh = sample_vndf_ggx(wo, mat.alpha_x, mat.alpha_y,
                               rng.next_float(), rng.next_float());
    if (wh.z <= 0.f) return false;

    res.wi = reflect(-wo, wh);
    if (res.wi.z <= 0.f) return false;

    res.pdf = pdf(wo, res.wi, mat);
    if (res.pdf < 1e-8f) return false;

    res.f = eval(wo, res.wi, mat);
    res.is_delta = false;
    res.lobe = LOBE_MICROFACET_REFL;
    return true;
}

} // namespace microfacet_refl_bsdf

// ═════════════════════════════════════════════════════════════════════════════
// QUEUE 3: MICROFACET TRANSMISSION
//   Walter 2007 GGX BTDF, VNDF sampling, correct half-vector for transmission
//   Handles TIR by returning invalid (classification should prevent this)
// ═════════════════════════════════════════════════════════════════════════════

namespace microfacet_trans_bsdf {

inline Vec3 eval(Vec3 wo, Vec3 wi, const Material& mat) {
    // Transmission: wo and wi on opposite sides
    if (wo.z * wi.z > 0.f) return Vec3(0.f);

    float eta = (wo.z > 0.f) ? (1.f / mat.ior) : mat.ior;

    // Half-vector for transmission (Walter 2007)
    Vec3 wh = normalize(wo + wi * (1.f / eta));
    if (wh.z < 0.f) wh = -wh;

    float cos_o = dot(wo, wh);
    float cos_i = dot(wi, wh);

    // Must have opposite signs of dot products
    if (cos_o * cos_i > 0.f) return Vec3(0.f);

    float ax = mat.alpha_x;
    float ay = mat.alpha_y;

    float D = ggx_d_aniso(wh, ax, ay);
    float G = ggx_g2_aniso(wo, wi, ax, ay);
    float F = fresnel_dielectric(cos_o, mat.ior);

    float denom_ht = cos_o + eta * cos_i;
    if (std::abs(denom_ht) < 1e-8f) return Vec3(0.f);

    float factor = std::abs(cos_o * cos_i) * D * G * (1.f - F) /
                   (std::abs(wo.z * wi.z) * denom_ht * denom_ht);

    // Account for non-symmetry of BTDF
    factor *= (eta * eta);

    return mat.baseColor * std::abs(factor);
}

inline float pdf(Vec3 wo, Vec3 wi, const Material& mat) {
    if (wo.z * wi.z > 0.f) return 0.f;

    float eta = (wo.z > 0.f) ? (1.f / mat.ior) : mat.ior;

    Vec3 wh = normalize(wo + wi * (1.f / eta));
    if (wh.z < 0.f) wh = -wh;

    float cos_o = dot(wo, wh);
    float cos_i = dot(wi, wh);
    if (cos_o * cos_i > 0.f) return 0.f;

    float Dv = vndf_pdf(wo, wh, mat.alpha_x, mat.alpha_y);

    // Jacobian for transmission: |cos_i| / (cos_o + eta * cos_i)^2
    float denom = cos_o + eta * cos_i;
    if (std::abs(denom) < 1e-8f) return 0.f;
    float jacobian = std::abs(cos_i) / (denom * denom);

    return Dv * jacobian;
}

inline bool sample(Vec3 wo, const Material& mat, PCGState& rng, BSDFSample& res) {
    if (std::abs(wo.z) < 1e-8f) return false;

    Vec3 wh = sample_vndf_ggx(wo, mat.alpha_x, mat.alpha_y,
                               rng.next_float(), rng.next_float());
    if (wh.z < 0.f) wh = -wh;

    float eta = (wo.z > 0.f) ? (1.f / mat.ior) : mat.ior;

    // Refract through wh
    float cos_o = dot(wo, wh);
    float sin2_t = eta * eta * (1.f - cos_o * cos_o);

    // TIR check — return invalid
    if (sin2_t >= 1.f) return false;

    float cos_t = std::sqrt(std::max(0.f, 1.f - sin2_t));
    // wi points away from wh on the other side
    res.wi = eta * (-wo) + (eta * cos_o - cos_t) * wh;
    res.wi = normalize(res.wi);

    // Verify opposite hemispheres
    if (res.wi.z * wo.z > 0.f) return false;

    res.pdf = pdf(wo, res.wi, mat);
    if (res.pdf < 1e-8f) return false;

    res.f = eval(wo, res.wi, mat);
    res.is_delta = false;
    res.lobe = LOBE_MICROFACET_TRANS;
    return true;
}

} // namespace microfacet_trans_bsdf

// ═════════════════════════════════════════════════════════════════════════════
// QUEUE 4: DELTA REFLECTION
//   Perfect mirror. pdf = 1 (discrete event), eval = 0 for continuous measure.
//   sample() returns mirror direction with throughput = F / |cos_theta_i|.
// ═════════════════════════════════════════════════════════════════════════════

namespace delta_refl_bsdf {

// eval() always returns 0 for delta distributions (continuous measure)
inline Vec3 eval(Vec3 /*wo*/, Vec3 /*wi*/, const Material& /*mat*/) {
    return Vec3(0.f);
}

inline float pdf(Vec3 /*wo*/, Vec3 /*wi*/, const Material& /*mat*/) {
    return 0.f; // delta distribution — pdf is zero for any specific direction
}

inline bool sample(Vec3 wo, const Material& mat, PCGState& /*rng*/, BSDFSample& res) {
    if (std::abs(wo.z) < 1e-8f) return false;

    Vec3 n = Vec3(0.f, 0.f, (wo.z > 0.f) ? 1.f : -1.f);
    res.wi = reflect(-wo, n);

    // Fresnel term
    Vec3 F_val;
    if (mat.isConductor) {
        F_val = fresnel_schlick(std::abs(wo.z), mat.F0);
    } else {
        float F = fresnel_dielectric(wo.z, mat.ior);
        F_val = Vec3(F);
    }

    // For delta: throughput = F / cos (cos cancels in rendering equation)
    float cos_i = std::abs(res.wi.z);
    if (cos_i < 1e-8f) return false;

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
//   Classification should route TIR cases to delta_reflection.
// ═════════════════════════════════════════════════════════════════════════════

namespace delta_trans_bsdf {

inline Vec3 eval(Vec3 /*wo*/, Vec3 /*wi*/, const Material& /*mat*/) {
    return Vec3(0.f);
}

inline float pdf(Vec3 /*wo*/, Vec3 /*wi*/, const Material& /*mat*/) {
    return 0.f;
}

inline bool sample(Vec3 wo, const Material& mat, PCGState& /*rng*/, BSDFSample& res) {
    if (std::abs(wo.z) < 1e-8f) return false;

    float eta = (wo.z > 0.f) ? (1.f / mat.ior) : mat.ior;
    Vec3 n = Vec3(0.f, 0.f, (wo.z > 0.f) ? 1.f : -1.f);

    // Snell's law refraction
    if (!refract(-wo, n, eta, res.wi)) {
        return false; // TIR
    }
    res.wi = normalize(res.wi);

    float F   = fresnel_dielectric(wo.z, mat.ior);
    float cos_i = std::abs(res.wi.z);
    if (cos_i < 1e-8f) return false;

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

inline Vec3 eval(Vec3 /*wo*/, Vec3 wi, const Material& mat) {
    if (wi.z <= 0.f) return Vec3(0.f);
    // Blend between surface diffuse and subsurface color based on subsurface weight
    Vec3 color = lerp(mat.baseColor, mat.subsurfaceColor, mat.subsurface);
    return color * (1.f - mat.metallic) * kInvPi;
}

inline float pdf(Vec3 /*wo*/, Vec3 wi) {
    if (wi.z <= 0.f) return 0.f;
    return wi.z * kInvPi;
}

inline bool sample(Vec3 wo, const Material& mat, PCGState& rng, BSDFSample& res) {
    res.wi = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
    if (wo.z < 0.f) res.wi.z = -res.wi.z;
    res.pdf = std::abs(res.wi.z) * kInvPi;
    if (res.pdf < 1e-8f) return false;

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
// Given a material and a lobe type, evaluate f and pdf for an arbitrary
// (wo, wi) pair. Used by NEE to compute MIS weights.
// This dispatches to the appropriate queue's eval/pdf.
// For delta lobes, eval = 0 and pdf = 0 (by convention).
// ═════════════════════════════════════════════════════════════════════════════

inline Vec3 bsdf_eval_for_nee(Vec3 wo, Vec3 wi, const Material& mat) {
    // For NEE, we need to evaluate ALL non-delta lobes that could produce wi
    // and weight them by lobe selection probability.
    Vec3 result(0.f);

    if (mat.isDelta) return Vec3(0.f); // delta materials don't contribute to NEE

    if (mat.isConductor) {
        // Conductor: only microfacet reflection
        result = microfacet_refl_bsdf::eval(wo, wi, mat);
    } else if (mat.isTransmissive) {
        // Transmissive dielectric: both reflection and transmission
        float F = fresnel_dielectric(wo.z, mat.ior);
        if (wi.z > 0.f) {
            result = microfacet_refl_bsdf::eval(wo, wi, mat) * F;
        } else {
            result = microfacet_trans_bsdf::eval(wo, wi, mat) * (1.f - F);
        }
    } else if (mat.hasSubsurface) {
        // Subsurface: use subsurface eval
        result = subsurface_bsdf::eval(wo, wi, mat);
    } else {
        // Opaque dielectric: diffuse + specular weighted
        float spec_weight = mat.F0.x; // approximate with scalar F0
        float diff_weight = 1.f - spec_weight;
        if (wi.z > 0.f) {
            result = diffuse_bsdf::eval(wo, wi, mat) * diff_weight +
                     microfacet_refl_bsdf::eval(wo, wi, mat) * spec_weight;
        }
    }

    return result;
}

inline float bsdf_pdf_for_nee(Vec3 wo, Vec3 wi, const Material& mat) {
    if (mat.isDelta) return 0.f;

    if (mat.isConductor) {
        return microfacet_refl_bsdf::pdf(wo, wi, mat);
    } else if (mat.isTransmissive) {
        float F = fresnel_dielectric(wo.z, mat.ior);
        if (wi.z > 0.f) {
            return microfacet_refl_bsdf::pdf(wo, wi, mat) * F;
        } else {
            return microfacet_trans_bsdf::pdf(wo, wi, mat) * (1.f - F);
        }
    } else if (mat.hasSubsurface) {
        return subsurface_bsdf::pdf(wo, wi);
    } else {
        float spec_weight = mat.F0.x;
        float diff_weight = 1.f - spec_weight;
        if (wi.z > 0.f) {
            return diffuse_bsdf::pdf(wo, wi) * diff_weight +
                   microfacet_refl_bsdf::pdf(wo, wi, mat) * spec_weight;
        }
        return 0.f;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Unified BSDF sampling
//
// Selects exactly one lobe based on material properties and samples it.
// This matches the logic in Wavefront classification + shade kernels.
// ═════════════════════════════════════════════════════════════════════════════

inline bool bsdf_sample(Vec3 wo, const Material& mat, PCGState& rng, BSDFSample& res) {
    if (mat.isDelta) {
        if (mat.isTransmissive) {
            float F = fresnel_dielectric(wo.z, mat.ior);
            if (rng.next_float() < F) {
                return delta_refl_bsdf::sample(wo, mat, rng, res);
            } else {
                return delta_trans_bsdf::sample(wo, mat, rng, res);
            }
        } else {
            return delta_refl_bsdf::sample(wo, mat, rng, res);
        }
    } else if (mat.isConductor) {
        return microfacet_refl_bsdf::sample(wo, mat, rng, res);
    } else if (mat.hasSubsurface) {
        return subsurface_bsdf::sample(wo, mat, rng, res);
    } else if (mat.isTransmissive) {
        float F = fresnel_dielectric(std::abs(wo.z), mat.ior);
        if (rng.next_float() < F) {
            return microfacet_refl_bsdf::sample(wo, mat, rng, res);
        } else {
            return microfacet_trans_bsdf::sample(wo, mat, rng, res);
        }
    } else {
        float spec_weight = mat.F0.x;
        if (rng.next_float() < spec_weight) {
            return microfacet_refl_bsdf::sample(wo, mat, rng, res);
        } else {
            return diffuse_bsdf::sample(wo, mat, rng, res);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Legacy compatibility: bsdf_sample using the old PrincipledBSDF struct
// (Only used by integrator.cpp which still references PrincipledBSDF)
// ═════════════════════════════════════════════════════════════════════════════

struct PrincipledBSDF {
  Vec3  albedo = Vec3(0.5f);
  float metallic = 0.0f;
  float roughness = 0.5f;
  float ior = 1.5f;
  float specular = 0.5f;
  float transmission = 0.0f;
  float transmission_roughness = -1.0f;
};

inline Material material_from_principled(const PrincipledBSDF& p) {
    Material m;
    m.baseColor    = p.albedo;
    m.metallic     = p.metallic;
    m.roughness    = p.roughness;
    m.ior          = p.ior;
    m.transmission = p.transmission;
    material_prepare(m);
    return m;
}

} // namespace xn
