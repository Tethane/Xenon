#pragma once
// material/bsdf.h — Principled BSDF with reflection and transmission

#include "math/vec3.h"
#include "math/ray.h"
#include "camera/sampler.h"
#include "render/wavefront_state.h"
#include <cmath>
#include <algorithm>

namespace xn {

struct PrincipledBSDF {
    Vec3  albedo    = Vec3(0.5f);
    float metallic  = 0.0f;
    float roughness = 0.5f;
    float ior       = 1.5f;
    float specular  = 0.5f;
    float transmission = 0.0f;
    float transmission_roughness = -1.0f; // if < 0, use roughness
};

struct BSDFSample {
    Vec3  wi;
    Vec3  f;
    float pdf = 0.f;
    bool  is_delta = false;
    LobeType lobe = LOBE_DIFFUSE;
};

// ─────────────────────────────────────────────────────────────────────────────
// Math Helpers
// ─────────────────────────────────────────────────────────────────────────────

inline float cos_theta(Vec3 w) { return w.z; }
inline float cos2_theta(Vec3 w) { return w.z * w.z; }
inline float sin2_theta(Vec3 w) { return std::max(0.f, 1.f - cos2_theta(w)); }
inline float sin_theta(Vec3 w) { return std::sqrt(sin2_theta(w)); }
inline float tan_theta(Vec3 w) { return sin_theta(w) / cos_theta(w); }
inline float tan2_theta(Vec3 w) { return sin2_theta(w) / cos2_theta(w); }

// ─────────────────────────────────────────────────────────────────────────────
// GGX / Microfacet
// ─────────────────────────────────────────────────────────────────────────────

inline float ggx_d(Vec3 wh, float alpha) {
    float cos_theta_h = cos_theta(wh);
    if (cos_theta_h <= 0.f) return 0.f;
    float a2 = alpha * alpha;
    float d = (cos_theta_h * cos_theta_h) * (a2 - 1.f) + 1.f;
    return a2 / (kPi * d * d);
}

inline float ggx_g1(Vec3 w, float alpha) {
    float cos_theta_v = std::abs(cos_theta(w));
    if (cos_theta_v == 0.f) return 0.f;
    float a2 = alpha * alpha;
    return 2.f * cos_theta_v / (cos_theta_v + std::sqrt(a2 + (1.f - a2) * cos_theta_v * cos_theta_v));
}

inline float ggx_g2(Vec3 wo, Vec3 wi, float alpha) {
    return ggx_g1(wo, alpha) * ggx_g1(wi, alpha);
}

inline float fresnel_dielectric(float cos_theta_i, float ior) {
    cos_theta_i = std::clamp(cos_theta_i, -1.f, 1.f);
    float eta_i = 1.f, eta_t = ior;
    if (cos_theta_i < 0.f) {
        std::swap(eta_i, eta_t);
        cos_theta_i = -cos_theta_i;
    }
    float sin_theta_i = std::sqrt(std::max(0.f, 1.f - cos_theta_i * cos_theta_i));
    float sin_theta_t = eta_i / eta_t * sin_theta_i;
    if (sin_theta_t >= 1.f) return 1.f;
    float cos_theta_t = std::sqrt(std::max(0.f, 1.f - sin_theta_t * sin_theta_t));
    float r_parl = ((eta_t * cos_theta_i) - (eta_i * cos_theta_t)) / ((eta_t * cos_theta_i) + (eta_i * cos_theta_t));
    float r_perp = ((eta_i * cos_theta_i) - (eta_t * cos_theta_t)) / ((eta_i * cos_theta_i) + (eta_t * cos_theta_t));
    return (r_parl * r_parl + r_perp * r_perp) * 0.5f;
}

inline Vec3 fresnel_schlick(float cos_theta, Vec3 f0) {
    float m = std::clamp(1.f - cos_theta, 0.f, 1.f);
    float m2 = m * m;
    return f0 + (Vec3(1.f) - f0) * (m2 * m2 * m);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sampling
// ─────────────────────────────────────────────────────────────────────────────

inline Vec3 sample_cosine_hemisphere(float u1, float u2) {
    float r = std::sqrt(u1);
    float phi = 2.0f * kPi * u2;
    return { r * std::cos(phi), r * std::sin(phi), std::sqrt(std::max(0.0f, 1.0f - u1)) };
}

inline Vec3 sample_ggx(float u1, float u2, float alpha) {
    float phi = 2.f * kPi * u1;
    float cos_theta = std::sqrt((1.f - u2) / (1.f + (alpha * alpha - 1.f) * u2));
    float sin_theta = std::sqrt(std::max(0.f, 1.f - cos_theta * cos_theta));
    return { sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta };
}

// ─────────────────────────────────────────────────────────────────────────────
// Principled BSDF API
// ─────────────────────────────────────────────────────────────────────────────

inline void get_lobe_weights(const PrincipledBSDF& mat, float& w_diff, float& w_refl, float& w_trans) {
    w_trans = (1.f - mat.metallic) * mat.transmission;
    w_diff  = (1.f - mat.metallic) * (1.f - mat.transmission);
    w_refl  = 1.0f - w_diff - w_trans; // metallic influence and reflection
    float sum = w_diff + w_refl + w_trans;
    if (sum > 0.f) {
        w_diff /= sum; w_refl /= sum; w_trans /= sum;
    }
}

inline Vec3 bsdf_eval(Vec3 wo, Vec3 wi, const PrincipledBSDF& mat) {
    if (wo.z <= 0.f) return Vec3(0.f);
    
    float w_diff, w_refl, w_trans;
    get_lobe_weights(mat, w_diff, w_refl, w_trans);
    
    Vec3 result(0.f);
    
    // Diffuse
    if (wi.z > 0.f && w_diff > 0.f) {
        result += w_diff * mat.albedo * kInvPi * wi.z;
    }
    
    // Reflection (Rough)
    if (wi.z > 0.f && w_refl > 0.f && mat.roughness > 0.01f) {
        Vec3 wh = normalize(wo + wi);
        float alpha = mat.roughness * mat.roughness;
        float d = ggx_d(wh, alpha);
        float g = ggx_g2(wo, wi, alpha);
        Vec3 f0 = lerp(Vec3(0.04f) * mat.specular * 2.f, mat.albedo, mat.metallic);
        Vec3 f = fresnel_schlick(dot(wo, wh), f0);
        result += w_refl * f * (d * g / (4.f * wo.z)); // denominator wi.z cancels
    }
    
    // Transmission (Rough)
    if (wi.z < 0.f && w_trans > 0.f) {
        float alpha = (mat.transmission_roughness < 0.f ? mat.roughness : mat.transmission_roughness);
        alpha = alpha * alpha;
        if (alpha > 0.01f) {
            float eta = wo.z > 0.f ? (1.f / mat.ior) : mat.ior;
            Vec3 wh = normalize(wo + wi * (1.f / eta));
            if (wh.z < 0.f) wh = -wh;
            float cos_theta_o = dot(wo, wh);
            float cos_theta_i = dot(wi, wh);
            float d = ggx_d(wh, alpha);
            float g = ggx_g2(wo, wi, alpha);
            float f = fresnel_dielectric(cos_theta_o, mat.ior);
            float denom = (cos_theta_o + eta * cos_theta_i);
            denom *= denom;
            if (denom > 1e-6f) {
                float val = std::abs(cos_theta_o * cos_theta_i) * d * g * (1.f - f) / (std::abs(wo.z) * denom);
                result += w_trans * mat.albedo * val; // Jacobian and eta scaling simplified for now
            }
        }
    }
    
    return result;
}

inline float bsdf_pdf(Vec3 wo, Vec3 wi, const PrincipledBSDF& mat) {
    if (wo.z <= 0.f) return 0.f;
    
    float w_diff, w_refl, w_trans;
    get_lobe_weights(mat, w_diff, w_refl, w_trans);
    
    float pdf = 0.f;
    
    // Diffuse PDF
    if (wi.z > 0.f && w_diff > 0.f) {
        pdf += w_diff * wi.z * kInvPi;
    }
    
    // Reflection PDF
    if (wi.z > 0.f && w_refl > 0.f) {
        float alpha = mat.roughness * mat.roughness;
        if (alpha > 0.01f) {
            Vec3 wh = normalize(wo + wi);
            pdf += w_refl * ggx_d(wh, alpha) * wh.z / (4.f * dot(wo, wh));
        }
    }
    
    // Transmission PDF
    if (wi.z < 0.f && w_trans > 0.f) {
        float alpha = (mat.transmission_roughness < 0.f ? mat.roughness : mat.transmission_roughness);
        alpha = alpha * alpha;
        if (alpha > 0.01f) {
            float eta = wo.z > 0.f ? (1.f / mat.ior) : mat.ior;
            Vec3 wh = normalize(wo + wi * (1.f / eta));
            if (wh.z < 0.f) wh = -wh;
            float cos_theta_o = dot(wo, wh);
            float cos_theta_i = dot(wi, wh);
            float d = ggx_d(wh, alpha);
            float jacobian = std::abs(cos_theta_i) / std::pow(cos_theta_o + eta * cos_theta_i, 2.f);
            pdf += w_trans * d * wh.z * jacobian;
        }
    }
    
    return pdf;
}

inline bool bsdf_sample(Vec3 wo, const PrincipledBSDF& mat, PCGState& rng, BSDFSample& res) {
    if (wo.z <= 0.f) return false;
    
    float w_diff, w_refl, w_trans;
    get_lobe_weights(mat, w_diff, w_refl, w_trans);
    
    float u = rng.next_float();
    if (u < w_diff) {
        // Sample Diffuse
        res.wi = sample_cosine_hemisphere(rng.next_float(), rng.next_float());
        res.lobe = LOBE_DIFFUSE;
        res.is_delta = false;
    } else if (u < w_diff + w_refl) {
        // Sample Reflection
        float alpha = mat.roughness * mat.roughness;
        if (alpha < 0.01f) {
            res.wi = reflect(-wo, Vec3(0, 0, 1));
            res.is_delta = true;
            res.lobe = LOBE_DELTA;
        } else {
            Vec3 wh = sample_ggx(rng.next_float(), rng.next_float(), alpha);
            res.wi = reflect(-wo, wh);
            res.is_delta = false;
            res.lobe = LOBE_GLOSSY_REFL;
        }
    } else {
        // Sample Transmission
        float alpha = (mat.transmission_roughness < 0.f ? mat.roughness : mat.transmission_roughness);
        alpha = alpha * alpha;
        if (alpha < 0.01f) {
            float f = fresnel_dielectric(wo.z, mat.ior);
            if (rng.next_float() < f) {
                res.wi = reflect(-wo, Vec3(0,0,1));
            } else {
                refract(-wo, Vec3(0,0,1), 1.f/mat.ior, res.wi);
            }
            res.is_delta = true;
            res.lobe = LOBE_DELTA;
        } else {
            Vec3 wh = sample_ggx(rng.next_float(), rng.next_float(), alpha);
            float f = fresnel_dielectric(dot(wo, wh), mat.ior);
            if (rng.next_float() < f) {
                res.wi = reflect(-wo, wh);
                res.lobe = LOBE_GLOSSY_REFL;
            } else {
                refract(-wo, wh, 1.f / mat.ior, res.wi);
                res.lobe = LOBE_GLOSSY_TRANS;
            }
            res.is_delta = false;
        }
    }
    
    if (res.is_delta) {
        res.pdf = 1.0f;
        res.f = bsdf_eval(wo, res.wi, mat); // Delta handles need special eval usually, but for now we'll keep it simple
        if (res.wi.z == 0.f) return false;
        res.f = mat.albedo; // Approximate for delta
    } else {
        res.pdf = bsdf_pdf(wo, res.wi, mat);
        if (res.pdf <= 0.f) return false;
        res.f = bsdf_eval(wo, res.wi, mat) / res.pdf;
    }
    
    return true;
}

} // namespace xn
