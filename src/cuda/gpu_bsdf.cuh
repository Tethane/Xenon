#pragma once
// cuda/gpu_bsdf.cuh — Device-side BSDF evaluation/sampling/pdf
//
// Port of the CPU bsdf.h (6 lobes: diffuse, microfacet_refl, microfacet_trans,
// delta_refl, delta_trans, subsurface). All std:: calls replaced with CUDA
// device-safe equivalents.

#include "cuda/gpu_types.cuh"

namespace xn {

// ─── BSDFSample ──────────────────────────────────────────────────────────────

struct GpuBSDFSample {
  float3 wi    = {};
  float3 f     = {};
  float  pdf   = 0.f;
  bool   delta = false;
};

// ─── GGX / Smith helpers ─────────────────────────────────────────────────────

__device__ inline float gpu_ggx_D(float ndoth, float alpha) {
  float a2 = alpha * alpha;
  float d  = ndoth * ndoth * (a2 - 1.f) + 1.f;
  return a2 / (kGpuPi * d * d + 1e-7f);
}

__device__ inline float gpu_smith_G1(float ndotv, float alpha) {
  float a2 = alpha * alpha;
  float cos2 = ndotv * ndotv;
  return 2.f * ndotv / (ndotv + sqrtf(cos2 + a2 * (1.f - cos2)) + 1e-7f);
}

__device__ inline float gpu_smith_G(float ndotl, float ndotv, float alpha) {
  return gpu_smith_G1(ndotl, alpha) * gpu_smith_G1(ndotv, alpha);
}

__device__ inline float gpu_fresnel_schlick(float cos_theta, float f0) {
  float omc = 1.f - cos_theta;
  float omc2 = omc * omc;
  return f0 + (1.f - f0) * omc2 * omc2 * omc;
}

__device__ inline float3 gpu_fresnel_schlick3(float cos_theta, float3 f0) {
  float omc = 1.f - cos_theta;
  float omc2 = omc * omc;
  float factor = omc2 * omc2 * omc;
  return f0 + (make_f3(1.f) - f0) * factor;
}

__device__ inline float gpu_fresnel_dielectric(float cos_i, float eta) {
  float sin2_t = eta * eta * (1.f - cos_i * cos_i);
  if (sin2_t >= 1.f) return 1.f;
  float cos_t = sqrtf(1.f - sin2_t);
  float r_s = (eta * cos_i - cos_t) / (eta * cos_i + cos_t + 1e-8f);
  float r_p = (cos_i - eta * cos_t) / (cos_i + eta * cos_t + 1e-8f);
  return 0.5f * (r_s * r_s + r_p * r_p);
}

// ─── VNDF sampling (Heitz' visible normal distribution) ──────────────────────

__device__ inline float3 gpu_sample_vndf(float3 Ve, float alpha, float u1, float u2) {
  float3 Vh = normalize3(make_f3(alpha * Ve.x, alpha * Ve.y, Ve.z));
  float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
  float3 T1 = (lensq > 1e-7f)
    ? make_f3(-Vh.y, Vh.x, 0.f) / sqrtf(lensq)
    : make_f3(1.f, 0.f, 0.f);
  float3 T2 = cross3(Vh, T1);

  float r = sqrtf(u1);
  float phi = 2.f * kGpuPi * u2;
  float t1 = r * cosf(phi);
  float t2 = r * sinf(phi);
  float s = 0.5f * (1.f + Vh.z);
  t2 = (1.f - s) * sqrtf(1.f - t1 * t1) + s * t2;

  float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.f, 1.f - t1*t1 - t2*t2)) * Vh;
  float3 Ne = normalize3(make_f3(alpha * Nh.x, alpha * Nh.y, fmaxf(1e-6f, Nh.z)));
  return Ne;
}

// ─── MIS helpers ─────────────────────────────────────────────────────────────

__device__ inline float gpu_power_heuristic(float pdf_a, float pdf_b) {
  float a2 = pdf_a * pdf_a;
  return a2 / (a2 + pdf_b * pdf_b + 1e-10f);
}

// ─── Cosine-weighted hemisphere sampling ─────────────────────────────────────

__device__ inline float3 gpu_cosine_sample_hemisphere(float u1, float u2) {
  float phi = 2.f * kGpuPi * u1;
  float r = sqrtf(u2);
  return make_f3(r * cosf(phi), r * sinf(phi), sqrtf(fmaxf(0.f, 1.f - u2)));
}

// ═════════════════════════════════════════════════════════════════════════════
// Diffuse lobe
// ═════════════════════════════════════════════════════════════════════════════

__device__ inline float3 gpu_diffuse_eval(const GpuMaterial& m, float ndotl) {
  return m.baseColor * (kGpuInvPi * ndotl);
}

__device__ inline float gpu_diffuse_pdf(float ndotl) {
  return ndotl * kGpuInvPi;
}

__device__ inline bool gpu_diffuse_sample(float3 wo, const GpuMaterial& m, GpuPCG& rng, GpuBSDFSample& res) {
  float3 local_w = gpu_cosine_sample_hemisphere(rng.next_float(), rng.next_float());
  if (local_w.z < 1e-6f) return false;
  res.wi    = local_w; // caller transforms to world space
  float ndotl = local_w.z;
  res.f     = m.baseColor * kGpuInvPi;
  res.pdf   = ndotl * kGpuInvPi;
  res.delta = false;
  return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Microfacet reflection lobe (GGX)
// ═════════════════════════════════════════════════════════════════════════════

__device__ inline bool gpu_microfacet_refl_sample(float3 wo_local, const GpuMaterial& m, GpuPCG& rng, GpuBSDFSample& res) {
  if (wo_local.z < 1e-6f) return false;

  float3 wh = gpu_sample_vndf(wo_local, m.alpha, rng.next_float(), rng.next_float());
  if (dot3(wh, wo_local) < 1e-6f) return false;

  float3 wi = reflect3(-wo_local, wh);
  if (wi.z < 1e-6f) return false;

  float ndotl = wi.z;
  float ndotv = wo_local.z;
  float vdoth = dot3(wo_local, wh);
  float ndoth = wh.z;

  float D = gpu_ggx_D(ndoth, m.alpha);
  float G = gpu_smith_G(ndotl, ndotv, m.alpha);
  float3 F = gpu_fresnel_schlick3(vdoth, m.F0);

  res.wi  = wi;
  res.f   = D * G * F / (4.f * ndotv + 1e-7f);
  res.pdf = D * gpu_smith_G1(ndotv, m.alpha) * vdoth / (ndotv * ndoth + 1e-7f);
  // Jacobian: dw = dh / (4 VdotH)
  res.pdf /= (4.f * vdoth + 1e-7f);
  res.delta = false;
  return res.pdf > 1e-10f;
}

__device__ inline float3 gpu_microfacet_refl_eval(float3 wo_local, float3 wi_local, const GpuMaterial& m) {
  float ndotl = wi_local.z;
  float ndotv = wo_local.z;
  if (ndotl < 1e-6f || ndotv < 1e-6f) return make_f3(0.f);

  float3 wh = normalize3(wo_local + wi_local);
  float ndoth = wh.z;
  float vdoth = dot3(wo_local, wh);

  float D = gpu_ggx_D(ndoth, m.alpha);
  float G = gpu_smith_G(ndotl, ndotv, m.alpha);
  float3 F = gpu_fresnel_schlick3(vdoth, m.F0);

  return D * G * F / (4.f * ndotv + 1e-7f);
}

__device__ inline float gpu_microfacet_refl_pdf(float3 wo_local, float3 wi_local, const GpuMaterial& m) {
  float ndotv = wo_local.z;
  if (ndotv < 1e-6f) return 0.f;

  float3 wh = normalize3(wo_local + wi_local);
  float ndoth = wh.z;
  float vdoth = dot3(wo_local, wh);

  float D = gpu_ggx_D(ndoth, m.alpha);
  float pdf = D * gpu_smith_G1(ndotv, m.alpha) * vdoth / (ndotv * ndoth + 1e-7f);
  pdf /= (4.f * vdoth + 1e-7f);
  return pdf;
}

// ═════════════════════════════════════════════════════════════════════════════
// Microfacet transmission lobe  (GGX + Snell refraction)
// ═════════════════════════════════════════════════════════════════════════════

__device__ inline bool gpu_microfacet_trans_sample(float3 wo_local, const GpuMaterial& m, GpuPCG& rng, GpuBSDFSample& res) {
  if (fabsf(wo_local.z) < 1e-6f) return false;

  bool entering = wo_local.z > 0.f;
  float eta = entering ? (1.f / m.ior) : m.ior;
  float3 wo_flip = entering ? wo_local : make_f3(wo_local.x, wo_local.y, -wo_local.z);

  float3 wh = gpu_sample_vndf(wo_flip, m.alpha, rng.next_float(), rng.next_float());
  float cos_i = dot3(wo_flip, wh);
  if (cos_i < 1e-6f) return false;

  float F_val = gpu_fresnel_dielectric(cos_i, eta);
  // Stochastic selection: reflect or refract
  if (rng.next_float() < F_val) {
    // Reflect (rare for dielectrics, but physically correct)
    float3 wi = reflect3(-wo_flip, wh);
    if (wi.z < 1e-6f) return false;
    if (!entering) wi = make_f3(wi.x, wi.y, -wi.z);
    res.wi = wi;
    float ndotv = fabsf(wo_local.z);
    float ndotl = fabsf(wi.z);
    float ndoth = wh.z;
    float D = gpu_ggx_D(ndoth, m.alpha);
    float G = gpu_smith_G(ndotl, ndotv, m.alpha);
    res.f = m.baseColor * D * G / (4.f * ndotv + 1e-7f);
    res.pdf = F_val * D * gpu_smith_G1(ndotv, m.alpha) * cos_i / (ndotv * ndoth + 1e-7f) / (4.f * cos_i + 1e-7f);
    res.delta = false;
    return res.pdf > 1e-10f;
  }

  float sin2_t = eta * eta * (1.f - cos_i * cos_i);
  if (sin2_t >= 1.f) return false;
  float cos_t = sqrtf(1.f - sin2_t);
  float3 wi_local_refracted = eta * (-wo_flip) + (eta * cos_i - cos_t) * wh;
  wi_local_refracted = normalize3(wi_local_refracted);
  if (wi_local_refracted.z > -1e-6f) return false;

  if (!entering) wi_local_refracted = make_f3(wi_local_refracted.x, wi_local_refracted.y, -wi_local_refracted.z);

  float ndotv = fabsf(wo_local.z);
  float ndotl = fabsf(wi_local_refracted.z);
  float ndoth = wh.z;
  float D = gpu_ggx_D(ndoth, m.alpha);
  float G = gpu_smith_G(ndotl, ndotv, m.alpha);

  float denom = (eta * cos_i + cos_t);
  float dwh_dwi = cos_t / (denom * denom + 1e-7f);

  res.wi = wi_local_refracted;
  res.f = m.baseColor * (1.f - F_val) * D * G * fabsf(cos_i) * dwh_dwi / (ndotv * fabsf(ndoth) + 1e-7f);
  res.pdf = (1.f - F_val) * D * gpu_smith_G1(ndotv, m.alpha) * cos_i / (ndotv * ndoth + 1e-7f) * dwh_dwi;
  res.delta = false;
  return res.pdf > 1e-10f;
}

// ═════════════════════════════════════════════════════════════════════════════
// Delta reflection (perfect mirror)
// ═════════════════════════════════════════════════════════════════════════════

__device__ inline bool gpu_delta_refl_sample(float3 wo, const GpuMaterial& m, GpuBSDFSample& res) {
  if (wo.z < 1e-6f) return false;
  res.wi    = make_f3(-wo.x, -wo.y, wo.z);
  res.f     = gpu_fresnel_schlick3(wo.z, m.F0) / (wo.z + 1e-7f);
  res.pdf   = 1.f;
  res.delta = true;
  return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Delta transmission (perfect glass)
// ═════════════════════════════════════════════════════════════════════════════

__device__ inline bool gpu_delta_trans_sample(float3 wo_local, const GpuMaterial& m, GpuPCG& rng, GpuBSDFSample& res) {
  bool entering = wo_local.z > 0.f;
  float cos_i = fabsf(wo_local.z);
  float eta = entering ? (1.f / m.ior) : m.ior;
  float F_val = gpu_fresnel_dielectric(cos_i, eta);

  if (rng.next_float() < F_val) {
    res.wi    = make_f3(-wo_local.x, -wo_local.y, wo_local.z);
    res.f     = make_f3(1.f) / (cos_i + 1e-7f);
    res.pdf   = 1.f;
    res.delta = true;
    return true;
  }

  float sin2_t = eta * eta * (1.f - cos_i * cos_i);
  if (sin2_t >= 1.f) {
    res.wi    = make_f3(-wo_local.x, -wo_local.y, wo_local.z);
    res.f     = make_f3(1.f) / (cos_i + 1e-7f);
    res.pdf   = 1.f;
    res.delta = true;
    return true;
  }

  float cos_t = sqrtf(1.f - sin2_t);
  float3 n = make_f3(0, 0, 1);
  if (!entering) n = -n;
  float3 wo_n = normalize3(wo_local);
  float3 wi = eta * (-wo_n) + (eta * dot3(wo_n, n) - cos_t) * n;
  wi = normalize3(wi);

  res.wi    = wi;
  res.f     = m.baseColor / (fabsf(wi.z) + 1e-7f);
  res.pdf   = 1.f;
  res.delta = true;
  return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Subsurface approximation (diffuse-like)
// ═════════════════════════════════════════════════════════════════════════════

__device__ inline bool gpu_subsurface_sample(float3 wo, const GpuMaterial& m, GpuPCG& rng, GpuBSDFSample& res) {
  float3 local_w = gpu_cosine_sample_hemisphere(rng.next_float(), rng.next_float());
  // Subsurface: transmit through surface = flip z
  local_w.z = -local_w.z;
  if (fabsf(local_w.z) < 1e-6f) return false;

  res.wi    = local_w;
  float ndotl = fabsf(local_w.z);
  res.f     = m.subsurfaceColor * kGpuInvPi;
  res.pdf   = ndotl * kGpuInvPi;
  res.delta = false;
  return true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Unified BSDF sampling (dispatches to appropriate lobe)
// ═════════════════════════════════════════════════════════════════════════════

__device__ inline bool gpu_bsdf_sample(float3 wo_local, const GpuMaterial& m, GpuPCG& rng, GpuBSDFSample& res) {
  if (m.isDelta()) {
    if (m.isTransmissive()) {
      return gpu_delta_trans_sample(wo_local, m, rng, res);
    } else {
      return gpu_delta_refl_sample(wo_local, m, res);
    }
  }

  if (m.isTransmissive()) {
    return gpu_microfacet_trans_sample(wo_local, m, rng, res);
  }

  if (m.hasSubsurface()) {
    float prob = 0.5f;
    if (rng.next_float() < prob) {
      if (!gpu_subsurface_sample(wo_local, m, rng, res)) return false;
      res.pdf *= prob;
      return true;
    } else {
      if (!gpu_diffuse_sample(wo_local, m, rng, res)) return false;
      res.pdf *= (1.f - prob);
      return true;
    }
  }

  if (m.isConductor()) {
    return gpu_microfacet_refl_sample(wo_local, m, rng, res);
  }

  // Opaque dielectric: stochastic diffuse/specular split
  float spec_weight = fminf(fmaxf(m.F0.x, 0.f), 1.f);
  float diff_weight = 1.f - spec_weight;

  if (rng.next_float() < spec_weight) {
    if (!gpu_microfacet_refl_sample(wo_local, m, rng, res)) return false;
    res.pdf *= spec_weight;
    return true;
  } else {
    if (!gpu_diffuse_sample(wo_local, m, rng, res)) return false;
    res.pdf *= diff_weight;
    return true;
  }
}

// ─── BSDF eval for NEE ──────────────────────────────────────────────────────

__device__ inline float3 gpu_bsdf_eval_for_nee(float3 wo_local, float3 wi_local, const GpuMaterial& m) {
  if (m.isDelta()) return make_f3(0.f);
  if (m.isTransmissive()) return make_f3(0.f); // non-trivial; skip for now

  float ndotl = wi_local.z;
  float ndotv = wo_local.z;
  if (ndotl < 1e-6f || ndotv < 1e-6f) return make_f3(0.f);

  if (m.isConductor()) {
    return gpu_microfacet_refl_eval(wo_local, wi_local, m);
  }

  // Opaque dielectric: sum both lobes
  float spec_weight = fminf(fmaxf(m.F0.x, 0.f), 1.f);
  float diff_weight = 1.f - spec_weight;

  float3 f_diff = gpu_diffuse_eval(m, ndotl) * diff_weight;
  float3 f_spec = gpu_microfacet_refl_eval(wo_local, wi_local, m) * spec_weight;
  return f_diff + f_spec;
}

__device__ inline float gpu_bsdf_pdf_for_nee(float3 wo_local, float3 wi_local, const GpuMaterial& m) {
  if (m.isDelta()) return 0.f;
  if (m.isTransmissive()) return 0.f;

  float ndotl = wi_local.z;
  float ndotv = wo_local.z;
  if (ndotl < 1e-6f || ndotv < 1e-6f) return 0.f;

  if (m.isConductor()) {
    return gpu_microfacet_refl_pdf(wo_local, wi_local, m);
  }

  float spec_weight = fminf(fmaxf(m.F0.x, 0.f), 1.f);
  float diff_weight = 1.f - spec_weight;

  float pdf_d = gpu_diffuse_pdf(ndotl) * diff_weight;
  float pdf_s = gpu_microfacet_refl_pdf(wo_local, wi_local, m) * spec_weight;
  return pdf_d + pdf_s;
}

} // namespace xn
