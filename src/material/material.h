#pragma once
// material/material.h — Material description and .mat file loader

#include <cmath>
#include <string>

#include "math/vec3.h"

namespace xn {

// ─────────────────────────────────────────────────────────────────────────────
// Material — internal representation of a surface material
//   Stores both authoring parameters (from .mat files) and precomputed
//   derived quantities used by BSDF kernels and classification.
// ─────────────────────────────────────────────────────────────────────────────

constexpr float kDeltaAlphaThreshold = 0.001f;

struct Material {
    // ── Authoring parameters ─────────────────────────────────────────────────
    std::string name = "default";

    Vec3  baseColor{0.5f};
    float roughness       = 0.5f;
    float metallic        = 0.0f;
    float ior             = 1.5f;
    float transmission    = 0.0f;
    float subsurface      = 0.0f;
    Vec3  subsurfaceColor{1.0f};
    Vec3  subsurfaceMFP{1.0f};           // mean free path
    float clearcoat          = 0.0f;
    float clearcoatRoughness = 0.03f;
    float anisotropy         = 0.0f;
    Vec3  emission{0.0f};
    float emissionTemperature = 6500.0f;

    // ── Precomputed / derived ────────────────────────────────────────────────
    float alpha   = 0.25f;              // roughness^2, clamped
    float alpha_x = 0.25f;              // anisotropic alpha (tangent)
    float alpha_y = 0.25f;              // anisotropic alpha (bitangent)
    Vec3  F0{0.04f};                    // Fresnel reflectance at normal incidence

    // Flags (set by material_prepare)
    bool isConductor    = false;
    bool isTransmissive = false;
    bool hasSubsurface  = false;
    bool isDelta        = false;
    bool isEmissive     = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// Precompute derived fields from authoring parameters.
// Must be called after setting authoring params and before rendering.
// ─────────────────────────────────────────────────────────────────────────────
inline void material_prepare(Material& m) {
    // Alpha from roughness (clamp to small positive to avoid div-by-zero)
    m.alpha = std::max(m.roughness * m.roughness, 1e-6f);

    // Anisotropic alphas (from Burley/Disney parameterization)
    float aspect = std::sqrt(1.0f - 0.9f * std::abs(m.anisotropy));
    m.alpha_x = std::max(m.alpha / aspect, 1e-6f);
    m.alpha_y = std::max(m.alpha * aspect, 1e-6f);

    // F0: lerp between dielectric (from IOR) and conductor (baseColor)
    float f0_dielectric = ((m.ior - 1.0f) * (m.ior - 1.0f)) /
                          ((m.ior + 1.0f) * (m.ior + 1.0f));
    m.F0 = lerp(Vec3(f0_dielectric), m.baseColor, m.metallic);

    // Classification flags
    m.isConductor    = (m.metallic > 0.5f);
    m.isTransmissive = (m.transmission > 0.01f);
    m.hasSubsurface  = (m.subsurface > 0.01f);
    m.isDelta        = (m.alpha < kDeltaAlphaThreshold);
    m.isEmissive     = (m.emission.x > 0.0f || m.emission.y > 0.0f || m.emission.z > 0.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Load a .mat file from disk. Returns a prepared Material.
// Results are cached internally — repeated calls with the same path are free.
// ─────────────────────────────────────────────────────────────────────────────
Material load_material(const std::string& path);

// ─────────────────────────────────────────────────────────────────────────────
// Convert a legacy inline material definition to Material.
// Used for backward compatibility with old .xenon files.
// ─────────────────────────────────────────────────────────────────────────────
Material material_from_legacy(const std::string& name, Vec3 albedo, float metallic,
                              float roughness, float specular, float ior,
                              float transmission, float transmission_roughness);

} // namespace xn
