#pragma once
// material/material.h — Material description and .mat file loader

#include <cmath>
#include <string>
#include <stdexcept>

#include "math/vec3.h"
#include "math/ray.h"
#include "material/texture.h"
#include "material/opacity.h"

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

	Vec3 baseColor{0.5f};
	float roughness = 0.5f;
	float diffuseRoughness = 0.f;
	bool multipleScattering = false;
	float metallic = 0.0f;
	float ior = 1.5f;
	float transmission = 0.0f;
	float subsurface = 0.0f;
	Vec3 subsurfaceColor{1.0f};
	Vec3 subsurfaceMFP{1.0f}; // mean free path
	float clearcoat = 0.0f;
	float clearcoatRoughness = 0.03f;
	float clearcoatIor = 1.5f;
	Vec3 absorption{0.f}; // Beer absorption coefficient in inverse meters
	float anisotropy = 0.0f;
	Vec3 emission{0.0f};
	bool emissionTwoSided = false; // same radiance on each side when enabled
	float emissionTemperature = 6500.0f;
	enum class OpacityMode : uint8_t { Mask = 0, Blend = 1 };
	OpacityMode opacityMode = OpacityMode::Mask;
	float opacity = 1.f, alphaCutoff = 0.5f; // coverage alpha, distinct from BSDF transmission
	float normalScale = 1.f, bumpScale = 1.f, textureLod = 0.f;
	TextureMap baseColorTexture, roughnessTexture, metallicTexture, emissionTexture;
	TextureMap opacityTexture, normalTexture, bumpTexture;

	bool has_textures() const {
		return baseColorTexture || roughnessTexture || metallicTexture || emissionTexture ||
			   opacityTexture || normalTexture || bumpTexture;
	}

	// ── Precomputed / derived ────────────────────────────────────────────────
	float alpha = 0.25f;   // roughness^2, clamped
	float alpha_x = 0.25f; // anisotropic alpha (tangent)
	float alpha_y = 0.25f; // anisotropic alpha (bitangent)
	Vec3 F0{0.04f};		   // Fresnel reflectance at normal incidence

	// Flags (set by material_prepare)
	bool isConductor = false;
	bool isTransmissive = false;
	bool hasSubsurface = false;
	bool isDelta = false;
	bool isEmissive = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// Precompute derived fields from authoring parameters.
// Must be called after setting authoring params and before rendering.
// ─────────────────────────────────────────────────────────────────────────────
inline void material_prepare(Material &m) {
	for (float v :
		 {m.baseColor.x, m.baseColor.y, m.baseColor.z, m.roughness, m.diffuseRoughness, m.clearcoat,
		  m.clearcoatRoughness, m.metallic, m.transmission, m.subsurface, m.opacity, m.alphaCutoff,
		  m.subsurfaceColor.x, m.subsurfaceColor.y, m.subsurfaceColor.z}) {
		if (!std::isfinite(v) || v < 0.f || v > 1.f)
			throw std::runtime_error("Material reflectances and weights must be in [0,1]");
	}

	if (!std::isfinite(m.ior) || m.ior <= 0.f || !std::isfinite(m.anisotropy) ||
		std::abs(m.anisotropy) > 1.f)
		throw std::runtime_error("Invalid material IOR or anisotropy");

	for (float v : {m.emission.x, m.emission.y, m.emission.z})
		if (!std::isfinite(v) || v < 0.f)
			throw std::runtime_error("Invalid material emission");

	if (!std::isfinite(m.normalScale) || !std::isfinite(m.bumpScale) ||
		!std::isfinite(m.textureLod) || m.textureLod < 0.f)
		throw std::runtime_error("Invalid texture scale or LOD");

	if (m.multipleScattering &&
		(m.anisotropy != 0.f || m.transmission != 0.f || m.ior < 1.f || m.ior > 3.f))
		throw std::runtime_error(
			"multipleScattering supports opaque isotropic materials with IOR in [1,3]");

	if (!std::isfinite(m.clearcoatIor) || m.clearcoatIor < 1.f || m.clearcoatIor > 3.f)
		throw std::runtime_error("clearcoatIor must be in [1,3]");

	if (m.clearcoat > 0.f && m.transmission > 0.f)
		throw std::runtime_error("Clearcoat currently supports opaque substrates only");

	for (float v : {m.absorption.x, m.absorption.y, m.absorption.z})
		if (!std::isfinite(v) || v < 0.f)
			throw std::runtime_error("Absorption coefficients must be finite and nonnegative");

	if (max_component(m.absorption) > 0.f && (m.transmission <= 0.f || m.metallic >= 1.f))
		throw std::runtime_error("Absorption requires a transmissive dielectric material");
	// Alpha from roughness (clamp to small positive to avoid div-by-zero)
	m.alpha = std::max(m.roughness * m.roughness, 1e-6f);

	// Anisotropic alphas (from Burley/Disney parameterization)
	float aspect = std::sqrt(1.0f - 0.9f * std::abs(m.anisotropy));
	m.alpha_x = std::max(m.alpha / aspect, 1e-6f);
	m.alpha_y = std::max(m.alpha * aspect, 1e-6f);

	if (m.anisotropy < 0.f)
		std::swap(m.alpha_x, m.alpha_y);

	// F0: lerp between dielectric (from IOR) and conductor (baseColor)
	float f0_dielectric = ((m.ior - 1.0f) * (m.ior - 1.0f)) / ((m.ior + 1.0f) * (m.ior + 1.0f));
	m.F0 = lerp(Vec3(f0_dielectric), m.baseColor, m.metallic);

	// Classification flags
	m.isConductor = (m.metallic > 0.5f);
	m.isTransmissive = (m.transmission > 0.f && m.metallic < 1.f);
	m.hasSubsurface = (m.subsurface > 0.f);
	m.isDelta = ((std::max(m.alpha_x, m.alpha_y) < kDeltaAlphaThreshold ||
				  (m.ior == 1.f && m.metallic == 0.f)) &&
				 (m.metallic == 1.f || m.transmission == 1.f)) &&
				(m.clearcoat == 0.f || m.clearcoatIor == 1.f ||
				 m.clearcoatRoughness * m.clearcoatRoughness < kDeltaAlphaThreshold);
	m.isEmissive = (m.emission.x > 0.0f || m.emission.y > 0.0f || m.emission.z > 0.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Load a .mat file from disk. Returns a prepared Material.
// Results are cached internally — repeated calls with the same path are free.
// ─────────────────────────────────────────────────────────────────────────────
Material load_material(const std::string &path);
// Evaluate scalar/color maps and tangent normal/bump maps at a hit.
Material evaluate_material(const Material &material, HitRecord &hit);
float surface_opacity(const Material &material, const HitRecord &hit);
bool surface_opaque(const Material &material, const HitRecord &hit, uint32_t seed = 0);
Vec3 mapped_emission(const Material &material, float u, float v);

// ─────────────────────────────────────────────────────────────────────────────
// Convert a legacy inline material definition to Material.
// Used for backward compatibility with old .xenon files.
// ─────────────────────────────────────────────────────────────────────────────
Material material_from_legacy(const std::string &name, Vec3 albedo, float metallic, float roughness,
							  float specular, float ior, float transmission,
							  float transmission_roughness);

} // namespace xn
