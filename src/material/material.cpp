// material/material.cpp — .mat file parser and Material cache

#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "material/material.h"

namespace xn {

// ─────────────────────────────────────────────────────────────────────────────
// Global material cache (thread-safe)
// ─────────────────────────────────────────────────────────────────────────────
static std::mutex s_cache_mutex;
static std::unordered_map<std::string, Material> s_material_cache;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: strip quotes from a string token
// ─────────────────────────────────────────────────────────────────────────────
static std::string strip_quotes(const std::string &s) {
	if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
		return s.substr(1, s.size() - 2);

	return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parse a .mat file into a Material
// ─────────────────────────────────────────────────────────────────────────────
static bool parse_mat_file(const std::string &path, Material &mat) {
	std::ifstream f(path);

	if (!f.is_open()) {
		std::fprintf(stderr, "[Material] Failed to open: %s\n", path.c_str());

		return false;
	}

	std::string line;
	int line_num = 0;
	int version = 1;

	while (std::getline(f, line)) {
		++line_num;

		// Skip empty lines and comments
		size_t first = line.find_first_not_of(" \t");

		if (first == std::string::npos || line[first] == '#')
			continue;

		std::istringstream ss(line);
		std::string key;
		ss >> key;

		if (key == "version") {
			ss >> version;

			if (version != 2)
				throw std::runtime_error("Unsupported material version");
		} else if (key == "name") {
			std::string val;
			ss >> std::quoted(val);
			mat.name = strip_quotes(val);
		} else if (key == "type") {
			std::string val;
			ss >> val;
			// Currently only "principled" is supported
			val = strip_quotes(val);

			if (val != "principled") {
				throw std::runtime_error("Unsupported material type: " + val);
			}
		} else if (key == "baseColor") {
			ss >> mat.baseColor.x >> mat.baseColor.y >> mat.baseColor.z;
		} else if (key == "roughness") {
			ss >> mat.roughness;
		} else if (key == "diffuseRoughness") {
			ss >> mat.diffuseRoughness;
		} else if (key == "multipleScattering") {
			int enabled = 0;
			ss >> enabled;

			if (enabled != 0 && enabled != 1)
				throw std::runtime_error("multipleScattering must be 0 or 1");
			mat.multipleScattering = enabled != 0;
		} else if (key == "metallic") {
			ss >> mat.metallic;
		} else if (key == "ior") {
			ss >> mat.ior;
		} else if (key == "transmission") {
			ss >> mat.transmission;
		} else if (key == "subsurface") {
			ss >> mat.subsurface;
		} else if (key == "subsurfaceColor") {
			ss >> mat.subsurfaceColor.x >> mat.subsurfaceColor.y >> mat.subsurfaceColor.z;
		} else if (key == "subsurfaceMFP") {
			ss >> mat.subsurfaceMFP.x >> mat.subsurfaceMFP.y >> mat.subsurfaceMFP.z;

			if (mat.subsurfaceMFP != Vec3(1.f))
				throw std::runtime_error(
					"subsurfaceMFP is unsupported; legacy subsurface is only a diffuse proxy");
		} else if (key == "clearcoat") {
			ss >> mat.clearcoat;
		} else if (key == "clearcoatIor") {
			ss >> mat.clearcoatIor;
		} else if (key == "absorption") {
			ss >> mat.absorption.x >> mat.absorption.y >> mat.absorption.z;
		} else if (key == "clearcoatRoughness") {
			ss >> mat.clearcoatRoughness;
		} else if (key == "anisotropy" || key == "anisotropic") {
			ss >> mat.anisotropy;
		} else if (key == "emission") {
			ss >> mat.emission.x >> mat.emission.y >> mat.emission.z;
		} else if (key == "emissionTwoSided") {
			int enabled = 0;
			ss >> enabled;

			if (enabled != 0 && enabled != 1)
				throw std::runtime_error("emissionTwoSided must be 0 or 1");
			mat.emissionTwoSided = enabled != 0;
		} else if (key == "emissionTemperature") {
			ss >> mat.emissionTemperature;

			if (mat.emissionTemperature != 6500.f)
				throw std::runtime_error(
					"emissionTemperature is unsupported; supply linear RGB emission");
		} else if (key == "opacityMode") {
			std::string mode;
			ss >> mode;

			if (mode == "mask")
				mat.opacityMode = Material::OpacityMode::Mask;
			else if (mode == "blend")
				mat.opacityMode = Material::OpacityMode::Blend;
			else
				throw std::runtime_error("opacityMode must be mask or blend");
		} else if (key == "opacity") {
			ss >> mat.opacity;
		} else if (key == "alphaCutoff") {
			ss >> mat.alphaCutoff;
		} else if (key == "normalScale") {
			ss >> mat.normalScale;
		} else if (key == "bumpScale") {
			ss >> mat.bumpScale;
		} else if (key == "textureLod") {
			ss >> mat.textureLod;
		} else if (key.ends_with("Texture")) {
			TextureMap *map = nullptr;

			if (key == "baseColorTexture")
				map = &mat.baseColorTexture;
			else if (key == "roughnessTexture")
				map = &mat.roughnessTexture;
			else if (key == "metallicTexture")
				map = &mat.metallicTexture;
			else if (key == "emissionTexture")
				map = &mat.emissionTexture;
			else if (key == "opacityTexture")
				map = &mat.opacityTexture;
			else if (key == "normalTexture")
				map = &mat.normalTexture;
			else if (key == "bumpTexture")
				map = &mat.bumpTexture;

			if (!map)
				throw std::runtime_error("Unknown material key: " + key);
			std::string image, space;
			ss >> std::quoted(image) >> space;

			if (ss.fail())
				throw std::runtime_error("Texture requires path and srgb/linear color space: " +
										 key);

			if (space != "srgb" && space != "linear")
				throw std::runtime_error("Unknown texture color space: " + space);

			if (space != "linear" && key != "baseColorTexture" && key != "emissionTexture")
				throw std::runtime_error("Data textures must use linear color space: " + key);

			if (!ss.eof())
				ss >> std::ws;

			if (!ss.eof() && ss.peek() != '#') {
				std::string channel;
				ss >> channel;

				if (channel == "r")
					map->channel = 0;
				else if (channel == "g")
					map->channel = 1;
				else if (channel == "b")
					map->channel = 2;
				else if (channel == "a")
					map->channel = 3;
				else
					throw std::runtime_error("Unknown texture channel: " + channel);
			}
			map->image = load_texture(
				(std::filesystem::path(path).parent_path() / image).lexically_normal().string(),
				space == "srgb");
		} else {
			throw std::runtime_error(path + ":" + std::to_string(line_num) +
									 ": unknown material key '" + key + "'");
		}

		if (ss.fail()) {
			std::fprintf(stderr, "[Material] %s:%d: parse error for key '%s'\n", path.c_str(),
						 line_num, key.c_str());

			return false;
		}

		if (!ss.eof()) {
			ss >> std::ws;

			if (!ss.eof() && ss.peek() != '#')
				throw std::runtime_error(path + ":" + std::to_string(line_num) +
										 ": unexpected fields for " + key);
		}
	}

	if (version >= 2 && mat.subsurface != 0.f)
		throw std::runtime_error("Subsurface transport is unsupported in material version 2");

	return !f.bad();
}

// ─────────────────────────────────────────────────────────────────────────────
// load_material — public API, cached
// ─────────────────────────────────────────────────────────────────────────────
Material load_material(const std::string &path) {
	{
		std::lock_guard<std::mutex> lock(s_cache_mutex);
		auto it = s_material_cache.find(path);

		if (it != s_material_cache.end())
			return it->second;
	}

	Material mat;

	if (!parse_mat_file(path, mat)) {
		throw std::runtime_error("Failed to load material: " + path);
	}
	material_prepare(mat);

	{
		std::lock_guard<std::mutex> lock(s_cache_mutex);
		s_material_cache[path] = mat;
	}

	return mat;
}

// ─────────────────────────────────────────────────────────────────────────────
// material_from_legacy — convert old inline PrincipledBSDF params to Material
// ─────────────────────────────────────────────────────────────────────────────
Material material_from_legacy(const std::string &name, Vec3 albedo, float metallic, float roughness,
							  float specular, float ior, float transmission,
							  float transmission_roughness) {
	Material m;
	m.name = name;
	m.baseColor = albedo;
	m.metallic = metallic;
	m.roughness = roughness;
	m.ior = ior;
	m.transmission = transmission;
	// Legacy: specular scaled IOR-based F0 (approximate)
	if (specular != 0.5f)
		throw std::runtime_error(
			"Legacy specular other than 0.5 is unsupported; set dielectric IOR explicitly");
	// If transmission_roughness was set (>= 0), use it for roughness on transmissive lobes
	if (transmission_roughness >= 0.0f && transmission > 0.01f) {
		m.roughness = transmission_roughness;
	}
	material_prepare(m);

	return m;
}

float surface_opacity(const Material &m, const HitRecord &hit) {
	float a = m.opacity;

	if (m.opacityTexture)
		a *= m.opacityTexture.scalar(hit.tex_u, hit.tex_v, m.textureLod);

	return std::clamp(a, 0.f, 1.f);
}

bool surface_opaque(const Material &m, const HitRecord &hit, uint32_t seed) {
	float a = surface_opacity(m, hit);

	return m.opacityMode == Material::OpacityMode::Blend
			   ? opacity_random(seed, hit.instance_id, hit.prim_id, hit.front_face) < a
			   : a >= m.alphaCutoff;
}

Vec3 mapped_emission(const Material &m, float u, float v) {
	return m.emission * m.emissionTexture.sample(u, v, m.textureLod).rgb();
}

Material evaluate_material(const Material &source, HitRecord &hit) {
	Material m = source;

	if (!m.has_textures())
		return m;
	float u = hit.tex_u, v = hit.tex_v, lod = m.textureLod;
	m.baseColor = m.baseColor * m.baseColorTexture.sample(u, v, lod).rgb();
	m.baseColor = {std::clamp(m.baseColor.x, 0.f, 1.f), std::clamp(m.baseColor.y, 0.f, 1.f),
				   std::clamp(m.baseColor.z, 0.f, 1.f)};
	m.roughness = std::clamp(m.roughness * m.roughnessTexture.scalar(u, v, lod), 0.f, 1.f);
	m.metallic = std::clamp(m.metallic * m.metallicTexture.scalar(u, v, lod), 0.f, 1.f);
	m.emission = mapped_emission(m, u, v);
	material_prepare(m);
	Vec3 n = hit.normal;
	Vec3 tangent = hit.dpdu - n * dot(n, hit.dpdu);

	if (tangent.length_sq() > 1e-20f) {
		tangent = normalize(tangent);
		float handedness = dot(cross(n, tangent), hit.dpdv) < 0.f ? -1.f : 1.f;
		Vec3 bitangent = cross(n, tangent) * handedness;

		if (m.normalTexture) {
			Vec3 t = m.normalTexture.sample(u, v, lod).rgb() * 2.f - Vec3(1.f);
			t.x *= m.normalScale;
			t.y *= m.normalScale;
			n = normalize(tangent * t.x + bitangent * t.y + n * t.z);
		}

		if (m.bumpTexture) {
			const auto &level = m.bumpTexture.image->levels[0];
			float du = 1.f / level.width, dv = 1.f / level.height;
			float dhdu =
				(m.bumpTexture.scalar(u + du, v, lod) - m.bumpTexture.scalar(u - du, v, lod)) /
				(2.f * du);
			float dhdv =
				(m.bumpTexture.scalar(u, v + dv, lod) - m.bumpTexture.scalar(u, v - dv, lod)) /
				(2.f * dv);
			// Dual surface basis converts derivatives per UV unit to world-space slope.
			Vec3 pu = hit.dpdu - n * dot(n, hit.dpdu), pv = hit.dpdv - n * dot(n, hit.dpdv);
			float a = dot(pu, pu), b = dot(pu, pv), c = dot(pv, pv), det = a * c - b * b;

			if (det > 1e-20f)
				n = normalize(n - m.bumpScale * (pu * ((c * dhdu - b * dhdv) / det) +
												 pv * ((a * dhdv - b * dhdu) / det)));
		}

		if (n.length_sq() > 0.f && dot(n, hit.geo_normal) > 0.f)
			hit.normal = n;
	}

	return m;
}

} // namespace xn
