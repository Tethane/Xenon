// material/material.cpp — .mat file parser and Material cache

#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
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
static std::string strip_quotes(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parse a .mat file into a Material
// ─────────────────────────────────────────────────────────────────────────────
static bool parse_mat_file(const std::string& path, Material& mat) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "[Material] Failed to open: %s\n", path.c_str());
        return false;
    }

    std::string line;
    int line_num = 0;

    while (std::getline(f, line)) {
        ++line_num;

        // Skip empty lines and comments
        size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#')
            continue;

        std::istringstream ss(line);
        std::string key;
        ss >> key;

        if (key == "name") {
            std::string val;
            ss >> val;
            mat.name = strip_quotes(val);
        } else if (key == "type") {
            std::string val;
            ss >> val;
            // Currently only "principled" is supported
            val = strip_quotes(val);
            if (val != "principled") {
                std::fprintf(stderr, "[Material] %s:%d: unsupported type '%s' (using principled)\n",
                             path.c_str(), line_num, val.c_str());
            }
        } else if (key == "baseColor") {
            ss >> mat.baseColor.x >> mat.baseColor.y >> mat.baseColor.z;
        } else if (key == "roughness") {
            ss >> mat.roughness;
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
        } else if (key == "clearcoat") {
            ss >> mat.clearcoat;
        } else if (key == "clearcoatRoughness") {
            ss >> mat.clearcoatRoughness;
        } else if (key == "anisotropy") {
            ss >> mat.anisotropy;
        } else if (key == "emission") {
            ss >> mat.emission.x >> mat.emission.y >> mat.emission.z;
        } else if (key == "emissionTemperature") {
            ss >> mat.emissionTemperature;
        } else {
            std::fprintf(stderr, "[Material] %s:%d: unknown key '%s'\n",
                         path.c_str(), line_num, key.c_str());
        }

        if (ss.fail() && !ss.eof()) {
            std::fprintf(stderr, "[Material] %s:%d: parse error for key '%s'\n",
                         path.c_str(), line_num, key.c_str());
        }
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// load_material — public API, cached
// ─────────────────────────────────────────────────────────────────────────────
Material load_material(const std::string& path) {
    {
        std::lock_guard<std::mutex> lock(s_cache_mutex);
        auto it = s_material_cache.find(path);
        if (it != s_material_cache.end())
            return it->second;
    }

    Material mat;
    if (!parse_mat_file(path, mat)) {
        std::fprintf(stderr, "[Material] Using defaults for '%s'\n", path.c_str());
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
Material material_from_legacy(const std::string& name, Vec3 albedo, float metallic,
                              float roughness, float specular, float ior,
                              float transmission, float transmission_roughness) {
    Material m;
    m.name        = name;
    m.baseColor   = albedo;
    m.metallic    = metallic;
    m.roughness   = roughness;
    m.ior         = ior;
    m.transmission = transmission;
    // Legacy: specular scaled IOR-based F0 (approximate)
    (void)specular;
    // If transmission_roughness was set (>= 0), use it for roughness on transmissive lobes
    if (transmission_roughness >= 0.0f && transmission > 0.01f) {
        m.roughness = transmission_roughness;
    }
    material_prepare(m);
    return m;
}

} // namespace xn
