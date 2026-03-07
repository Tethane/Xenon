// geometry/mesh.cpp — TriangleMesh smooth normal computation + OBJ loader

#include "geometry/mesh.h"
#include <cassert>
#include <unordered_map>
#include <cstring>
#include <fstream>
#include <cstdio>

namespace xn {

void TriangleMesh::compute_smooth_normals() {
    int nv = num_vertices();
    int nt = num_triangles();
    std::fill(nx.begin(), nx.end(), 0.f);
    std::fill(ny.begin(), ny.end(), 0.f);
    std::fill(nz.begin(), nz.end(), 0.f);

    for (int t = 0; t < nt; ++t) {
        int i0 = indices[t*3+0], i1 = indices[t*3+1], i2 = indices[t*3+2];
        Vec3 p0{vx[i0],vy[i0],vz[i0]};
        Vec3 p1{vx[i1],vy[i1],vz[i1]};
        Vec3 p2{vx[i2],vy[i2],vz[i2]};
        Vec3 e1 = p1-p0, e2 = p2-p0;
        Vec3 fn = cross(e1, e2); // area-weighted normal
        nx[i0]+=fn.x; ny[i0]+=fn.y; nz[i0]+=fn.z;
        nx[i1]+=fn.x; ny[i1]+=fn.y; nz[i1]+=fn.z;
        nx[i2]+=fn.x; ny[i2]+=fn.y; nz[i2]+=fn.z;
    }
    for (int i = 0; i < nv; ++i) {
        Vec3 n = normalize({nx[i], ny[i], nz[i]});
        nx[i]=n.x; ny[i]=n.y; nz[i]=n.z;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OBJ Loader
// Supports: v, vn, f (triangles and quads via fan triangulation)
// Indices are 1-based in OBJ; we convert to 0-based here.
// If the OBJ has vertex normals, we use them; otherwise compute smooth normals.
// ─────────────────────────────────────────────────────────────────────────────
bool load_obj(const std::string& path, TriangleMesh& mesh, 
              const std::map<std::string, int>& mat_map, int default_mat_id) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "[OBJ] Cannot open file: %s\n", path.c_str());
        return false;
    }

    // Temporary OBJ-space arrays
    std::vector<Vec3> positions;
    std::vector<Vec3> normals_obj;

    // Per-unique-combo (pos_idx, norm_idx) → mesh vertex index
    std::unordered_map<int64_t, int32_t> vert_cache;
    vert_cache.reserve(4096);

    auto get_or_add_vert = [&](int pi, int ni) -> int32_t {
        int64_t key = ((int64_t)pi << 32) | (uint32_t)ni;
        auto it = vert_cache.find(key);
        if (it != vert_cache.end()) return it->second;
        int32_t idx = (int32_t)mesh.vx.size();
        Vec3 p = positions[pi];
        mesh.vx.push_back(p.x); mesh.vy.push_back(p.y); mesh.vz.push_back(p.z);
        if (ni >= 0 && ni < (int)normals_obj.size()) {
            Vec3 n = normals_obj[ni];
            mesh.nx.push_back(n.x); mesh.ny.push_back(n.y); mesh.nz.push_back(n.z);
        } else {
            mesh.nx.push_back(0); mesh.ny.push_back(0); mesh.nz.push_back(0);
        }
        vert_cache[key] = idx;
        return idx;
    };

    mesh.mat_id = default_mat_id;
    int current_mat_id = default_mat_id;
    bool has_normals = false;
    std::string line;
    positions.reserve(4096);
    normals_obj.reserve(4096);

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        char type[8] = {};
        const char* p_cstr = line.c_str();
        int n = std::sscanf(p_cstr, "%7s", type);
        if (n < 1) continue;
        const char* p = p_cstr + std::strlen(type);

        if (std::strcmp(type, "v") == 0) {
            float x, y, z;
            if (std::sscanf(p, " %f %f %f", &x, &y, &z) == 3)
                positions.push_back({x, y, z});
        } else if (std::strcmp(type, "vn") == 0) {
            float x, y, z;
            if (std::sscanf(p, " %f %f %f", &x, &y, &z) == 3) {
                normals_obj.push_back(normalize({x, y, z}));
                has_normals = true;
            }
        } else if (std::strcmp(type, "usemtl") == 0) {
            char mat_name[128];
            if (std::sscanf(p, " %127s", mat_name) == 1) {
                auto it = mat_map.find(mat_name);
                if (it != mat_map.end()) {
                    current_mat_id = it->second;
                }
            }
        } else if (std::strcmp(type, "f") == 0) {
            // Parse face: each vertex is pos[/uv[/norm]] (1-based)
            std::vector<int32_t> face_verts;
            const char* q = p;
            while (*q) {
                while (*q == ' ' || *q == '\t') ++q;
                if (!*q || *q == '\n' || *q == '\r') break;
                int pi = -1, ti = -1, ni = -1;
                int consumed = 0;
                if (std::sscanf(q, "%d/%d/%d%n", &pi, &ti, &ni, &consumed) >= 3) {}
                else if (std::sscanf(q, "%d//%d%n", &pi, &ni, &consumed) >= 2) {}
                else if (std::sscanf(q, "%d/%d%n", &pi, &ti, &consumed) >= 2) {}
                else if (std::sscanf(q, "%d%n", &pi, &consumed) >= 1) {}
                else { break; }
                if (pi > 0) {
                    --pi;
                    if (ni > 0) --ni; else ni = -1;
                    face_verts.push_back(get_or_add_vert(pi, ni));
                }
                q += consumed;
            }
            // Fan triangulation
            for (int i = 1; i+1 < (int)face_verts.size(); ++i)
                mesh.add_triangle(face_verts[0], face_verts[i], face_verts[i+1], current_mat_id);
        }
    }

    if (!has_normals) {
        mesh.compute_smooth_normals();
    }

    std::printf("[OBJ] Loaded %s: %d verts, %d tris\n",
        path.c_str(), mesh.num_vertices(), mesh.num_triangles());
    
    // Debug: Compute bounds
    Vec3 vmin(1e30f), vmax(-1e30f);
    for (int i = 0; i < mesh.num_vertices(); ++i) {
        vmin.x = std::min(vmin.x, mesh.vx[i]); vmin.y = std::min(vmin.y, mesh.vy[i]); vmin.z = std::min(vmin.z, mesh.vz[i]);
        vmax.x = std::max(vmax.x, mesh.vx[i]); vmax.y = std::max(vmax.y, mesh.vy[i]); vmax.z = std::max(vmax.z, mesh.vz[i]);
    }
    std::printf("      Bounds: min(%.2f, %.2f, %.2f) max(%.2f, %.2f, %.2f)\n", vmin.x, vmin.y, vmin.z, vmax.x, vmax.y, vmax.z);
    
    return true;
}
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void TriangleMesh::transform(Vec3 pos, float scale, Vec3 rot_deg) {
    Vec3 rad = rot_deg * (M_PI / 180.f);
    float cx = std::cos(rad.x), sx = std::sin(rad.x);
    float cy = std::cos(rad.y), sy = std::sin(rad.y);
    float cz = std::cos(rad.z), sz = std::sin(rad.z);

    // Vertex positions
    for (size_t i = 0; i < vx.size(); ++i) {
        Vec3 v(vx[i], vy[i], vz[i]);
        v *= scale;
        
        // Rotation Y (Yaw)
        float nx = v.x * cy + v.z * sy;
        float nz = -v.x * sy + v.z * cy;
        v.x = nx; v.z = nz;
        // Rotation X (Pitch)
        float ny = v.y * cx - v.z * sx;
        nz = v.y * sx + v.z * cx;
        v.y = ny; v.z = nz;
        // Rotation Z (Roll)
        nx = v.x * cz - v.y * sz;
        ny = v.x * sz + v.y * cz;
        v.x = nx; v.y = ny;

        v += pos;
        vx[i] = v.x; vy[i] = v.y; vz[i] = v.z;
    }

    // Vertex normals (only rotate and re-normalize)
    if (!nx.empty()) {
        for (size_t i = 0; i < nx.size(); ++i) {
            Vec3 n(nx[i], ny[i], nz[i]);
            // Y
            float tx = n.x * cy + n.z * sy;
            float tz = -n.x * sy + n.z * cy;
            n.x = tx; n.z = tz;
            // X
            float ty = n.y * cx - n.z * sx;
            tz = n.y * sx + n.z * cx;
            n.y = ty; n.z = tz;
            // Z
            tx = n.x * cz - n.y * sz;
            ty = n.x * sz + n.y * cz;
            n.x = tx; n.y = ty;

            n = normalize(n);
            nx[i] = n.x; ny[i] = n.y; nz[i] = n.z;
        }
    }
}

void TriangleMesh::merge(const TriangleMesh& other) {
    int v_offset = (int)vx.size();
    vx.insert(vx.end(), other.vx.begin(), other.vx.end());
    vy.insert(vy.end(), other.vy.begin(), other.vy.end());
    vz.insert(vz.end(), other.vz.begin(), other.vz.end());
    
    // Normal merge: if one has normals and other doesn't, this becomes tricky.
    // However, xenon's load_obj ensures normals exist (computed if missing).
    if (!other.nx.empty()) {
        if (nx.empty()) {
            nx.resize(v_offset, 0.f); ny.resize(v_offset, 0.f); nz.resize(v_offset, 0.f);
        }
        nx.insert(nx.end(), other.nx.begin(), other.nx.end());
        ny.insert(ny.end(), other.ny.begin(), other.ny.end());
        nz.insert(nz.end(), other.nz.begin(), other.nz.end());
    } else if (!nx.empty()) {
        nx.resize(vx.size(), 0.f); ny.resize(vy.size(), 0.f); nz.resize(vz.size(), 0.f);
    }

    for (size_t i = 0; i < other.indices.size(); ++i) {
        indices.push_back(other.indices[i] + v_offset);
    }
    mat_ids.insert(mat_ids.end(), other.mat_ids.begin(), other.mat_ids.end());
    
    std::printf("[Mesh] Merged: total %d verts, %d tris\n", (int)vx.size(), (int)indices.size()/3);
}

} // namespace xn