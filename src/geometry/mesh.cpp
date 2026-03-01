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
bool load_obj(const std::string& path, TriangleMesh& mesh, int mat_id) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "[OBJ] Cannot open file: %s\n", path.c_str());
        return false;
    }

    // Temporary OBJ-space arrays
    std::vector<Vec3> positions;
    std::vector<Vec3> normals_obj;

    // Per-unique-combo (pos_idx, norm_idx) → mesh vertex index
    // We flatten OBJ's separate index arrays into a unified vertex buffer.
    struct VertKey { int pi, ni; };
    struct VertKeyHash {
        size_t operator()(VertKey k) const {
            return std::hash<int64_t>()(((int64_t)k.pi << 32) | (uint32_t)k.ni);
        }
        bool operator()(VertKey a, VertKey b) const { return a.pi==b.pi && a.ni==b.ni; }
    };

    // Simple flat map (small meshes) — for large meshes this is fine too
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

    mesh.mat_id = mat_id;
    bool has_normals = false;
    std::string line;
    positions.reserve(4096);
    normals_obj.reserve(4096);

    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        char type[8] = {};
        const char* p = line.c_str();
        int n = std::sscanf(p, "%7s", type);
        if (n < 1) continue;
        p += std::strlen(type);

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
        } else if (std::strcmp(type, "f") == 0) {
            // Parse face: each vertex is pos[/uv[/norm]] (1-based)
            // We ignore UV, support v, v/t, v//n, v/t/n
            std::vector<int32_t> face_verts;
            const char* q = p;
            while (*q) {
                while (*q == ' ' || *q == '\t') ++q;
                if (!*q || *q == '\n' || *q == '\r') break;
                int pi = -1, ti = -1, ni = -1;
                int consumed = 0;
                // Try v/t/n
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
                mesh.add_triangle(face_verts[0], face_verts[i], face_verts[i+1]);
        }
    }

    if (!has_normals) {
        mesh.compute_smooth_normals();
    }

    std::printf("[OBJ] Loaded %s: %d verts, %d tris\n",
        path.c_str(), mesh.num_vertices(), mesh.num_triangles());
    return true;
}

} // namespace xn
