// geometry/mesh.cpp — TriangleMesh smooth normal computation + OBJ loader

#include "geometry/mesh.h"
#include <cassert>
#include <unordered_map>
#include <cstring>
#include <fstream>
#include <cstdio>
#include <sstream>
#include <charconv>
#include <array>

namespace xn {

void TriangleMesh::compute_smooth_normals() {
	int nv = num_vertices();
	int nt = num_triangles();
	nx.assign(nv, 0.f);
	ny.assign(nv, 0.f);
	nz.assign(nv, 0.f);

	for (int t = 0; t < nt; ++t) {
		int i0 = indices[t * 3 + 0], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
		Vec3 p0{vx[i0], vy[i0], vz[i0]};
		Vec3 p1{vx[i1], vy[i1], vz[i1]};
		Vec3 p2{vx[i2], vy[i2], vz[i2]};
		Vec3 e1 = p1 - p0, e2 = p2 - p0;
		Vec3 fn = cross(e1, e2); // area-weighted normal
		nx[i0] += fn.x;
		ny[i0] += fn.y;
		nz[i0] += fn.z;
		nx[i1] += fn.x;
		ny[i1] += fn.y;
		nz[i1] += fn.z;
		nx[i2] += fn.x;
		ny[i2] += fn.y;
		nz[i2] += fn.z;
	}

	for (int i = 0; i < nv; ++i) {
		Vec3 n = normalize({nx[i], ny[i], nz[i]});
		nx[i] = n.x;
		ny[i] = n.y;
		nz[i] = n.z;
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// OBJ Loader
// Supports: v, vn, f (triangles and quads via fan triangulation)
// Indices are 1-based in OBJ; we convert to 0-based here.
// If the OBJ has vertex normals, we use them; otherwise compute smooth normals.
// ─────────────────────────────────────────────────────────────────────────────
bool load_obj(const std::string &path, TriangleMesh &mesh,
			  const std::map<std::string, int> &mat_map, int default_mat_id) {
	std::ifstream f(path);

	if (!f.is_open()) {
		std::fprintf(stderr, "[OBJ] Cannot open file: %s\n", path.c_str());

		return false;
	}

	mesh = TriangleMesh{};

	// Temporary OBJ-space arrays
	std::vector<Vec3> positions;
	std::vector<Vec3> normals_obj;
	std::vector<std::array<float, 2>> uv_obj;
	std::vector<int> source_positions;

	// Per-unique-combo (pos_idx, norm_idx) → mesh vertex index
	struct Key {
		int p, t, n;
		bool operator==(const Key &) const = default;
	};

	struct Hash {
		size_t operator()(Key k) const {
			return size_t(k.p) * 73856093u ^ size_t(k.t) * 19349663u ^ size_t(k.n) * 83492791u;
		}
	};

	std::unordered_map<Key, int32_t, Hash> vert_cache;
	vert_cache.reserve(4096);

	auto get_or_add_vert = [&](int pi, int ti, int ni) -> int32_t {
		Key key{pi, ti, ni};
		auto it = vert_cache.find(key);

		if (it != vert_cache.end())
			return it->second;
		int32_t idx = (int32_t)mesh.vx.size();
		Vec3 p = positions[pi];
		mesh.vx.push_back(p.x);
		mesh.vy.push_back(p.y);
		mesh.vz.push_back(p.z);

		if (ni >= 0 && ni < (int)normals_obj.size()) {
			Vec3 n = normals_obj[ni];
			mesh.nx.push_back(n.x);
			mesh.ny.push_back(n.y);
			mesh.nz.push_back(n.z);
		} else {
			mesh.nx.push_back(0);
			mesh.ny.push_back(0);
			mesh.nz.push_back(0);
		}
		mesh.tu.push_back(ti >= 0 ? uv_obj[ti][0] : 0.f);
		mesh.tv.push_back(ti >= 0 ? uv_obj[ti][1] : 0.f);
		mesh.uv_valid.push_back(ti >= 0);
		source_positions.push_back(pi);
		vert_cache[key] = idx;

		return idx;
	};

	mesh.mat_id = default_mat_id;
	int current_mat_id = default_mat_id;

	std::string line;
	positions.reserve(4096);
	normals_obj.reserve(4096);

	while (std::getline(f, line)) {
		std::istringstream ss(line);
		std::string type;

		if (!(ss >> type) || type[0] == '#')
			continue;
		auto fail = [&]() {
			std::fprintf(stderr, "[OBJ] Invalid %s record in %s\n", type.c_str(), path.c_str());

			return false;
		};

		if (type == "v" || type == "vn") {
			Vec3 v;

			if (!(ss >> v.x >> v.y >> v.z) || !std::isfinite(v.x) || !std::isfinite(v.y) ||
				!std::isfinite(v.z))
				return fail();

			if (type == "v")
				positions.push_back(v);
			else
				normals_obj.push_back(normalize(v));
		} else if (type == "vt") {
			float u, v = 0.f;

			if (!(ss >> u) || !std::isfinite(u))
				return fail();

			if (!ss.eof())
				ss >> std::ws;

			if (!ss.eof() && ss.peek() != '#') {
				if (!(ss >> v) || !std::isfinite(v))
					return fail();
			}
			uv_obj.push_back({u, v});
		} else if (type == "usemtl") {
			std::string name;

			if (!(ss >> name))
				return fail();
			auto it = mat_map.find(name);

			if (it == mat_map.end()) {
				std::fprintf(stderr, "[OBJ] %s: unknown material '%s'\n", path.c_str(),
							 name.c_str());

				return false;
			}
			current_mat_id = it->second;
		} else if (type == "f") {
			std::vector<int32_t> face_verts;
			std::string token;

			while (ss >> token) {
				if (token[0] == '#')
					break;
				int ids[3] = {-1, -1, -1};
				int counts[3] = {(int)positions.size(), (int)uv_obj.size(),
								 (int)normals_obj.size()};
				size_t start = 0;

				for (int part = 0;; ++part) {
					if (part == 3)
						return fail();
					size_t end = token.find('/', start);

					if (end == std::string::npos)
						end = token.size();

					if (end > start) {
						int index;
						auto parsed =
							std::from_chars(token.data() + start, token.data() + end, index);

						if (parsed.ec != std::errc{} || parsed.ptr != token.data() + end ||
							index == 0)
							return fail();
						ids[part] = index > 0 ? index - 1 : counts[part] + index;

						if (ids[part] < 0 || ids[part] >= counts[part])
							return fail();
					} else if (part != 1 || end == token.size())
						return fail();

					if (end == token.size())
						break;
					start = end + 1;
				}

				if (ids[0] < 0)
					return fail();
				face_verts.push_back(get_or_add_vert(ids[0], ids[1], ids[2]));
			}

			if (face_verts.size() < 3)
				return fail();

			if (face_verts.size() == 3) {
				mesh.add_triangle(face_verts[0], face_verts[1], face_verts[2], current_mat_id);
				continue;
			}
			auto point = [&](int i) { return Vec3(mesh.vx[i], mesh.vy[i], mesh.vz[i]); };
			Vec3 normal(0.f);

			for (size_t i = 0; i < face_verts.size(); ++i)
				normal +=
					cross(point(face_verts[i]), point(face_verts[(i + 1) % face_verts.size()]));
			int axis = max_axis(abs3(normal)), u = (axis + 1) % 3, v = (axis + 2) % 3;

			if (normal[axis] == 0.f)
				return fail();
			double sign = normal[axis] > 0.f ? 1.0 : -1.0;
			auto turn = [&](int ia, int ib, int ic) {
				Vec3 a = point(ia), b = point(ib), c = point(ic);

				return sign * ((double(b[u]) - a[u]) * (double(c[v]) - a[v]) -
							   (double(b[v]) - a[v]) * (double(c[u]) - a[u]));
			};
			bool convex = true;

			for (size_t i = 0; i < face_verts.size(); ++i)
				convex &= turn(face_verts[i], face_verts[(i + 1) % face_verts.size()],
							   face_verts[(i + 2) % face_verts.size()]) > 0.0;

			if (convex) {
				for (size_t i = 1; i + 1 < face_verts.size(); ++i)
					mesh.add_triangle(face_verts[0], face_verts[i], face_verts[i + 1],
									  current_mat_id);
				continue;
			}
			while (face_verts.size() > 3) {
				bool clipped = false;

				for (size_t i = 0; i < face_verts.size(); ++i) {
					int a = face_verts[(i + face_verts.size() - 1) % face_verts.size()];
					int b = face_verts[i], c = face_verts[(i + 1) % face_verts.size()];

					if (turn(a, b, c) <= 0.0)
						continue;
					bool contains = false;

					for (int j : face_verts) {
						if (j == a || j == b || j == c)
							continue;

						if (turn(a, b, j) >= 0.0 && turn(b, c, j) >= 0.0 && turn(c, a, j) >= 0.0) {
							contains = true;
							break;
						}
					}

					if (contains)
						continue;
					mesh.add_triangle(a, b, c, current_mat_id);
					face_verts.erase(face_verts.begin() + i);
					clipped = true;
					break;
				}

				if (!clipped)
					return fail();
			}
			mesh.add_triangle(face_verts[0], face_verts[1], face_verts[2], current_mat_id);
		}
	}

	if (f.bad() || mesh.num_triangles() == 0) {
		std::fprintf(stderr, "[OBJ] Failed to read triangles: %s\n", path.c_str());

		return false;
	}

	auto nx = mesh.nx, ny = mesh.ny, nz = mesh.nz;
	// Accumulate generated normals by OBJ position, before UV seam duplication.
	// A UV seam alone must not introduce a geometric shading discontinuity.
	std::vector<Vec3> generated(positions.size(), Vec3(0.f));

	for (int t = 0; t < mesh.num_triangles(); ++t) {
		Vec3 a, b, c;
		mesh.get_triangle(t, a, b, c);
		Vec3 face = cross(b - a, c - a);

		for (int k = 0; k < 3; ++k)
			generated[source_positions[mesh.indices[3 * t + k]]] += face;
	}

	for (int i = 0; i < mesh.num_vertices(); ++i) {
		Vec3 n = normalize(generated[source_positions[i]]);
		mesh.nx[i] = n.x;
		mesh.ny[i] = n.y;
		mesh.nz[i] = n.z;

		if (Vec3(nx[i], ny[i], nz[i]).length_sq() > 0.f) {
			mesh.nx[i] = nx[i];
			mesh.ny[i] = ny[i];
			mesh.nz[i] = nz[i];
		}
	}

	std::printf("[OBJ] Loaded %s: %d verts, %d tris\n", path.c_str(), mesh.num_vertices(),
				mesh.num_triangles());

	// Debug: Compute bounds
	Vec3 vmin(1e30f), vmax(-1e30f);

	for (int i = 0; i < mesh.num_vertices(); ++i) {
		vmin.x = std::min(vmin.x, mesh.vx[i]);
		vmin.y = std::min(vmin.y, mesh.vy[i]);
		vmin.z = std::min(vmin.z, mesh.vz[i]);
		vmax.x = std::max(vmax.x, mesh.vx[i]);
		vmax.y = std::max(vmax.y, mesh.vy[i]);
		vmax.z = std::max(vmax.z, mesh.vz[i]);
	}
	std::printf("    Bounds: min(%.2f, %.2f, %.2f) max(%.2f, %.2f, %.2f)\n", vmin.x, vmin.y, vmin.z,
				vmax.x, vmax.y, vmax.z);

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
		v.x = nx;
		v.z = nz;
		// Rotation X (Pitch)
		float ny = v.y * cx - v.z * sx;
		nz = v.y * sx + v.z * cx;
		v.y = ny;
		v.z = nz;
		// Rotation Z (Roll)
		nx = v.x * cz - v.y * sz;
		ny = v.x * sz + v.y * cz;
		v.x = nx;
		v.y = ny;

		v += pos;
		vx[i] = v.x;
		vy[i] = v.y;
		vz[i] = v.z;
	}

	if (scale < 0.f) {
		for (size_t i = 0; i < indices.size(); i += 3)
			std::swap(indices[i + 1], indices[i + 2]);

		for (size_t i = 0; i < nx.size(); ++i) {
			nx[i] = -nx[i];
			ny[i] = -ny[i];
			nz[i] = -nz[i];
		}
	}

	// Vertex normals (only rotate and re-normalize)
	if (!nx.empty()) {
		for (size_t i = 0; i < nx.size(); ++i) {
			Vec3 n(nx[i], ny[i], nz[i]);
			// Y
			float tx = n.x * cy + n.z * sy;
			float tz = -n.x * sy + n.z * cy;
			n.x = tx;
			n.z = tz;
			// X
			float ty = n.y * cx - n.z * sx;
			tz = n.y * sx + n.z * cx;
			n.y = ty;
			n.z = tz;
			// Z
			tx = n.x * cz - n.y * sz;
			ty = n.x * sz + n.y * cz;
			n.x = tx;
			n.y = ty;

			n = normalize(n);
			nx[i] = n.x;
			ny[i] = n.y;
			nz[i] = n.z;
		}
	}
}

void TriangleMesh::merge(const TriangleMesh &other) {
	int v_offset = (int)vx.size();
	vx.insert(vx.end(), other.vx.begin(), other.vx.end());
	vy.insert(vy.end(), other.vy.begin(), other.vy.end());
	vz.insert(vz.end(), other.vz.begin(), other.vz.end());

	// Normal merge: if one has normals and other doesn't, this becomes tricky.
	// However, xenon's load_obj ensures normals exist (computed if missing).
	if (!other.nx.empty()) {
		if (nx.empty()) {
			nx.resize(v_offset, 0.f);
			ny.resize(v_offset, 0.f);
			nz.resize(v_offset, 0.f);
		}
		nx.insert(nx.end(), other.nx.begin(), other.nx.end());
		ny.insert(ny.end(), other.ny.begin(), other.ny.end());
		nz.insert(nz.end(), other.nz.begin(), other.nz.end());
	} else if (!nx.empty()) {
		nx.resize(vx.size(), 0.f);
		ny.resize(vy.size(), 0.f);
		nz.resize(vz.size(), 0.f);
	}

	tu.resize(v_offset);
	tv.resize(v_offset);
	uv_valid.resize(v_offset);

	if (other.tu.size() == other.vx.size()) {
		tu.insert(tu.end(), other.tu.begin(), other.tu.end());
		tv.insert(tv.end(), other.tv.begin(), other.tv.end());
		uv_valid.insert(uv_valid.end(), other.uv_valid.begin(), other.uv_valid.end());
	} else {
		tu.resize(vx.size());
		tv.resize(vx.size());
		uv_valid.resize(vx.size());
	}

	for (size_t i = 0; i < other.indices.size(); ++i) {
		indices.push_back(other.indices[i] + v_offset);
	}
	mat_ids.insert(mat_ids.end(), other.mat_ids.begin(), other.mat_ids.end());

	std::printf("[Mesh] Merged: total %d verts, %d tris\n", (int)vx.size(),
				(int)indices.size() / 3);
}

} // namespace xn
