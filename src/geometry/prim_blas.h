#pragma once
// geometry/prim_blas.h — Bottom-level BVH over analytic geometric primitives
//
// PrimBLAS mirrors the BLAS API exactly:
//   • intersect(ray, rec) — closest-hit
//   • intersects(ray)     — any-hit (shadow)
//   • root_aabb()         — root bounding box (called by TLAS Instance)
//
// PrimGroup: a flat list of analytic primitives from primitives.h.
//   All types (Sphere, Plane, Box, Disk, Quad) can coexist in one group.
//   PrimBLAS builds a binned-SAH BVH over their bounding boxes.
//
// Integration:
//   PrimBLAS is plugged into the TLAS via GeomHandle (see tlas.h).
//   Scene::build_acceleration() creates one PrimBLAS over all_prims.
//
// Notes:
//   • Unbounded plane groups use an unsplit leaf to preserve infinite support.
//   • prim_id is populated by PrimBLAS traversal = index in PrimGroup.

#include <cstdint>
#include <variant>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/primitives.h"
#include "math/ray.h"

namespace xn {

// ─── PrimVariant — tagged union over all primitive types ─────────────────────
using PrimVariant = std::variant<Sphere, Plane, Box, Disk, Quad>;

// Returns the AABB for any PrimVariant.
inline AABB prim_aabb(const PrimVariant &p) {
	return std::visit([](const auto &v) { return v.aabb(); }, p);
}

// Returns the mat_id for any PrimVariant.
inline int prim_mat_id(const PrimVariant &p) {
	return std::visit([](const auto &v) -> int { return v.mat_id; }, p);
}

// Intersect any PrimVariant, setting rec.prim_id = idx.
// Returns true on hit.
inline bool prim_intersect(const PrimVariant &p, const Ray &ray, HitRecord &rec, int idx) {
	bool hit = std::visit([&](const auto &v) { return v.intersect(ray, rec); }, p);

	if (hit)
		rec.prim_id = idx;

	return hit;
}

inline bool prim_intersect_filtered(const PrimVariant &p, const Ray &ray, HitRecord &rec, int idx,
									const HitFilter *filter) {
	if (!filter)
		return prim_intersect(p, ray, rec, idx);
	Ray query = ray;
	query.tmax = std::min(ray.tmax, rec.t);
	// The supported convex analytic primitives have at most two intersections.
	for (int root = 0; root < 2; ++root) {
		HitRecord candidate;
		candidate.t = query.tmax;

		if (!prim_intersect(p, query, candidate, idx))
			return false;

		if ((*filter)(candidate, ray.alpha_seed)) {
			rec = candidate;

			return true;
		}
		query.tmin = std::nextafter(candidate.t, kInfinity);
	}

	return false;
}

// ─── PrimGroup — a flat collection of primitives ─────────────────────────────
struct PrimGroup {
	std::vector<PrimVariant> prims;

	bool empty() const noexcept {
		return prims.empty();
	}

	int size() const noexcept {
		return static_cast<int>(prims.size());
	}

	template <typename T> void add(T prim) {
		prims.emplace_back(std::move(prim));
	}
};

// ─── PrimBLASNode — 32 bytes ──────────────────────────────────────────────────
struct alignas(32) PrimBLASNode {
	AABB bbox;
	uint32_t child; // internal: left child; leaf: first prim slot
	uint32_t count; // 0 = internal, >0 = leaf

	bool is_leaf() const noexcept {
		return count > 0;
	}
};

static_assert(sizeof(PrimBLASNode) == 32, "PrimBLASNode must be 32 bytes");

// ─── PrimBLAS ─────────────────────────────────────────────────────────────────
class PrimBLAS {
  public:
	PrimBLAS() = default;
	~PrimBLAS() = default;

	PrimBLAS(const PrimBLAS &) = delete;
	PrimBLAS &operator=(const PrimBLAS &) = delete;
	PrimBLAS(PrimBLAS &&) = default;
	PrimBLAS &operator=(PrimBLAS &&) = default;

	// Build a binned-SAH BVH over all primitives in group.
	void build(const PrimGroup &group);

	// Root AABB in local (world) space — used by TLAS Instance.
	const AABB &root_aabb() const noexcept;

	// Closest-hit — updates rec on success.
	[[nodiscard]] bool intersect(const Ray &ray, HitRecord &rec,
								 const HitFilter *filter = nullptr) const;

	// Any-hit — fast early exit for shadow rays.
	[[nodiscard]] bool intersects(const Ray &ray, const HitFilter *filter = nullptr) const;

	const PrimGroup &group() const {
		return *group_;
	}

  private:
	const PrimGroup *group_ = nullptr;
	std::vector<PrimBLASNode> nodes_;
	std::vector<uint32_t> prims_; // prims_[i] → index into group_->prims

	struct PrimEntry {
		AABB bbox;
		Vec3 center;
		uint32_t idx;
	};

	uint32_t alloc_node();
	void subdivide(uint32_t node, uint32_t lo, uint32_t hi, std::vector<PrimEntry> &ps,
				   int depth = 0);
	void make_leaf(uint32_t node, uint32_t lo, uint32_t hi, std::vector<PrimEntry> &ps);
	void refit_bounds(uint32_t node, uint32_t lo, uint32_t hi, const std::vector<PrimEntry> &ps);
};

} // namespace xn
