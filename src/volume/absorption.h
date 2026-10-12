#pragma once
// Fixed-capacity active dielectric interiors; only transmission crosses a
// boundary. Removing an interior restores its enclosing medium's absorption.
#ifdef __CUDACC__
#define XN_MEDIUM_HD __host__ __device__
#else
#define XN_MEDIUM_HD
#endif
namespace xn {
template <class Color, int Capacity = 8> struct AbsorptionStack {
	Color coefficients[Capacity]{};
	int material_ids[Capacity]{};
	int depth = 0;

	XN_MEDIUM_HD Color current() const {
		return depth ? coefficients[depth - 1] : Color{};
	}

	XN_MEDIUM_HD Color after_leave(int material) const {
		if (depth && material_ids[depth - 1] == material)
			return depth > 1 ? coefficients[depth - 2] : Color{};

		return current();
	}

	XN_MEDIUM_HD bool enter(int material, Color absorption) {
		if (depth == Capacity)
			return false;
		coefficients[depth] = absorption;
		material_ids[depth] = material;
		++depth;

		return true;
	}

	XN_MEDIUM_HD void leave(int material) {
		for (int i = depth - 1; i >= 0; --i) {
			if (material_ids[i] != material)
				continue;

			for (int j = i; j + 1 < depth; ++j) {
				material_ids[j] = material_ids[j + 1];
				coefficients[j] = coefficients[j + 1];
			}
			--depth;

			return;
		}
	}
};
} // namespace xn

#undef XN_MEDIUM_HD
