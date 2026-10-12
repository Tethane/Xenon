#pragma once
#include <cstdint>

namespace xn {
#ifdef __CUDACC__
#define XENON_ALPHA_HD __host__ __device__
#else
#define XENON_ALPHA_HD
#endif
// One reproducible Bernoulli decision per candidate surface and query. The
// integer hash is independent of BVH traversal order and floating-point hits.
XENON_ALPHA_HD inline uint32_t opacity_hash(uint32_t x) {
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;

	return x ^ (x >> 16);
}

XENON_ALPHA_HD inline float opacity_random(uint32_t seed, int instance, int primitive,
										   bool front_face) {
	uint32_t h = opacity_hash(seed ^ 0x9e3779b9u);
	h = opacity_hash(h ^ uint32_t(instance));
	h = opacity_hash(h ^ uint32_t(primitive));
	h = opacity_hash(h ^ (front_face ? 0x243f6a88u : 0xb7e15162u));

	return float(h >> 8) * (1.f / 16777216.f);
}

#undef XENON_ALPHA_HD
} // namespace xn
