#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define XENON_SEED_HD __host__ __device__
#else
#define XENON_SEED_HD
#endif

namespace xn {
// SplitMix64 finalizer, including its Weyl increment. Used only to decorrelate
// path initialization; PCG remains the generator for subsequent path variates.
XENON_SEED_HD inline uint64_t mix_path_seed(uint64_t value) {
	value += 0x9e3779b97f4a7c15ULL;
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
	value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;

	return value ^ (value >> 31);
}

struct PathSeed {
	uint64_t state, stream;
};

XENON_SEED_HD inline PathSeed make_path_seed(uint32_t pixel, uint32_t sample, uint64_t user_seed) {
	// Packing preserves distinct pixel/sample pairs. Hash all user-seed bits
	// before PCG's 63-bit stream selection, including the high user-seed bit.
	const uint64_t key = (uint64_t(sample) << 32) | uint64_t(pixel);
	const uint64_t state = mix_path_seed(key ^ mix_path_seed(user_seed));

	return {state, mix_path_seed(state ^ 0xd2b74407b1ce6e93ULL)};
}
} // namespace xn

#undef XENON_SEED_HD
